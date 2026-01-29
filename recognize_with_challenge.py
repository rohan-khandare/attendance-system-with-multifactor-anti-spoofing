# # recognize_with_challenge.py
# import cv2
# import numpy as np
# import os
# import time
# import random
# from insightface.app import FaceAnalysis
# from sklearn.metrics.pairwise import cosine_similarity
# from attendance_logger import log_attendance, init_db
# from anti_spoof import (LivenessTracker,
#                         eye_aspect_ratio,
#                         compute_optical_flow,
#                         laplacian_variance,
#                         is_live_simple,
#                         LEFT_EYE_IDX,
#                         RIGHT_EYE_IDX)
# import mediapipe as mp

# # -----------------------
# # Challenge configuration
# # -----------------------
# CHALLENGE_TIMEOUT = 5.0       # seconds to complete the challenge
# BLINK_EAR_THRESH = 0.20      # EAR threshold for blink
# TURN_RATIO_THRESH = 0.06     # normalized shift threshold for head turn detection
# DEPTH_VAR_THRESH = 0.002     # threshold: low => likely flat (photo/screen)
# FLOW_THRESH = 0.25
# LAP_THRESH = 40.0

# # Log file for spoof events
# SPOOF_LOG = "spoof_log.txt"

# # -----------------------
# # Simple tracker (centroid/IOU)
# # -----------------------
# class Track:
#     def __init__(self, tid, bbox):
#         self.tid = tid
#         self.bbox = bbox
#         self.missing = 0
#         self.history = LivenessTracker(maxlen=40)
#         self.name = None
#         self.best_score = 0.0
#         self.logged = False
#         # challenge state
#         self.challenge = None
#         self.challenge_start = None
#         self.challenge_baseline = None  # baseline metric (for head-turn)
#         self.challenge_passed = False

# def iou(a, b):
#     ax1, ay1, ax2, ay2 = a
#     bx1, by1, bx2, by2 = b
#     xa = max(ax1, bx1); ya = max(ay1, by1)
#     xb = min(ax2, bx2); yb = min(ay2, by2)
#     inter = max(0, xb - xa) * max(0, yb - ya)
#     areaA = max(1, (ax2 - ax1) * (ay2 - ay1))
#     areaB = max(1, (bx2 - bx1) * (by2 - by1))
#     union = areaA + areaB - inter
#     return inter / union if union > 0 else 0

# # -----------------------
# # Recognizer with challenge-response and depth check
# # -----------------------
# class RealTimeRecognizerChallenge:
#     def __init__(self, db_file="templates.npz", threshold=0.35):
#         self.db_file = db_file
#         self.threshold = threshold
#         self._load_db()

#         print("[INFO] Initializing FaceAnalysis model...")
#         self.app = FaceAnalysis(name='buffalo_l')
#         self.app.prepare(ctx_id=0, det_size=(640, 640))
#         print("[INFO] Model loaded successfully.")

#         # MediaPipe face mesh for landmarks (used for EAR + depth + baseline)
#         self.mp_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
#                                                        max_num_faces=1,
#                                                        refine_landmarks=True,
#                                                        min_detection_confidence=0.5,
#                                                        min_tracking_confidence=0.5)
#         self.tracks = []
#         self.next_tid = 0
#         self.prev_gray = None

#     def _load_db(self):
#         if os.path.exists(self.db_file):
#             data = np.load(self.db_file, allow_pickle=True)
#             self.embeddings = np.array(data["embeddings"].tolist())
#             self.names = data["names"].tolist()
#             print(f"[INFO] Loaded {len(self.names)} enrolled users.")
#         else:
#             raise FileNotFoundError("templates.npz not found. Please enroll users first.")

#     def _issue_challenge(self, track, frame, landmarks_xy):
#         """Pick a random challenge and set baseline state if needed."""
#         challenge = random.choice(["blink", "turn_left", "turn_right"])
#         track.challenge = challenge
#         track.challenge_start = time.time()
#         track.challenge_passed = False
#         # set baseline for turn detection: use normalized eye-center difference
#         left_eye = np.mean([landmarks_xy[i] for i in LEFT_EYE_IDX], axis=0)
#         right_eye = np.mean([landmarks_xy[i] for i in RIGHT_EYE_IDX], axis=0)
#         # normalized ratio: (left_x - right_x) / face_width
#         bbox = track.bbox
#         face_w = max(1, bbox[2] - bbox[0])
#         track.challenge_baseline = (left_eye[0] - right_eye[0]) / face_w
#         # show on frame (handled at draw)
#         print(f"[CHALLENGE] Track {track.tid} -> {challenge}")

#     def _check_challenge(self, track, landmarks_xy, frame):
#         """Return True if challenge satisfied, False if timeout or not yet."""
#         if track.challenge is None:
#             return False
#         elapsed = time.time() - track.challenge_start
#         if elapsed > CHALLENGE_TIMEOUT:
#             # timeout -> fail
#             return False

#         # Evaluate according to challenge type
#         if track.challenge == "blink":
#             # compute EAR
#             left_ear = eye_aspect_ratio(landmarks_xy, LEFT_EYE_IDX, frame.shape[1], frame.shape[0])
#             right_ear = eye_aspect_ratio(landmarks_xy, RIGHT_EYE_IDX, frame.shape[1], frame.shape[0])
#             ear = (left_ear + right_ear) / 2.0
#             if ear < BLINK_EAR_THRESH:
#                 track.challenge_passed = True
#                 return True
#             return False

#         elif track.challenge in ("turn_left", "turn_right"):
#             left_eye = np.mean([landmarks_xy[i] for i in LEFT_EYE_IDX], axis=0)
#             right_eye = np.mean([landmarks_xy[i] for i in RIGHT_EYE_IDX], axis=0)
#             bbox = track.bbox
#             face_w = max(1, bbox[2] - bbox[0])
#             curr_ratio = (left_eye[0] - right_eye[0]) / face_w
#             delta = curr_ratio - track.challenge_baseline
#             # For left turn: left eye appears further left relative to right => delta negative (subject dependent)
#             # We use magnitude and direction heuristics:
#             if track.challenge == "turn_left" and delta < -TURN_RATIO_THRESH:
#                 track.challenge_passed = True
#                 return True
#             if track.challenge == "turn_right" and delta > TURN_RATIO_THRESH:
#                 track.challenge_passed = True
#                 return True
#             return False

#         return False

#     def match_embedding(self, emb):
#         sims = cosine_similarity(emb.reshape(1, -1), self.embeddings)[0]
#         best_idx = int(np.argmax(sims))
#         best_score = float(sims[best_idx])
#         return best_idx, best_score

#     def update_tracks(self, detections):
#         # detections: list of (bbox, face_obj)
#         for bbox, f in detections:
#             assigned = None
#             for t in self.tracks:
#                 if iou(t.bbox, bbox) > 0.3:
#                     assigned = t
#                     break
#             if assigned:
#                 assigned.bbox = bbox
#                 assigned.missing = 0
#             else:
#                 t = Track(self.next_tid, bbox)
#                 self.next_tid += 1
#                 self.tracks.append(t)

#         # mark missing and remove stale
#         for t in self.tracks[:]:
#             t.missing += 1
#             if t.missing > 40:
#                 self.tracks.remove(t)

#     def process_frame(self, frame):
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = self.app.get(frame)
#         detections = []
#         for f in faces:
#             bbox = f.bbox.astype(int).tolist()
#             detections.append((bbox, f))
#         # update tracker list
#         self.update_tracks(detections)

#         # MediaPipe landmarks processing (single-face mesh)
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         mp_res = self.mp_mesh.process(rgb)
#         lm_xy = None
#         if mp_res.multi_face_landmarks:
#             # use first face mesh
#             lm = mp_res.multi_face_landmarks[0].landmark
#             h, w, _ = frame.shape
#             lm_xy = [(pt.x * w, pt.y * h, pt.z) for pt in lm]  # list of tuples

#         # for optical flow compute prev_gray patch later
#         for t in self.tracks:
#             # find matching detection for this track (by IOU)
#             matched_face = None
#             for bbox, f in detections:
#                 if iou(bbox, t.bbox) > 0.3:
#                     matched_face = f
#                     break
#             if not matched_face:
#                 continue

#             box = t.bbox
#             x1, y1, x2, y2 = box
#             # embedding & match
#             emb = matched_face.normed_embedding
#             best_idx, best_score = self.match_embedding(emb)
#             if best_score > t.best_score:
#                 t.best_score = best_score
#                 t.name = self.names[best_idx]

#             # ANTI-SPOOF FEATURES
#             # Laplacian variance
#             patch = gray[y1:y2, x1:x2] if y2>y1 and x2>x1 else None
#             lap_var = laplacian_variance(patch) if patch is not None else 0.0
#             t.history.update_lap(lap_var)

#             # Optical flow
#             if self.prev_gray is not None:
#                 flow_mag = compute_optical_flow(self.prev_gray, gray, box)
#             else:
#                 flow_mag = 0.0
#             t.history.update_flow(flow_mag)

#             # Depth variation via mediapipe z-coordinates if available
#             depth_var = 0.0
#             if lm_xy is not None:
#                 zs = [p[2] for p in lm_xy]
#                 depth_var = float(np.std(zs))
#                 # update history (we'll use lap + flow primarily)
#             # update tracker history
#             # (blink handling also uses landmarks below)
#             # decide liveness baseline
#             blink_detected = False
#             # Blink detection via EAR
#             if lm_xy is not None:
#                 left_ear = eye_aspect_ratio(lm_xy, LEFT_EYE_IDX, frame.shape[1], frame.shape[0])
#                 right_ear = eye_aspect_ratio(lm_xy, RIGHT_EYE_IDX, frame.shape[1], frame.shape[0])
#                 ear = (left_ear + right_ear) / 2.0
#                 t.history.update_ear(ear)
#                 # simple blink heuristic: sudden drop in ear
#                 if ear < BLINK_EAR_THRESH:
#                     blink_detected = True

#             t.history.update_flow(flow_mag)

#             # Passive liveness check
#             live_passive = is_live_simple(t.history,
#                                           blink_detected=blink_detected,
#                                           flow_thresh=FLOW_THRESH,
#                                           lap_thresh=LAP_THRESH)
#             # Depth check
#             depth_ok = (depth_var > DEPTH_VAR_THRESH)

#             # Decision logic:
#             # If match strong and depth_ok and passive live -> accept immediately
#             # Else if match strong but either depth or passive uncertain -> issue challenge (if not already)
#             # If challenge exists -> check for pass; if pass -> accept; if timeout -> log fail
#             # label = "Unknown"
#             # color = (0,0,255)

#             # if best_score > (1 - self.threshold):
#             #     label = f"{t.name} {best_score:.2f}"
#             #     # strong & clearly live
#             #     if live_passive and depth_ok:
#             #         color = (0, 255, 0)
#             #         if not t.logged:
#             #             log_attendance(t.name, float(best_score))
#             #             t.logged = True
#             #     else:
#             #         # borderline -> handle challenge
#             #         if t.challenge is None:
#             #             # start a challenge
#             #             if lm_xy is not None:
#             #                 self._issue_challenge(t, frame, lm_xy)
#             #             else:
#             #                 # if no landmarks available, rely on passive - show orange
#             #                 color = (0, 165, 255)
#             #         else:
#             #             # challenge ongoing -> check it
#             #             passed = False
#             #             if lm_xy is not None:
#             #                 passed = self._check_challenge(t, lm_xy, frame)
#             #             if passed:
#             #                 color = (0, 255, 0)
#             #                 if not t.logged:
#             #                     log_attendance(t.name, float(best_score))
#             #                     t.logged = True
#             #                 # clear challenge
#             #                 t.challenge = None
#             #                 t.challenge_start = None
#             #             else:
#             #                 # not yet passed, check timeout
#             #                 if time.time() - t.challenge_start > CHALLENGE_TIMEOUT:
#             #                     # challenge failed
#             #                     color = (0, 165, 255)  # mark suspicious / orange
#             #                     # log spoof attempt
#             #                     with open(SPOOF_LOG, "a") as f:
#             #                         f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - FAILED CHALLENGE for {t.name} - score {best_score:.3f}\n")
#             #                     # clear challenge to avoid repeated logs
#             #                     t.challenge = None
#             #                     t.challenge_start = None
#             #                 else:
#             #                     color = (0, 165, 255) # still waiting for response
#             # else:
#             #     label = "Unknown"
#             #     color = (0,0,255)
#             # Decision logic
#             label = "Unknown"
#             color = (0, 0, 255)

#             if best_score > (1 - self.threshold):
#                 label = f"{t.name} {best_score:.2f}"

#                 # --- Stage 1: passive liveness check ---
#                 # Must have real movement, blink, and some depth
#                 passive_live = live_passive and depth_ok

#                 # --- Stage 2: Challenge if passive live is weak ---
#                 if not passive_live:
#                     if t.challenge is None and lm_xy is not None:
#                         self._issue_challenge(t, frame, lm_xy)
#                         color = (0, 165, 255)  # orange
#                     elif t.challenge is not None and lm_xy is not None:
#                         passed = self._check_challenge(t, lm_xy, frame)
#                         if passed:
#                             color = (0, 255, 0)
#                             if not t.logged:
#                                 log_attendance(t.name, float(best_score))
#                                 t.logged = True
#                             t.challenge = None
#                         else:
#                             if time.time() - t.challenge_start > CHALLENGE_TIMEOUT:
#                                 color = (0, 165, 255)
#                                 with open(SPOOF_LOG, "a") as f:
#                                     f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - FAILED CHALLENGE for {t.name} - score {best_score:.3f}\n")
#                                 t.challenge = None
#                             else:
#                                 color = (0, 165, 255)  # waiting
#                 else:
#                     # passive liveness good -> still require small challenge to confirm
#                     if t.challenge is None and lm_xy is not None:
#                         self._issue_challenge(t, frame, lm_xy)
#                     elif t.challenge is not None and lm_xy is not None:
#                         passed = self._check_challenge(t, lm_xy, frame)
#                         if passed:
#                             color = (0, 255, 0)
#                             if not t.logged:
#                                 log_attendance(t.name, float(best_score))
#                                 t.logged = True
#                             t.challenge = None
#                         else:
#                             color = (0, 165, 255)

#             else:
#                 label = "Unknown"
#                 color = (0, 0, 255)
            

#             # Draw rectangle and label and challenge text
#             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#             # If challenge active, show prompt and timer
#             if t.challenge is not None:
#                 prompt = f"Please {t.challenge.replace('_', ' ')} ({int(CHALLENGE_TIMEOUT - (time.time()-t.challenge_start))}s)"
#                 cv2.putText(frame, prompt, (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)

#             # show metrics
#             cv2.putText(frame, f"Flow:{t.history.avg_flow():.2f}", (x1, y2 + 20),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
#             cv2.putText(frame, f"Lap:{t.history.avg_lap():.1f}", (x1, y2 + 35),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
#             cv2.putText(frame, f"DepthVar:{depth_var:.4f}", (x1, y2 + 50),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

#         # update prev_gray
#         self.prev_gray = gray.copy()
#         return frame

# if __name__ == "__main__":
#     init_db()
#     recognizer = RealTimeRecognizerChallenge()
#     cap = cv2.VideoCapture(0)
#     print("[INFO] Starting recognition with challenge & depth. Press 'q' to quit.")
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         out = recognizer.process_frame(frame)
#         cv2.imshow("Recognition + AntiSpoof (Challenge)", out)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     cap.release()
#     cv2.destroyAllWindows()




















# recognize_with_challenge.py
import cv2
import numpy as np
import os
import time
import random
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from attendance_logger import log_attendance, init_db
from anti_spoof import (LivenessTracker,
                        eye_aspect_ratio,
                        compute_optical_flow,
                        laplacian_variance,
                        is_live_simple,
                        LEFT_EYE_IDX,
                        RIGHT_EYE_IDX)
import mediapipe as mp

# -----------------------
# Challenge configuration
# -----------------------
CHALLENGE_TIMEOUT = 5.0       # seconds to complete the challenge
BLINK_EAR_THRESH = 0.20       # EAR threshold for blink
TURN_RATIO_THRESH = 0.04      # normalized shift threshold for head turn detection (reduced slightly)
DEPTH_VAR_THRESH = 0.002      # threshold: low => likely flat (photo/screen)
FLOW_THRESH = 0.25
LAP_THRESH = 40.0

# Log file for spoof events
SPOOF_LOG = "spoof_log.txt"

# -----------------------
# Simple tracker (centroid/IOU)
# -----------------------
class Track:
    def __init__(self, tid, bbox):
        self.tid = tid
        self.bbox = bbox
        self.missing = 0
        self.history = LivenessTracker(maxlen=40)
        self.name = None
        self.best_score = 0.0
        self.logged = False
        # challenge state
        self.challenge = None
        self.challenge_start = None
        self.challenge_baseline = None  # baseline metric (for head-turn)
        self.challenge_passed = False

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    xa = max(ax1, bx1); ya = max(ay1, by1)
    xb = min(ax2, bx2); yb = min(ay2, by2)
    inter = max(0, xb - xa) * max(0, yb - ya)
    areaA = max(1, (ax2 - ax1) * (ay2 - ay1))
    areaB = max(1, (bx2 - bx1) * (by2 - by1))
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0

# -----------------------
# Recognizer with challenge-response and depth check
# -----------------------
# class RealTimeRecognizerChallenge:
#     def __init__(self, db_file="templates.npz", threshold=0.35):
#         self.db_file = db_file
class RealTimeRecognizerChallenge:
    def __init__(self, db_file=os.path.join(os.path.dirname(__file__), "..", "templates.npz"), threshold=0.45):
        self.db_file = os.path.abspath(db_file)
        self.threshold = threshold
        self._load_db()

        print("[INFO] Initializing FaceAnalysis model...")
        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        print("[INFO] Model loaded successfully.")

        # MediaPipe face mesh for landmarks (used for EAR + depth + baseline)
        self.mp_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
                                                       max_num_faces=1,
                                                       refine_landmarks=True,
                                                       min_detection_confidence=0.5,
                                                       min_tracking_confidence=0.5)
        self.tracks = []
        self.next_tid = 0
        self.prev_gray = None
        # mirror_sign: +1 or -1 depending on webcam mirroring; computed on first challenge
        self.mirror_sign = None

    def _load_db(self):
        if os.path.exists(self.db_file):
            data = np.load(self.db_file, allow_pickle=True)
            self.embeddings = np.array(data["embeddings"].tolist())
            self.names = data["names"].tolist()
            print(f"[INFO] Loaded {len(self.names)} enrolled users.")
        else:
            raise FileNotFoundError("templates.npz not found. Please enroll users first.")

    def _issue_challenge(self, track, frame, landmarks_xy):
        """Pick a random challenge and set baseline state if needed."""
        challenge = random.choice(["blink", "turn_left", "turn_right"])
        track.challenge = challenge
        track.challenge_start = time.time()
        track.challenge_passed = False
        # set baseline for turn detection: use normalized eye-center difference
        left_eye = np.mean([landmarks_xy[i] for i in LEFT_EYE_IDX], axis=0)
        right_eye = np.mean([landmarks_xy[i] for i in RIGHT_EYE_IDX], axis=0)
        bbox = track.bbox
        face_w = max(1, bbox[2] - bbox[0])
        track.challenge_baseline = (left_eye[0] - right_eye[0]) / face_w

        # determine mirror sign once if not set (accounts for mirrored webcam)
        if self.mirror_sign is None:
            # typically left_eye.x < right_eye.x in normal (non-mirrored) camera
            self.mirror_sign = 1 if (left_eye[0] < right_eye[0]) else -1
            print(f"[INFO] mirror_sign set to {self.mirror_sign}")

        print(f"[CHALLENGE] Track {track.tid} -> {challenge}")

    def _check_challenge(self, track, landmarks_xy, frame):
        """Return True if challenge satisfied, False if timeout or not yet."""
        if track.challenge is None:
            return False
        elapsed = time.time() - track.challenge_start
        if elapsed > CHALLENGE_TIMEOUT:
            # timeout -> fail
            return False

        # Evaluate according to challenge type
        if track.challenge == "blink":
            # compute EAR
            left_ear = eye_aspect_ratio(landmarks_xy, LEFT_EYE_IDX, frame.shape[1], frame.shape[0])
            right_ear = eye_aspect_ratio(landmarks_xy, RIGHT_EYE_IDX, frame.shape[1], frame.shape[0])
            ear = (left_ear + right_ear) / 2.0
            if ear < BLINK_EAR_THRESH:
                track.challenge_passed = True
                return True
            return False

        elif track.challenge in ("turn_left", "turn_right"):
            left_eye = np.mean([landmarks_xy[i] for i in LEFT_EYE_IDX], axis=0)
            right_eye = np.mean([landmarks_xy[i] for i in RIGHT_EYE_IDX], axis=0)
            bbox = track.bbox
            face_w = max(1, bbox[2] - bbox[0])
            curr_ratio = (left_eye[0] - right_eye[0]) / face_w
            # adjust for mirror sign
            if self.mirror_sign is not None:
                curr_ratio_adj = curr_ratio * self.mirror_sign
                baseline_adj = track.challenge_baseline * self.mirror_sign
            else:
                curr_ratio_adj = curr_ratio
                baseline_adj = track.challenge_baseline
            delta = curr_ratio_adj - baseline_adj

            # For left turn: delta should be negative beyond threshold
            if track.challenge == "turn_left" and delta < -TURN_RATIO_THRESH:
                track.challenge_passed = True
                return True
            if track.challenge == "turn_right" and delta > TURN_RATIO_THRESH:
                track.challenge_passed = True
                return True
            return False

        return False

    def match_embedding(self, emb):
        sims = cosine_similarity(emb.reshape(1, -1), self.embeddings)[0]
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        return best_idx, best_score

    def update_tracks(self, detections):
        # detections: list of (bbox, face_obj)
        for bbox, f in detections:
            assigned = None
            for t in self.tracks:
                if iou(t.bbox, bbox) > 0.3:
                    assigned = t
                    break
            if assigned:
                assigned.bbox = bbox
                assigned.missing = 0
            else:
                t = Track(self.next_tid, bbox)
                self.next_tid += 1
                self.tracks.append(t)

        # mark missing and remove stale
        for t in self.tracks[:]:
            t.missing += 1
            if t.missing > 40:
                self.tracks.remove(t)

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.app.get(frame)
        detections = []
        for f in faces:
            bbox = f.bbox.astype(int).tolist()
            detections.append((bbox, f))
        # update tracker list
        self.update_tracks(detections)

        # MediaPipe landmarks processing (single-face mesh)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_res = self.mp_mesh.process(rgb)
        lm_xy = None
        if mp_res.multi_face_landmarks:
            # use first face mesh
            lm = mp_res.multi_face_landmarks[0].landmark
            h, w, _ = frame.shape
            lm_xy = [(pt.x * w, pt.y * h, pt.z) for pt in lm]  # list of tuples

        # for optical flow compute prev_gray patch later
        for t in self.tracks:
            # find matching detection for this track (by IOU)
            matched_face = None
            for bbox, f in detections:
                if iou(bbox, t.bbox) > 0.3:
                    matched_face = f
                    break
            if not matched_face:
                continue

            box = t.bbox
            x1, y1, x2, y2 = box
            # embedding & match
            emb = matched_face.normed_embedding
            best_idx, best_score = self.match_embedding(emb)
            if best_score > t.best_score:
                t.best_score = best_score
                t.name = self.names[best_idx]

            # ANTI-SPOOF FEATURES
            # Laplacian variance
            patch = gray[y1:y2, x1:x2] if y2>y1 and x2>x1 else None
            lap_var = laplacian_variance(patch) if patch is not None else 0.0
            t.history.update_lap(lap_var)

            # Optical flow
            if self.prev_gray is not None:
                flow_mag = compute_optical_flow(self.prev_gray, gray, box)
            else:
                flow_mag = 0.0
            t.history.update_flow(flow_mag)

            # Depth variation via mediapipe z-coordinates if available
            depth_var = 0.0
            if lm_xy is not None:
                zs = [p[2] for p in lm_xy]
                depth_var = float(np.std(zs))

            # decide liveness baseline
            blink_detected = False
            # Blink detection via EAR
            if lm_xy is not None:
                left_ear = eye_aspect_ratio(lm_xy, LEFT_EYE_IDX, frame.shape[1], frame.shape[0])
                right_ear = eye_aspect_ratio(lm_xy, RIGHT_EYE_IDX, frame.shape[1], frame.shape[0])
                ear = (left_ear + right_ear) / 2.0
                t.history.update_ear(ear)
                if ear < BLINK_EAR_THRESH:
                    blink_detected = True

            t.history.update_flow(flow_mag)

            # Passive liveness check
            live_passive = is_live_simple(t.history,
                                          blink_detected=blink_detected,
                                          flow_thresh=FLOW_THRESH,
                                          lap_thresh=LAP_THRESH)
            # Depth check
            depth_ok = (depth_var > DEPTH_VAR_THRESH)

            # Decision logic: STRICT - attendance only after challenge pass
            label = "Unknown"
            color = (0, 0, 255)

            if best_score > (1 - self.threshold):
                label = f"{t.name} {best_score:.2f}"

                # Always require a short challenge even if passive checks look OK.
                if t.challenge is None and lm_xy is not None:
                    # start a challenge right away (makes video replay much harder)
                    self._issue_challenge(t, frame, lm_xy)
                    color = (0, 165, 255)  # waiting (orange)
                elif t.challenge is not None and lm_xy is not None:
                    # check challenge progress
                    passed = self._check_challenge(t, lm_xy, frame)
                    remaining = max(0.0, CHALLENGE_TIMEOUT - (time.time() - t.challenge_start))
                    if passed:
                        color = (0, 255, 0)
                        if not t.logged:
                            log_attendance(t.name, float(best_score))
                            t.logged = True
                            print(f"[+] Attendance logged for {t.name} (score={best_score:.3f})")
                        # clear challenge state
                        t.challenge = None
                        t.challenge_start = None
                    else:
                        # if timeout happened, it's a failed challenge; log and clear
                        if time.time() - t.challenge_start > CHALLENGE_TIMEOUT:
                            color = (0, 165, 255)
                            with open(SPOOF_LOG, "a") as f:
                                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - FAILED CHALLENGE for {t.name} - score {best_score:.3f}\n")
                            print(f"[!] FAILED challenge for {t.name}")
                            t.challenge = None
                            t.challenge_start = None
                        else:
                            color = (0, 165, 255)  # still waiting
                            # show remaining seconds in prompt below

            else:
                label = "Unknown"
                color = (0, 0, 255)

            # Draw rectangle and label and challenge text
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # If challenge active, show prompt and timer (clamped at 0)
            if t.challenge is not None:
                remaining = max(0.0, CHALLENGE_TIMEOUT - (time.time() - t.challenge_start))
                prompt = f"Please {t.challenge.replace('_', ' ')} ({int(remaining)}s)"
                cv2.putText(frame, prompt, (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)

            # show metrics
            cv2.putText(frame, f"Flow:{t.history.avg_flow():.2f}", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            cv2.putText(frame, f"Lap:{t.history.avg_lap():.1f}", (x1, y2 + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            cv2.putText(frame, f"DepthVar:{depth_var:.4f}", (x1, y2 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        # update prev_gray
        self.prev_gray = gray.copy()
        return frame

if __name__ == "__main__":
    init_db()
    recognizer = RealTimeRecognizerChallenge()
    cap = cv2.VideoCapture(0)
    print("[INFO] Starting recognition with challenge & depth. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out = recognizer.process_frame(frame)
        cv2.imshow("Recognition + AntiSpoof (Challenge)", out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
