
# # # recognize_with_challenge.py
# # import cv2
# # import numpy as np
# # import os
# # import time
# # import random
# # from insightface.app import FaceAnalysis
# # from sklearn.metrics.pairwise import cosine_similarity
# # from attendance_logger import log_attendance, init_db
# # from anti_spoof import (LivenessTracker,
# #                         eye_aspect_ratio,
# #                         compute_optical_flow,
# #                         laplacian_variance,
# #                         is_live_simple,
# #                         LEFT_EYE_IDX,
# #                         RIGHT_EYE_IDX)
# # import mediapipe as mp

# # # -----------------------
# # # Challenge configuration
# # # -----------------------
# # CHALLENGE_TIMEOUT = 5.0       # seconds to complete the challenge
# # BLINK_EAR_THRESH = 0.20       # EAR threshold for blink
# # TURN_RATIO_THRESH = 0.04      # normalized shift threshold for head turn detection (reduced slightly)
# # DEPTH_VAR_THRESH = 0.002      # threshold: low => likely flat (photo/screen)
# # FLOW_THRESH = 0.25
# # LAP_THRESH = 40.0

# # # Log file for spoof events
# # SPOOF_LOG = "spoof_log.txt"

# # # -----------------------
# # # Simple tracker (centroid/IOU)
# # # -----------------------
# # class Track:
# #     def __init__(self, tid, bbox):
# #         self.tid = tid
# #         self.bbox = bbox
# #         self.missing = 0
# #         self.history = LivenessTracker(maxlen=40)
# #         self.name = None
# #         self.best_score = 0.0
# #         self.logged = False
# #         # challenge state
# #         self.challenge = None
# #         self.challenge_start = None
# #         self.challenge_baseline = None  # baseline metric (for head-turn)
# #         self.challenge_passed = False

# # def iou(a, b):
# #     ax1, ay1, ax2, ay2 = a
# #     bx1, by1, bx2, by2 = b
# #     xa = max(ax1, bx1); ya = max(ay1, by1)
# #     xb = min(ax2, bx2); yb = min(ay2, by2)
# #     inter = max(0, xb - xa) * max(0, yb - ya)
# #     areaA = max(1, (ax2 - ax1) * (ay2 - ay1))
# #     areaB = max(1, (bx2 - bx1) * (by2 - by1))
# #     union = areaA + areaB - inter
# #     return inter / union if union > 0 else 0

# # # -----------------------
# # # Recognizer with challenge-response and depth check
# # # -----------------------
# # class RealTimeRecognizerChallenge:
# #     # def __init__(self, db_file="templates.npz", threshold=0.35):
# #     def __init__(self, db_file=os.path.join(os.path.dirname(__file__), "..", "templates.npz"), threshold=0.45):
# #         self.db_file = db_file
# #         self.threshold = threshold
# #         self._load_db()

# #         print("[INFO] Initializing FaceAnalysis model...")
# #         self.app = FaceAnalysis(name='buffalo_l')
# #         self.app.prepare(ctx_id=0, det_size=(640, 640))
# #         print("[INFO] Model loaded successfully.")

# #         # MediaPipe face mesh for landmarks (used for EAR + depth + baseline)
# #         self.mp_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
# #                                                        max_num_faces=1,
# #                                                        refine_landmarks=True,
# #                                                        min_detection_confidence=0.5,
# #                                                        min_tracking_confidence=0.5)
# #         self.tracks = []
# #         self.next_tid = 0
# #         self.prev_gray = None
# #         # mirror_sign: +1 or -1 depending on webcam mirroring; computed on first challenge
# #         self.mirror_sign = None

# #     def _load_db(self):
# #         if os.path.exists(self.db_file):
# #             data = np.load(self.db_file, allow_pickle=True)
# #             self.embeddings = np.array(data["embeddings"].tolist())
# #             self.names = data["names"].tolist()
# #             print(f"[INFO] Loaded {len(self.names)} enrolled users.")
# #         else:
# #             raise FileNotFoundError("templates.npz not found. Please enroll users first.")

# #     def _issue_challenge(self, track, frame, landmarks_xy):
# #         """Pick a random challenge and set baseline state if needed."""
# #         challenge = random.choice(["blink", "turn_left", "turn_right"])
# #         track.challenge = challenge
# #         track.challenge_start = time.time()
# #         track.challenge_passed = False
# #         # set baseline for turn detection: use normalized eye-center difference
# #         left_eye = np.mean([landmarks_xy[i] for i in LEFT_EYE_IDX], axis=0)
# #         right_eye = np.mean([landmarks_xy[i] for i in RIGHT_EYE_IDX], axis=0)
# #         bbox = track.bbox
# #         face_w = max(1, bbox[2] - bbox[0])
# #         track.challenge_baseline = (left_eye[0] - right_eye[0]) / face_w

# #         # determine mirror sign once if not set (accounts for mirrored webcam)
# #         if self.mirror_sign is None:
# #             # typically left_eye.x < right_eye.x in normal (non-mirrored) camera
# #             self.mirror_sign = 1 if (left_eye[0] < right_eye[0]) else -1
# #             print(f"[INFO] mirror_sign set to {self.mirror_sign}")

# #         print(f"[CHALLENGE] Track {track.tid} -> {challenge}")

# #     def _check_challenge(self, track, landmarks_xy, frame):
# #         """Return True if challenge satisfied, False if timeout or not yet."""
# #         if track.challenge is None:
# #             return False
# #         elapsed = time.time() - track.challenge_start
# #         if elapsed > CHALLENGE_TIMEOUT:
# #             # timeout -> fail
# #             return False

# #         # Evaluate according to challenge type
# #         if track.challenge == "blink":
# #             # compute EAR
# #             left_ear = eye_aspect_ratio(landmarks_xy, LEFT_EYE_IDX, frame.shape[1], frame.shape[0])
# #             right_ear = eye_aspect_ratio(landmarks_xy, RIGHT_EYE_IDX, frame.shape[1], frame.shape[0])
# #             ear = (left_ear + right_ear) / 2.0
# #             if ear < BLINK_EAR_THRESH:
# #                 track.challenge_passed = True
# #                 return True
# #             return False

# #         elif track.challenge in ("turn_left", "turn_right"):
# #             left_eye = np.mean([landmarks_xy[i] for i in LEFT_EYE_IDX], axis=0)
# #             right_eye = np.mean([landmarks_xy[i] for i in RIGHT_EYE_IDX], axis=0)
# #             bbox = track.bbox
# #             face_w = max(1, bbox[2] - bbox[0])
# #             curr_ratio = (left_eye[0] - right_eye[0]) / face_w
# #             # adjust for mirror sign
# #             if self.mirror_sign is not None:
# #                 curr_ratio_adj = curr_ratio * self.mirror_sign
# #                 baseline_adj = track.challenge_baseline * self.mirror_sign
# #             else:
# #                 curr_ratio_adj = curr_ratio
# #                 baseline_adj = track.challenge_baseline
# #             delta = curr_ratio_adj - baseline_adj

# #             # For left turn: delta should be negative beyond threshold
# #             if track.challenge == "turn_left" and delta < -TURN_RATIO_THRESH:
# #                 track.challenge_passed = True
# #                 return True
# #             if track.challenge == "turn_right" and delta > TURN_RATIO_THRESH:
# #                 track.challenge_passed = True
# #                 return True
# #             return False

# #         return False

# #     def match_embedding(self, emb):
# #         sims = cosine_similarity(emb.reshape(1, -1), self.embeddings)[0]
# #         best_idx = int(np.argmax(sims))
# #         best_score = float(sims[best_idx])
# #         return best_idx, best_score

# #     def update_tracks(self, detections):
# #         # detections: list of (bbox, face_obj)
# #         for bbox, f in detections:
# #             assigned = None
# #             for t in self.tracks:
# #                 if iou(t.bbox, bbox) > 0.3:
# #                     assigned = t
# #                     break
# #             if assigned:
# #                 assigned.bbox = bbox
# #                 assigned.missing = 0
# #             else:
# #                 t = Track(self.next_tid, bbox)
# #                 self.next_tid += 1
# #                 self.tracks.append(t)

# #         # mark missing and remove stale
# #         for t in self.tracks[:]:
# #             t.missing += 1
# #             if t.missing > 40:
# #                 self.tracks.remove(t)

# #     def process_frame(self, frame):
# #         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# #         faces = self.app.get(frame)
# #         detections = []
# #         for f in faces:
# #             bbox = f.bbox.astype(int).tolist()
# #             detections.append((bbox, f))
# #         # update tracker list
# #         self.update_tracks(detections)

# #         # MediaPipe landmarks processing (single-face mesh)
# #         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# #         mp_res = self.mp_mesh.process(rgb)
# #         lm_xy = None
# #         if mp_res.multi_face_landmarks:
# #             # use first face mesh
# #             lm = mp_res.multi_face_landmarks[0].landmark
# #             h, w, _ = frame.shape
# #             lm_xy = [(pt.x * w, pt.y * h, pt.z) for pt in lm]  # list of tuples

# #         # for optical flow compute prev_gray patch later
# #         for t in self.tracks:
# #             # find matching detection for this track (by IOU)
# #             matched_face = None
# #             for bbox, f in detections:
# #                 if iou(bbox, t.bbox) > 0.3:
# #                     matched_face = f
# #                     break
# #             if not matched_face:
# #                 continue

# #             box = t.bbox
# #             x1, y1, x2, y2 = box
# #             # embedding & match
# #             emb = matched_face.normed_embedding
# #             best_idx, best_score = self.match_embedding(emb)
# #             if best_score > t.best_score:
# #                 t.best_score = best_score
# #                 t.name = self.names[best_idx]

# #             # ANTI-SPOOF FEATURES
# #             # Laplacian variance
# #             patch = gray[y1:y2, x1:x2] if y2>y1 and x2>x1 else None
# #             lap_var = laplacian_variance(patch) if patch is not None else 0.0
# #             t.history.update_lap(lap_var)

# #             # Optical flow
# #             if self.prev_gray is not None:
# #                 flow_mag = compute_optical_flow(self.prev_gray, gray, box)
# #             else:
# #                 flow_mag = 0.0
# #             t.history.update_flow(flow_mag)

# #             # Depth variation via mediapipe z-coordinates if available
# #             depth_var = 0.0
# #             if lm_xy is not None:
# #                 zs = [p[2] for p in lm_xy]
# #                 depth_var = float(np.std(zs))

# #             # decide liveness baseline
# #             blink_detected = False
# #             # Blink detection via EAR
# #             if lm_xy is not None:
# #                 left_ear = eye_aspect_ratio(lm_xy, LEFT_EYE_IDX, frame.shape[1], frame.shape[0])
# #                 right_ear = eye_aspect_ratio(lm_xy, RIGHT_EYE_IDX, frame.shape[1], frame.shape[0])
# #                 ear = (left_ear + right_ear) / 2.0
# #                 t.history.update_ear(ear)
# #                 if ear < BLINK_EAR_THRESH:
# #                     blink_detected = True

# #             t.history.update_flow(flow_mag)

# #             # Passive liveness check
# #             live_passive = is_live_simple(t.history,
# #                                           blink_detected=blink_detected,
# #                                           flow_thresh=FLOW_THRESH,
# #                                           lap_thresh=LAP_THRESH)
# #             # Depth check
# #             depth_ok = (depth_var > DEPTH_VAR_THRESH)

# #             # Decision logic: STRICT - attendance only after challenge pass
# #             label = "Unknown"
# #             color = (0, 0, 255)

# #             if best_score > (1 - self.threshold):
# #                 label = f"{t.name} {best_score:.2f}"

# #                 # Always require a short challenge even if passive checks look OK.
# #                 if t.challenge is None and lm_xy is not None:
# #                     # start a challenge right away (makes video replay much harder)
# #                     self._issue_challenge(t, frame, lm_xy)
# #                     color = (0, 165, 255)  # waiting (orange)
# #                 elif t.challenge is not None and lm_xy is not None:
# #                     # check challenge progress
# #                     passed = self._check_challenge(t, lm_xy, frame)
# #                     remaining = max(0.0, CHALLENGE_TIMEOUT - (time.time() - t.challenge_start))
# #                     if passed:
# #                         color = (0, 255, 0)
# #                         if not t.logged:
# #                             log_attendance(t.name, float(best_score))
# #                             t.logged = True
# #                             print(f"[+] Attendance logged for {t.name} (score={best_score:.3f})")
# #                         # clear challenge state
# #                         t.challenge = None
# #                         t.challenge_start = None
# #                     else:
# #                         # if timeout happened, it's a failed challenge; log and clear
# #                         if time.time() - t.challenge_start > CHALLENGE_TIMEOUT:
# #                             color = (0, 165, 255)
# #                             with open(SPOOF_LOG, "a") as f:
# #                                 f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - FAILED CHALLENGE for {t.name} - score {best_score:.3f}\n")
# #                             print(f"[!] FAILED challenge for {t.name}")
# #                             t.challenge = None
# #                             t.challenge_start = None
# #                         else:
# #                             color = (0, 165, 255)  # still waiting
# #                             # show remaining seconds in prompt below

# #             else:
# #                 label = "Unknown"
# #                 color = (0, 0, 255)

# #             # Draw rectangle and label and challenge text
# #             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
# #             cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# #             # If challenge active, show prompt and timer (clamped at 0)
# #             if t.challenge is not None:
# #                 remaining = max(0.0, CHALLENGE_TIMEOUT - (time.time() - t.challenge_start))
# #                 prompt = f"Please {t.challenge.replace('_', ' ')} ({int(remaining)}s)"
# #                 cv2.putText(frame, prompt, (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)

# #             # show metrics
# #             cv2.putText(frame, f"Flow:{t.history.avg_flow():.2f}", (x1, y2 + 20),
# #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
# #             cv2.putText(frame, f"Lap:{t.history.avg_lap():.1f}", (x1, y2 + 35),
# #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
# #             cv2.putText(frame, f"DepthVar:{depth_var:.4f}", (x1, y2 + 50),
# #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

# #         # update prev_gray
# #         self.prev_gray = gray.copy()
# #         return frame

# # if __name__ == "__main__":
# #     init_db()
# #     recognizer = RealTimeRecognizerChallenge()
# #     cap = cv2.VideoCapture(0)
# #     print("[INFO] Starting recognition with challenge & depth. Press 'q' to quit.")
# #     while True:
# #         ret, frame = cap.read()
# #         if not ret:
# #             break
# #         out = recognizer.process_frame(frame)
# #         cv2.imshow("Recognition + AntiSpoof (Challenge)", out)
# #         if cv2.waitKey(1) & 0xFF == ord('q'):
# #             break
# #     cap.release()
# #     cv2.destroyAllWindows()













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
# from collections import deque

# # -----------------------
# # Config
# # -----------------------
# CHALLENGE_TIMEOUT = 7.0       # allow more time for task
# BLINK_EAR_THRESH = 0.20
# TURN_RATIO_THRESH = 0.03
# DEPTH_VAR_THRESH = 0.002
# FLOW_THRESH = 0.25
# LAP_THRESH = 40.0

# CONSECUTIVE_PASS_REQ = 2
# LOG_DEBOUNCE_SEC = 30

# SPOOF_LOG = "spoof_log.txt"


# # -----------------------
# # Tracker class
# # -----------------------
# class Track:
#     def __init__(self, tid, bbox):
#         self.tid = tid
#         self.bbox = bbox
#         self.missing = 0
#         self.history = LivenessTracker(maxlen=40)
#         self.name = None
#         self.best_score = 0.0
#         self.logged_at = 0.0
#         self.challenge = None
#         self.challenge_start = None
#         self.challenge_baseline = None
#         self.challenge_passed = False
#         self.recent_passes = deque(maxlen=CONSECUTIVE_PASS_REQ)
#         self.last_challenge_issued = 0.0


# def iou(a, b):
#     ax1, ay1, ax2, ay2 = a
#     bx1, by1, bx2, by2 = b
#     xa = max(ax1, bx1)
#     ya = max(ay1, by1)
#     xb = min(ax2, bx2)
#     yb = min(ay2, by2)
#     inter = max(0, xb - xa) * max(0, yb - ya)
#     areaA = max(1, (ax2 - ax1) * (ay2 - ay1))
#     areaB = max(1, (bx2 - bx1) * (by2 - by1))
#     union = areaA + areaB - inter
#     return inter / union if union > 0 else 0


# # -----------------------
# # Recognizer with challenge
# # -----------------------
# class RealTimeRecognizerChallenge:
#     # def __init__(self, db_file=os.path.join(os.path.dirname(__file__), "templates.npz"), threshold=0.30):
#     #     self.db_file = db_file
#     #     self.threshold = threshold
#     #     self._load_db()
#     def __init__(self, db_file="templates.npz", threshold=0.45):
#         self.db_file = db_file
#         self.threshold = threshold
#         self._load_db()

#         print("[INFO] Initializing FaceAnalysis model...")
#         self.app = FaceAnalysis(name='buffalo_l')
#         self.app.prepare(ctx_id=0, det_size=(640, 640))
#         print("[INFO] Model loaded.")

#         self.mp_mesh = mp.solutions.face_mesh.FaceMesh(
#             static_image_mode=False,
#             max_num_faces=1,
#             refine_landmarks=True,
#             min_detection_confidence=0.5,
#             min_tracking_confidence=0.5,
#         )

#         self.tracks = []
#         self.next_tid = 0
#         self.prev_gray = None
#         self.mirror_sign = None
#         self.last_logged_time = {}

#     def _load_db(self):
#         if os.path.exists(self.db_file):
#             data = np.load(self.db_file, allow_pickle=True)
#             self.embeddings = np.array(data["embeddings"].tolist())
#             self.names = data["names"].tolist()
#             print(f"[INFO] Loaded {len(self.names)} enrolled users.")
#         else:
#             raise FileNotFoundError("templates.npz not found. Please enroll users first.")

#     def _issue_challenge(self, track, landmarks_xy):
#         now = time.time()
#         if now - track.last_challenge_issued < 0.5:
#             return
#         challenge = random.choice(["blink", "turn_left", "turn_right"])
#         track.challenge = challenge
#         track.challenge_start = now
#         track.challenge_passed = False
#         track.last_challenge_issued = now

#         left_eye = np.mean([landmarks_xy[i] for i in LEFT_EYE_IDX], axis=0)
#         right_eye = np.mean([landmarks_xy[i] for i in RIGHT_EYE_IDX], axis=0)
#         bbox = track.bbox
#         face_w = max(1, bbox[2] - bbox[0])
#         track.challenge_baseline = (left_eye[0] - right_eye[0]) / face_w

#         if self.mirror_sign is None:
#             self.mirror_sign = 1 if (left_eye[0] < right_eye[0]) else -1
#             print(f"[INFO] mirror_sign set to {self.mirror_sign}")

#         print(f"[CHALLENGE] Track {track.tid} -> {challenge}")

#     def _check_challenge(self, track, landmarks_xy, frame):
#         if track.challenge is None:
#             return False
#         elapsed = time.time() - track.challenge_start
#         if elapsed > CHALLENGE_TIMEOUT:
#             return False

#         if track.challenge == "blink":
#             left_ear = eye_aspect_ratio(landmarks_xy, LEFT_EYE_IDX, frame.shape[1], frame.shape[0])
#             right_ear = eye_aspect_ratio(landmarks_xy, RIGHT_EYE_IDX, frame.shape[1], frame.shape[0])
#             ear = (left_ear + right_ear) / 2.0
#             if ear < BLINK_EAR_THRESH:
#                 track.challenge_passed = True
#                 return True
#             return False

#         if track.challenge in ("turn_left", "turn_right"):
#             left_eye = np.mean([landmarks_xy[i] for i in LEFT_EYE_IDX], axis=0)
#             right_eye = np.mean([landmarks_xy[i] for i in RIGHT_EYE_IDX], axis=0)
#             bbox = track.bbox
#             face_w = max(1, bbox[2] - bbox[0])
#             curr_ratio = (left_eye[0] - right_eye[0]) / face_w
#             if self.mirror_sign is not None:
#                 curr_ratio *= self.mirror_sign
#                 baseline = track.challenge_baseline * self.mirror_sign
#             else:
#                 baseline = track.challenge_baseline
#             delta = curr_ratio - baseline
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
#         for t in self.tracks[:]:
#             t.missing += 1
#             if t.missing > 40:
#                 self.tracks.remove(t)

#     def process_frame(self, frame):
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = self.app.get(frame)
#         detections = [(f.bbox.astype(int).tolist(), f) for f in faces]
#         self.update_tracks(detections)

#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         mp_res = self.mp_mesh.process(rgb)
#         lm_xy = None
#         if mp_res.multi_face_landmarks:
#             lm = mp_res.multi_face_landmarks[0].landmark
#             h, w, _ = frame.shape
#             lm_xy = [(pt.x * w, pt.y * h, pt.z) for pt in lm]

#         status_obj = {"status": "no_face", "name": None, "confidence": 0.0, "challenge": None, "remaining": 0}

#         for t in self.tracks:
#             matched_face = None
#             for bbox, f in detections:
#                 if iou(bbox, t.bbox) > 0.3:
#                     matched_face = f
#                     break
#             if not matched_face:
#                 continue

#             x1, y1, x2, y2 = t.bbox
#             emb = matched_face.normed_embedding
#             best_idx, best_score = self.match_embedding(emb)
#             if best_score > t.best_score:
#                 t.best_score = best_score
#                 t.name = self.names[best_idx]

#             patch = gray[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else None
#             lap_var = laplacian_variance(patch) if patch is not None else 0.0
#             t.history.update_lap(lap_var)

#             flow_mag = compute_optical_flow(self.prev_gray, gray, t.bbox) if self.prev_gray is not None else 0.0
#             t.history.update_flow(flow_mag)

#             depth_var = 0.0
#             if lm_xy is not None:
#                 zs = [p[2] for p in lm_xy]
#                 depth_var = float(np.std(zs))

#             blink_detected = False
#             if lm_xy is not None:
#                 left_ear = eye_aspect_ratio(lm_xy, LEFT_EYE_IDX, frame.shape[1], frame.shape[0])
#                 right_ear = eye_aspect_ratio(lm_xy, RIGHT_EYE_IDX, frame.shape[1], frame.shape[0])
#                 ear = (left_ear + right_ear) / 2.0
#                 t.history.update_ear(ear)
#                 if ear < BLINK_EAR_THRESH:
#                     blink_detected = True

#             live_passive = is_live_simple(t.history, blink_detected=blink_detected, flow_thresh=FLOW_THRESH, lap_thresh=LAP_THRESH)
#             depth_ok = (depth_var > DEPTH_VAR_THRESH)

#             conf_threshold = 1 - self.threshold
#             if best_score >= conf_threshold:
#                 if t.challenge is None and lm_xy is not None:
#                     self._issue_challenge(t, lm_xy)
#                     status_obj = {"status": "challenge", "name": t.name, "confidence": float(t.best_score),
#                                   "challenge": t.challenge, "remaining": int(CHALLENGE_TIMEOUT)}
#                 elif t.challenge is not None and lm_xy is not None:
#                     passed = self._check_challenge(t, lm_xy, frame)
#                     remaining = max(0.0, CHALLENGE_TIMEOUT - (time.time() - t.challenge_start))
#                     if passed:
#                         t.recent_passes.append(1)
#                         if len(t.recent_passes) == CONSECUTIVE_PASS_REQ and all(t.recent_passes):
#                             now = time.time()
#                             if now - self.last_logged_time.get(t.name, 0.0) > LOG_DEBOUNCE_SEC:
#                                 log_attendance(t.name, float(t.best_score))
#                                 self.last_logged_time[t.name] = now
#                             status_obj = {"status": "real", "name": t.name, "confidence": float(t.best_score),
#                                           "challenge": None, "remaining": 0}
#                             t.challenge = None
#                             t.challenge_start = None
#                             t.recent_passes.clear()
#                         else:
#                             status_obj = {"status": "challenge", "name": t.name, "confidence": float(t.best_score),
#                                           "challenge": t.challenge, "remaining": int(remaining)}
#                     else:
#                         status_obj = {"status": "challenge", "name": t.name, "confidence": float(t.best_score),
#                                       "challenge": t.challenge, "remaining": int(remaining)}
#                         if time.time() - t.challenge_start > CHALLENGE_TIMEOUT:
#                             with open(SPOOF_LOG, "a") as f:
#                                 f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - FAILED CHALLENGE for {t.name}\n")
#                             status_obj = {"status": "spoof", "name": t.name, "confidence": float(t.best_score),
#                                           "challenge": None, "remaining": 0}
#                             t.challenge = None
#                             t.challenge_start = None
#                             t.recent_passes.clear()
#             else:
#                 status_obj = {"status": "unknown", "name": None, "confidence": float(best_score),
#                               "challenge": None, "remaining": 0}

#         self.prev_gray = gray.copy()
#         return frame, status_obj


# if __name__ == "__main__":
#     init_db()
#     recognizer = RealTimeRecognizerChallenge()
#     cap = cv2.VideoCapture(0)
#     print("[INFO] Running. Press 'q' to quit.")
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         out, status = recognizer.process_frame(frame)
#         cv2.imshow("Recognition Challenge", out)
#         print(status)
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
from collections import deque

# -----------------------
# Config (tune these)
# -----------------------
CHALLENGE_TIMEOUT = 6.0       # allow slightly more time to perform task
BLINK_EAR_THRESH = 0.20
TURN_RATIO_THRESH = 0.03
SMILE_RATIO_THRESH = 0.42    # mouth width / face width threshold for smile (tune if needed)
DEPTH_VAR_THRESH = 0.002
FLOW_THRESH = 0.20
LAP_THRESH = 30.0

CONSECUTIVE_PASS_REQ = 2     # require consecutive successes
LOG_DEBOUNCE_SEC = 30       # seconds before logging same user again

SPOOF_LOG = "spoo f_log.txt"  # note: if you prefer a different path, change here

# -----------------------
# Helper / Track class
# -----------------------
class Track:
    def __init__(self, tid, bbox):
        self.tid = tid
        self.bbox = bbox
        self.missing = 0
        self.history = LivenessTracker(maxlen=40)
        self.name = None
        self.best_score = 0.0
        self.logged_at = 0.0
        self.challenge = None
        self.challenge_start = None
        self.challenge_baseline = None
        self.challenge_passed = False
        self.recent_passes = deque(maxlen=CONSECUTIVE_PASS_REQ)
        self.last_challenge_issued = 0.0

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
# Recognizer with challenge-response & smile detection
# -----------------------
class RealTimeRecognizerChallenge:
    def __init__(self, db_file="templates.npz", threshold=0.45):
        self.db_file = db_file
        self.threshold = threshold
        self._load_db()

        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        # MediaPipe face mesh for landmarks (used for EAR, smile, baseline)
        self.mp_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
                                                       max_num_faces=1,
                                                       refine_landmarks=True,
                                                       min_detection_confidence=0.5,
                                                       min_tracking_confidence=0.5)
        self.tracks = []
        self.next_tid = 0
        self.prev_gray = None
        self.mirror_sign = None
        self.last_logged_time = {}

    def _load_db(self):
        if os.path.exists(self.db_file):
            data = np.load(self.db_file, allow_pickle=True)
            self.embeddings = np.array(data["embeddings"].tolist())
            self.names = data["names"].tolist()
            print(f"[INFO] Loaded {len(self.names)} enrolled users.")
        else:
            raise FileNotFoundError("templates.npz not found. Please enroll users first.")

    def _issue_challenge(self, track, landmarks_xy):
        # throttle challenges a bit
        now = time.time()
        if now - track.last_challenge_issued < 0.5:
            return
        challenge = random.choice(["blink", "smile", "turn_left", "turn_right"])
        track.challenge = challenge
        track.challenge_start = now
        track.challenge_passed = False
        track.last_challenge_issued = now

        # baseline for head-turn detection using eyes
        left_eye = np.mean([landmarks_xy[i] for i in LEFT_EYE_IDX], axis=0)
        right_eye = np.mean([landmarks_xy[i] for i in RIGHT_EYE_IDX], axis=0)
        bbox = track.bbox
        face_w = max(1, bbox[2] - bbox[0])
        track.challenge_baseline = (left_eye[0] - right_eye[0]) / face_w

        if self.mirror_sign is None:
            self.mirror_sign = 1 if (left_eye[0] < right_eye[0]) else -1
            print(f"[INFO] mirror_sign set to {self.mirror_sign}")

        print(f"[CHALLENGE] Track {track.tid} -> {challenge}")

    def _check_challenge(self, track, landmarks_xy, frame):
        if track.challenge is None:
            return False
        elapsed = time.time() - track.challenge_start
        if elapsed > CHALLENGE_TIMEOUT:
            return False

        if track.challenge == "blink":
            left_ear = eye_aspect_ratio(landmarks_xy, LEFT_EYE_IDX, frame.shape[1], frame.shape[0])
            right_ear = eye_aspect_ratio(landmarks_xy, RIGHT_EYE_IDX, frame.shape[1], frame.shape[0])
            ear = (left_ear + right_ear) / 2.0
            return ear < BLINK_EAR_THRESH

        if track.challenge == "smile":
            # estimate smile by mouth corner distance normalized by face width
            # Mediapipe face mesh mouth corners indices commonly: 61 (left), 291 (right) (works in many setups)
            # Use safe lookup: if indices exist, compute ratio
            try:
                left_mouth = landmarks_xy[61]
                right_mouth = landmarks_xy[291]
                bbox = track.bbox
                face_w = max(1, bbox[2] - bbox[0])
                mouth_w = abs(right_mouth[0] - left_mouth[0])
                ratio = mouth_w / face_w
                return ratio > SMILE_RATIO_THRESH
            except Exception:
                return False

        if track.challenge in ("turn_left", "turn_right"):
            left_eye = np.mean([landmarks_xy[i] for i in LEFT_EYE_IDX], axis=0)
            right_eye = np.mean([landmarks_xy[i] for i in RIGHT_EYE_IDX], axis=0)
            bbox = track.bbox
            face_w = max(1, bbox[2] - bbox[0])
            curr_ratio = (left_eye[0] - right_eye[0]) / face_w
            if self.mirror_sign is not None:
                curr_ratio_adj = curr_ratio * self.mirror_sign
                baseline_adj = track.challenge_baseline * self.mirror_sign
            else:
                curr_ratio_adj = curr_ratio
                baseline_adj = track.challenge_baseline
            delta = curr_ratio_adj - baseline_adj
            if track.challenge == "turn_left" and delta < -TURN_RATIO_THRESH:
                return True
            if track.challenge == "turn_right" and delta > TURN_RATIO_THRESH:
                return True
            return False

        return False

    def match_embedding(self, emb):
        sims = cosine_similarity(emb.reshape(1, -1), self.embeddings)[0]
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        return best_idx, best_score

    def update_tracks(self, detections):
        # match detections to tracks by IOU
        for bbox, f in detections:
            assigned = None
            for t in self.tracks:
                if iou(t.bbox, bbox) > 0.3:
                    assigned = t; break
            if assigned:
                assigned.bbox = bbox
                assigned.missing = 0
            else:
                t = Track(self.next_tid, bbox)
                self.next_tid += 1
                self.tracks.append(t)
        # age out missing
        for t in self.tracks[:]:
            t.missing += 1
            if t.missing > 40:
                self.tracks.remove(t)

    def process_frame(self, frame):
        """
        Processes a BGR frame and returns:
          - annotated frame (for debugging)
          - status object for frontend:
            {"status": "no_face"|"unknown"|"challenge"|"real"|"spoof",
             "name": str|None, "confidence": float, "challenge": str|None, "remaining": int}
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.app.get(frame)
        detections = []
        for f in faces:
            bbox = f.bbox.astype(int).tolist()
            detections.append((bbox, f))

        self.update_tracks(detections)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_res = self.mp_mesh.process(rgb)
        lm_xy = None
        if mp_res.multi_face_landmarks:
            lm = mp_res.multi_face_landmarks[0].landmark
            h, w, _ = frame.shape
            lm_xy = [(pt.x * w, pt.y * h, pt.z) for pt in lm]

        # default status
        status_obj = {"status": "no_face", "name": None, "confidence": 0.0, "challenge": None, "remaining": 0}

        for t in self.tracks:
            matched_face = None
            for bbox, f in detections:
                if iou(bbox, t.bbox) > 0.3:
                    matched_face = f; break
            if not matched_face:
                continue

            x1, y1, x2, y2 = t.bbox
            emb = matched_face.normed_embedding
            best_idx, best_score = self.match_embedding(emb)
            # update track best
            if best_score > t.best_score:
                t.best_score = best_score
                t.name = self.names[best_idx]

            # anti-spoof signals
            patch = gray[y1:y2, x1:x2] if y2>y1 and x2>x1 else None
            lap_var = laplacian_variance(patch) if patch is not None else 0.0
            t.history.update_lap(lap_var)

            flow_mag = compute_optical_flow(self.prev_gray, gray, t.bbox) if self.prev_gray is not None else 0.0
            t.history.update_flow(flow_mag)

            depth_var = 0.0
            if lm_xy is not None:
                zs = [p[2] for p in lm_xy]
                depth_var = float(np.std(zs))

            blink_detected = False
            if lm_xy is not None:
                left_ear = eye_aspect_ratio(lm_xy, LEFT_EYE_IDX, frame.shape[1], frame.shape[0])
                right_ear = eye_aspect_ratio(lm_xy, RIGHT_EYE_IDX, frame.shape[1], frame.shape[0])
                ear = (left_ear + right_ear) / 2.0
                t.history.update_ear(ear)
                if ear < BLINK_EAR_THRESH:
                    blink_detected = True

            live_passive = is_live_simple(t.history, blink_detected=blink_detected, flow_thresh=FLOW_THRESH, lap_thresh=LAP_THRESH)
            depth_ok = (depth_var > DEPTH_VAR_THRESH)

            conf_threshold = 1 - self.threshold
            # Default visuals
            display_label = "Unknown"
            color = (0,0,255)
            challenge_text = None
            remaining = 0

            if best_score >= conf_threshold:
                display_label = f"{t.name} {best_score:.2f}"

                # Issue challenge if none
                if t.challenge is None and lm_xy is not None:
                    # always require a short challenge to avoid video replay
                    self._issue_challenge(t, lm_xy)
                    color = (0,165,255)
                    status_obj = {"status": "challenge", "name": t.name, "confidence": float(best_score),
                                  "challenge": t.challenge, "remaining": int(CHALLENGE_TIMEOUT)}
                elif t.challenge is not None and lm_xy is not None:
                    passed = self._check_challenge(t, lm_xy, frame)
                    remaining = max(0.0, CHALLENGE_TIMEOUT - (time.time() - t.challenge_start))
                    challenge_text = t.challenge
                    if passed:
                        t.recent_passes.append(1)
                        if len(t.recent_passes) == CONSECUTIVE_PASS_REQ and all(t.recent_passes):
                            now = time.time()
                            last = self.last_logged_time.get(t.name, 0.0)
                            if now - last > LOG_DEBOUNCE_SEC:
                                # commit attendance
                                log_attendance(t.name, float(t.best_score))
                                self.last_logged_time[t.name] = now
                                t.logged_at = now
                                print(f"[+] Attendance logged for {t.name} (score={t.best_score:.3f})")
                            status_obj = {"status": "real", "name": t.name, "confidence": float(t.best_score),
                                          "challenge": None, "remaining": 0}
                            # reset
                            t.challenge = None; t.challenge_start = None; t.recent_passes.clear()
                        else:
                            # waiting for consecutive stability
                            status_obj = {"status": "challenge", "name": t.name, "confidence": float(t.best_score),
                                          "challenge": t.challenge, "remaining": int(remaining)}
                    else:
                        # still running or failed
                        status_obj = {"status": "challenge", "name": t.name, "confidence": float(t.best_score),
                                      "challenge": t.challenge, "remaining": int(remaining)}
                        # if timeout -> fail -> mark spoof
                        if time.time() - t.challenge_start > CHALLENGE_TIMEOUT:
                            with open(SPOOF_LOG, "a") as f:
                                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - FAILED CHALLENGE for {t.name}\n")
                            status_obj = {"status": "spoof", "name": t.name, "confidence": float(t.best_score),
                                          "challenge": None, "remaining": 0}
                            t.challenge = None; t.challenge_start = None; t.recent_passes.clear()
                else:
                    status_obj = {"status": "unknown", "name": t.name, "confidence": float(best_score),
                                  "challenge": None, "remaining": 0}
            else:
                status_obj = {"status": "unknown", "name": None, "confidence": float(best_score),
                              "challenge": None, "remaining": 0}

            # Drawing for debug/visuals
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, display_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if t.challenge is not None:
                remaining = max(0.0, CHALLENGE_TIMEOUT - (time.time() - t.challenge_start))
                prompt = f"Please {t.challenge.replace('_', ' ')} ({int(remaining)}s)"
                cv2.putText(frame, prompt, (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)

            cv2.putText(frame, f"Flow:{t.history.avg_flow():.2f}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            cv2.putText(frame, f"Lap:{t.history.avg_lap():.1f}", (x1, y2 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            cv2.putText(frame, f"DepthVar:{depth_var:.4f}", (x1, y2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        self.prev_gray = gray.copy()
        return frame, status_obj

    def get_status(self):
        """Return aggregated status across tracks (useful if frontend polls)."""
        if not self.tracks:
            return {"status":"no_face","name":None,"confidence":0.0,"challenge":None,"remaining":0}
        best = max(self.tracks, key=lambda t: t.best_score)
        if best.challenge is not None:
            remaining = max(0.0, CHALLENGE_TIMEOUT - (time.time() - best.challenge_start))
            return {"status":"challenge","name":best.name,"confidence":float(best.best_score),"challenge":best.challenge,"remaining":int(remaining)}
        if best.best_score >= (1 - self.threshold) and (time.time() - best.logged_at) < 1.0:
            return {"status":"real","name":best.name,"confidence":float(best.best_score),"challenge":None,"remaining":0}
        return {"status":"unknown","name":best.name,"confidence":float(best.best_score),"challenge":None,"remaining":0}


if __name__ == "__main__":
    init_db()
    recognizer = RealTimeRecognizerChallenge()
    cap = cv2.VideoCapture(0)
    print("[INFO] Starting recognition with challenge set (blink, smile, turns). Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out, status = recognizer.process_frame(frame)
        cv2.imshow("Recognition + AntiSpoof (Challenge)", out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
