# anti_spoof.py
import numpy as np
import cv2
import mediapipe as mp
from collections import deque

mp_face_mesh = mp.solutions.face_mesh

# MediaPipe landmark indices for eyes (common recommended set)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(landmarks, eye_indices, w, h):
    """
    Compute EAR using mediapipe normalized landmarks
    landmarks: list of (x,y) normalized coords
    eye_indices: 6 indices for the eye
    w,h: image size
    """
    pts = [(int(landmarks[i][0] * w), int(landmarks[i][1] * h)) for i in eye_indices]
    # vertical distances
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    # horizontal distance
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    if C == 0:
        return 0.0
    ear = (A + B) / (2.0 * C)
    return ear

class LivenessTracker:
    """
    Track short history per track_id:
      - last N EAR values (to detect blinks)
      - last N optical flow magnitudes
      - last N laplacian variances (sharpness)
    """
    def __init__(self, maxlen=40):
        self.ears = deque(maxlen=maxlen)
        self.flows = deque(maxlen=maxlen)
        self.laps = deque(maxlen=maxlen)
        self.last_gray = None
        self.blinked = False
        self.blink_count = 0

    def update_ear(self, ear, ear_thresh=0.20):
        self.ears.append(ear)
        # blink detection heuristic: ear below thresh from a previous higher ear
        if len(self.ears) >= 3:
            if self.ears[-2] < ear_thresh and self.ears[-3] >= ear_thresh:
                # already low — ignore
                pass
            # count transition high -> low -> high
            # simple detection: if any ear < thresh and previous > thresh
        # We'll count blinks using sliding window passed externally.

    def update_flow(self, flow_mag):
        self.flows.append(flow_mag)

    def update_lap(self, lap_var):
        self.laps.append(lap_var)

    def avg_flow(self):
        return float(np.mean(self.flows)) if len(self.flows) else 0.0

    def avg_lap(self):
        return float(np.mean(self.laps)) if len(self.laps) else 0.0

def compute_optical_flow(prev_gray, gray, bbox):
    """
    Compute mean optical flow magnitude in bbox using Farneback on whole bbox area.
    prev_gray, gray: full gray frames
    bbox: (x1,y1,x2,y2)
    """
    x1, y1, x2, y2 = bbox
    h, w = gray.shape[:2]
    x1, x2 = max(0, x1), min(w - 1, x2)
    y1, y2 = max(0, y1), min(h - 1, y2)
    if x2 - x1 <= 2 or y2 - y1 <= 2:
        return 0.0
    prev_patch = prev_gray[y1:y2, x1:x2]
    cur_patch = gray[y1:y2, x1:x2]
    try:
        flow = cv2.calcOpticalFlowFarneback(prev_patch, cur_patch, None,
                                            pyr_scale=0.5, levels=1, winsize=15,
                                            iterations=2, poly_n=5, poly_sigma=1.1, flags=0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mean_mag = float(np.mean(mag))
        return mean_mag
    except Exception:
        return 0.0

def laplacian_variance(patch):
    return float(cv2.Laplacian(patch, cv2.CV_64F).var())

def is_live_simple(history: LivenessTracker, blink_count_window=1,
                   blink_detected=False,
                   flow_thresh=0.3, lap_thresh=50.0):
    """
    Decision function:
      - If a blink was detected in window -> LIVE
      - Else if avg_flow > flow_thresh AND avg_lap > lap_thresh -> LIVE
      - Otherwise -> NOT LIVE
    Thresholds are heuristic—tune on your environment.
    """
    if blink_detected:
        return True
    if history.avg_flow() > flow_thresh and history.avg_lap() > lap_thresh:
        return True
    return False
