import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine

# -------------------
# Config
# -------------------
TEMPLATES_FILE = "templates.npz"
CSV_FILE = "attendance.csv"
THRESHOLD = 0.35  # cosine distance threshold

# -------------------
# Load templates
# -------------------
data = np.load(TEMPLATES_FILE, allow_pickle=True)
templates = {k: data[k] for k in data.files}

# -------------------
# Init face analysis
# -------------------
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# -------------------
# Streamlit UI
# -------------------
st.title("Attendance System")
st.write("Shows **Yes** if detected face matches enrolled student and logs attendance.")

run = st.checkbox("Start camera")
FRAME_WINDOW = st.image([])

# Initialize CSV
try:
    attendance = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    attendance = pd.DataFrame(columns=["name", "timestamp", "confidence"])
    attendance.to_csv(CSV_FILE, index=False)

cap = None
if run:
    cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Camera not available")
        break

    faces = app.get(frame)
    label = "❌ No"

    if len(faces) > 0:
        # take largest face
        face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
        feat = face.normed_embedding

        best_name = None
        best_score = 1.0
        for name, templ in templates.items():
            d = cosine(feat, templ)
            if d < best_score:
                best_score = d
                best_name = name

        if best_score <= THRESHOLD:
            conf = 1.0 - best_score
            label = f"✅ Yes ({best_name})"

            # Log attendance if not already logged today
            today = time.strftime("%Y-%m-%d")
            if not ((attendance["name"] == best_name) &
                    (attendance["timestamp"].str.startswith(today))).any():
                new_entry = pd.DataFrame([{
                    "name": best_name,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "confidence": round(conf, 3)
                }])
                attendance = pd.concat([attendance, new_entry], ignore_index=True)
                attendance.to_csv(CSV_FILE, index=False)

        # Draw bbox
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

cap.release()

# -------------------
# Show attendance table
# -------------------
st.subheader("📋 Attendance Log")
st.dataframe(attendance)

# Download button
st.download_button("Download Attendance CSV", attendance.to_csv(index=False).encode("utf-8"),
                   "attendance.csv", "text/csv")
