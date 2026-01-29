# backend/main.py
from fastapi import FastAPI, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import os
import sqlite3
from datetime import datetime
import base64

from enroll_user_api import FaceEnroller
from recognize_with_challenge import RealTimeRecognizerChallenge

# ==========================================================
# FIX: Absolute DB and Template Paths
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "attendance.db")
TEMPLATE_PATH = os.path.join(BASE_DIR, "templates.npz")

# ==========================================================
# FASTAPI SETUP
# ==========================================================
app = FastAPI(title="Face Recognition Attendance API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================
# DB HELPER
# ==========================================================
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# ==========================================================
# ENROLLER + RECOGNIZER INIT
# ==========================================================
enroller = FaceEnroller(out_file=TEMPLATE_PATH)

try:
    recognizer = RealTimeRecognizerChallenge(db_file=TEMPLATE_PATH)
except FileNotFoundError:
    recognizer = None
    print("[WARN] No templates found. Recognition disabled until enrollment.")

# ==========================================================
# ROOT
# ==========================================================
@app.get("/")
def root():
    return {"message": "Face Recognition Attendance API running"}

# ==========================================================
# ENROLL USER (with recognizer reload fix)
# ==========================================================
@app.post("/enroll")
async def enroll_user(
    name: str = Form(...),
    email: str = Form(...),
    department: str = Form(...),
    image: UploadFile = None,
):
    if not image:
        return JSONResponse({"error": "No image provided"}, status_code=400)

    img_bytes = await image.read()
    np_img = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    try:
        enroller.enroll_from_image(name, frame)

        # FIX: reload recognizer database
        global recognizer
        if recognizer:
            recognizer._load_db()
        else:
            recognizer = RealTimeRecognizerChallenge(db_file=TEMPLATE_PATH)

        return {"status": "success", "message": f"User '{name}' enrolled successfully"}

    except Exception as e:
        print("[ENROLL ERROR]", e)
        return JSONResponse({"error": str(e)}, status_code=500)

# ==========================================================
# LIVE RECOGNITION
# ==========================================================
@app.post("/recognize")
async def recognize_user(payload: dict):
    if recognizer is None:
        return {"status": "error", "message": "No enrolled users"}

    try:
        frame_data = payload.get("frame", "")
        if not frame_data:
            return {"status": "error", "message": "No frame received"}

        frame_bytes = base64.b64decode(frame_data.split(",")[1])
        np_img = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        _, status = recognizer.process_frame(frame)

        return {
            "status": status["status"],
            "name": status.get("name"),
            "confidence": status.get("confidence", 0),
            "challenge": status.get("challenge"),
            "remaining": status.get("remaining", 0)
        }

    except Exception as e:
        print("[RECOGNITION ERROR]", e)
        return {"status": "error", "message": str(e)}

# ==========================================================
# ATTENDANCE ENDPOINTS
# ==========================================================
@app.get("/attendance")
def get_all_attendance():
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM Attendance ORDER BY timestamp DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.get("/attendance/today")
def get_today_attendance():
    today = datetime.now().strftime("%Y-%m-%d")
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM Attendance WHERE DATE(timestamp)=? ORDER BY timestamp DESC",
        (today,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.get("/attendance/{name}")
def get_user_attendance(name: str):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM Attendance WHERE name=? ORDER BY timestamp DESC",
        (name,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

# ==========================================================
# FIXED — CLEAN ENROLLED USERS + DEDUPLICATION
# ==========================================================
@app.get("/enrolled-users")
def get_enrolled_users():
    if not os.path.exists(TEMPLATE_PATH):
        return []
    data = np.load(TEMPLATE_PATH, allow_pickle=True)
    names = data["names"].tolist()
    return sorted(list(set(names)))

# ==========================================================
# NEW — Attendance Summary (for Dashboard)
# ==========================================================
@app.get("/attendance/summary")
def attendance_summary():
    conn = get_connection()
    rows = conn.execute("""
        SELECT name,
               COUNT(*) AS total,
               MAX(timestamp) AS last_seen
        FROM Attendance
        GROUP BY name
        ORDER BY name ASC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]
