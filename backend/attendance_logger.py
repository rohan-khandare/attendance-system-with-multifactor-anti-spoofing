
# attendance_logger.py
import sqlite3
import os
from datetime import datetime

DB_PATH = "attendance.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        timestamp TEXT,
        confidence REAL
    )
    """)
    conn.commit()
    conn.close()

def log_attendance(name, confidence):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO attendance (name, timestamp, confidence) VALUES (?, ?, ?)", (name, ts, confidence))
    conn.commit()
    conn.close()
    print(f"[DB] Attendance logged for {name} at {ts} (conf={confidence:.2f})")

