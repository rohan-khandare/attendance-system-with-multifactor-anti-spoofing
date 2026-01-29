# import sqlite3
# from sqlmodel import SQLModel, Field, create_engine, Session, select
# from datetime import datetime
# import os

# DB_FILE = "attendance.db"
# engine = create_engine(f"sqlite:///{DB_FILE}", echo=False)

# class Attendance(SQLModel, table=True):
#     id: int | None = Field(default=None, primary_key=True)
#     name: str
#     timestamp: datetime
#     confidence: float

# def init_db():
#     if not os.path.exists(DB_FILE):
#         SQLModel.metadata.create_all(engine)
#         print("[INFO] Database created.")
#     else:
#         print("[INFO] Database already exists.")

# # def log_attendance(name: str, confidence: float):
# #     """Logs a new attendance entry if not already present for the current day."""
# #     with Session(engine) as session:
# #         today = datetime.now().date()
# #         statement = select(Attendance).where(
# #             Attendance.name == name,
# #             Attendance.timestamp >= datetime(today.year, today.month, today.day)
# #         )
# #         existing = session.exec(statement).first()
# #         if not existing:
# #             new_entry = Attendance(
# #                 name=name,
# #                 timestamp=datetime.now(),
# #                 confidence=confidence
# #             )
# #             session.add(new_entry)
# #             session.commit()
# #             print(f"[+] Logged attendance for {name}")
# #         else:
# #             print(f"[INFO] {name} already logged today.")
# def log_attendance(name, confidence):
#     conn = sqlite3.connect("attendance.db")
#     c = conn.cursor()

#     today = datetime.now().strftime("%Y-%m-%d")
#     # prevent duplicate log same user same day
#     existing = c.execute("SELECT * FROM Attendance WHERE name=? AND DATE(timestamp)=?", (name, today)).fetchone()
#     if existing:
#         print(f"[INFO] Attendance already logged today for {name}")
#         conn.close()
#         return False

#     c.execute("INSERT INTO Attendance (name, confidence, timestamp) VALUES (?, ?, ?)",
#               (name, confidence, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
#     conn.commit()
#     conn.close()
#     print(f"[LOGGED] Attendance recorded for {name}")
#     return True




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

