# check_attendance_db.py
import sqlite3
from tabulate import tabulate

def show_attendance():
    conn = sqlite3.connect("attendance.db")
    cur = conn.cursor()
    cur.execute("SELECT name, timestamp, confidence FROM attendance ORDER BY timestamp DESC LIMIT 10")
    rows = cur.fetchall()
    print(tabulate(rows, headers=["Name", "Timestamp", "Confidence"], tablefmt="pretty"))
    conn.close()

if __name__ == "__main__":
    show_attendance()
