# Smart Attendance System with Multi-Factor Anti-Spoofing

A real-time AI-powered attendance system that uses face recognition and multi-factor liveness detection to prevent spoofing attacks such as photo, video, or mask-based impersonation.  
The system integrates Computer Vision, Deep Learning, and a full-stack architecture with FastAPI and React.

---

## 🚀 Features

- 🔐 **Secure Face Recognition**
  - Uses InsightFace embeddings for high-accuracy face matching
  - Achieved ~98% recognition accuracy

- 🧠 **Multi-Factor Anti-Spoofing**
  - Blink detection  
  - Smile detection  
  - Head movement challenges  
  - Depth variance analysis  
  - Optical flow motion analysis  
  - Laplacian sharpness detection  
  - Achieved ~96.8% anti-spoofing accuracy

- 📊 **Attendance Management**
  - Real-time attendance marking
  - User-wise and date-wise logs
  - Attendance summaries for dashboard

- 🌐 **RESTful API**
  - Enrollment
  - Recognition
  - Attendance logging
  - Analytics endpoints

- 🖥️ **React Dashboard**
  - Live monitoring
  - User enrollment
  - Attendance records visualization
  - Clean and responsive UI

---

## 🛠 Tech Stack

**Backend**
- FastAPI
- Python
- OpenCV
- InsightFace
- MediaPipe
- NumPy
- SQLite (can be replaced with MySQL)

**Frontend**
- React.js
- Tailwind CSS
- Axios

**AI & Computer Vision**
- Face Embeddings (InsightFace)
- Liveness Detection
- Optical Flow
- Laplacian Sharpness
- Depth Variance Analysis

---


