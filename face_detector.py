# face_detector.py
from insightface.app import FaceAnalysis
import cv2

class RetinaFaceDetector:
    def __init__(self):
        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def detect_faces(self, frame):
        faces = self.app.get(frame)
        results = []
        for f in faces:
            box = f.bbox.astype(int)
            landmarks = f.landmark_2d_106 if hasattr(f, 'landmark_2d_106') else None
            face_crop = frame[box[1]:box[3], box[0]:box[2]]
            results.append({
                "bbox": box.tolist(),
                "landmarks": landmarks.tolist() if landmarks is not None else None,
                "face_crop": face_crop
            })
        return results

if __name__ == "__main__":
    detector = RetinaFaceDetector()
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = detector.detect_faces(frame)
        for f in faces:
            x1, y1, x2, y2 = f["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("RetinaFace Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
