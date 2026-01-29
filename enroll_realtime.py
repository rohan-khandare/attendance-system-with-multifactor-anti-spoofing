# enroll_realtime.py
from insightface.app import FaceAnalysis
import numpy as np
import cv2
import os

class RealTimeEnroller:
    def __init__(self, out_file="templates.npz"):
        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.out_file = out_file
        self.embeddings = []
        self.names = []
        self._load_existing()

    def _load_existing(self):
        """Load existing embeddings if templates.npz exists."""
        if os.path.exists(self.out_file):
            data = np.load(self.out_file)
            self.embeddings = data["embeddings"].tolist()
            self.names = data["names"].tolist()
            print(f"[INFO] Loaded {len(self.names)} existing embeddings.")
        else:
            print("[INFO] No existing enrollment found. Starting fresh.")

    def enroll(self, person_name, samples=5):
        cap = cv2.VideoCapture(0)
        print(f"[INFO] Starting enrollment for: {person_name}")
        print(f"[INFO] Press 'c' to capture a face ({samples} total), 'q' to quit early.")
        captured = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            faces = self.app.get(frame)
            for f in faces:
                box = f.bbox.astype(int)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.imshow("Enrollment - Press 'c' to capture", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') and len(faces) > 0:
                emb = faces[0].normed_embedding
                self.embeddings.append(emb)
                self.names.append(person_name)
                captured += 1
                print(f"[✓] Captured sample {captured}/{samples}")
                if captured >= samples:
                    break
            elif key == ord('q'):
                print("[INFO] Enrollment cancelled.")
                break

        cap.release()
        cv2.destroyAllWindows()
        self._save()
        print(f"[✓] Enrollment complete for {person_name}.")

    def _save(self):
        np.savez(self.out_file, embeddings=np.array(self.embeddings), names=np.array(self.names))
        print(f"[INFO] Saved to {self.out_file}")

if __name__ == "__main__":
    person_name = input("Enter name to enroll: ")
    enroller = RealTimeEnroller()
    enroller.enroll(person_name)
