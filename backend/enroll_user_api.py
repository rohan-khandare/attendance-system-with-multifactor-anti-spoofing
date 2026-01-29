# enroll_user_api.py
from insightface.app import FaceAnalysis
import numpy as np
import cv2
import os

# class FaceEnroller:
#     def __init__(self, out_file="templates.npz"):
#         self.out_file = out_file
#         self.app = FaceAnalysis(name='buffalo_l')
#         self.app.prepare(ctx_id=0, det_size=(640, 640))
#         self.embeddings = []
#         self.names = []
#         self._load_existing()
class FaceEnroller:
    def __init__(self, out_file=os.path.join(os.path.dirname(__file__), "..", "templates.npz")):
        self.out_file = os.path.abspath(out_file)
        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.embeddings = []
        self.names = []
        self._load_existing()

    def _load_existing(self):
        if os.path.exists(self.out_file):
            data = np.load(self.out_file, allow_pickle=True)
            self.embeddings = data["embeddings"].tolist()
            self.names = data["names"].tolist()
            print(f"[INFO] Loaded {len(self.names)} existing embeddings.")
        else:
            print("[INFO] No existing templates found. Starting new file.")

    def enroll_from_image(self, person_name, frame):
        faces = self.app.get(frame)
        if len(faces) == 0:
            raise ValueError("No face detected. Please try again with clear lighting.")

        # Use first detected face
        emb = faces[0].normed_embedding
        self.embeddings.append(emb)
        self.names.append(person_name)
        self._save()
        print(f"[✓] Enrolled {person_name}")
        return True

    def _save(self):
        np.savez(self.out_file, embeddings=np.array(self.embeddings), names=np.array(self.names))
        print(f"[INFO] Saved embeddings to {self.out_file}")
