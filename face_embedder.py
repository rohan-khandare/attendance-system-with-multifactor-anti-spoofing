# face_embedder.py
from insightface.app import FaceAnalysis
import numpy as np
import cv2

class FaceEmbedder:
    def __init__(self):
        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def get_embedding(self, frame):
        faces = self.app.get(frame)
        embeddings = []
        for f in faces:
            bbox = f.bbox.astype(int)
            embedding = f.normed_embedding  # 512-d normalized vector
            embeddings.append({
                "bbox": bbox.tolist(),
                "embedding": embedding.tolist()
            })
        return embeddings

if __name__ == "__main__":
    fe = FaceEmbedder()
    cap = cv2.VideoCapture(0)
    print("[INFO] Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        embeds = fe.get_embedding(frame)
        for e in embeds:
            x1, y1, x2, y2 = e["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("ArcFace Embedding View", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
