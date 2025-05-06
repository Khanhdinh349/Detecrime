import cv2
import numpy as np
import os
import pickle
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity

# === 1. Khởi tạo ===
embedder = FaceNet()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# === 2. Load embedding theo label ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, "Code", "embeddings", "face_cosine_data.pkl"), "rb") as f:
    face_data = pickle.load(f)  # dict: {label: [emb1, emb2, ...]}

# === 3. Thiết lập webcam và cửa sổ ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Face Recognition", 1000, 800)

print("🎥 Camera đang chạy... Nhấn 'q' để thoát.")

def match_face(embedding, threshold=0.7):
    best_score = -1
    best_label = "unknown"
    for label, emb_list in face_data.items():
        sims = cosine_similarity([embedding], emb_list)
        max_sim = np.max(sims)
        if max_sim > best_score and max_sim > threshold:
            best_score = max_sim
            best_label = label
    return best_label, best_score

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        try:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (160, 160))
            emb = embedder.embeddings([face])[0]

            label, score = match_face(emb, threshold=0.7)
            color = (0, 255, 0) if label != "unknown" else (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        except Exception as e:
            print("⚠️ Lỗi nhận diện:", e)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
