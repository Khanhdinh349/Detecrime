import os
import cv2
import numpy as np
import pickle
from mtcnn import MTCNN
from keras_facenet import FaceNet

# === 1. Khởi tạo ===
embedder = FaceNet()
detector = MTCNN()

# === 2. Lấy thư mục dataset ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, "Code", "dataset")

# Hàm trích xuất khuôn mặt
def extract_face(img_path, required_size=(160, 160)):
    img = cv2.imread(img_path)
    if img is None:
        return None
    results = detector.detect_faces(img)
    if len(results) == 0:
        return None
    x, y, w, h = results[0]['box']
    x, y = max(x, 0), max(y, 0)
    face = img[y:y + h, x:x + w]
    face = cv2.resize(face, required_size)
    return face

# === 3. Duyệt tất cả người và ảnh ===
face_data = {}  # Dict: {label: [emb1, emb2, ...]}

for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_path):
        continue
    face_data[person_name] = []
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        face = extract_face(img_path)
        if face is not None:
            emb = embedder.embeddings([face])[0]
            face_data[person_name].append(emb)
            print(f"Đã lấy embedding: {person_name} / {img_name}")

# === 4. Lưu vào embeddings/face_cosine_data.pkl ===
embedding_dir = os.path.join(BASE_DIR, "Code", "embeddings")
os.makedirs(embedding_dir, exist_ok=True)
with open(os.path.join(embedding_dir, "face_cosine_data.pkl"), "wb") as f:
    pickle.dump(face_data, f)

print("Huấn luyện thành công! Đã lưu vào face_cosine_data.pkl")
