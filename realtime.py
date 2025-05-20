import cv2
import numpy as np
import os
import pickle
import datetime
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity
from mtcnn import MTCNN
import firebase_admin
from firebase_admin import credentials, firestore
import base64

embedder = FaceNet()
detector = MTCNN()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, "Code", "embeddings", "face_cosine_data.pkl"), "rb") as f:
    face_data = pickle.load(f)  

cred = credentials.Certificate(os.path.join(BASE_DIR,"Code", "serviceAccountKey.json"))
firebase_admin.initialize_app(cred)
db = firestore.client()

# === 1. Chuyển ảnh sang base64 ===
def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

# === 2. Hàm so sánh mặt ===
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

# === 3. Hàm upload ảnh lên Firebase ===
def upload_image_to_firestore(image_path, label="unknown"):
    try:
        base64_img = image_to_base64(image_path)

        timestamp = datetime.datetime.utcnow()

        full_name = label  

        doc_ref = db.collection("tracking").document()
        doc_ref.set({
            "img": base64_img,
            "location": "Văn phòng 1",  
            "timestamp": timestamp,
            "full_name": full_name  
        })
        print(f"Đã upload: {os.path.basename(image_path)}")

    except Exception as e:
        print(f"Lỗi upload ảnh {image_path}:", e)

# === 4. Thiết lập webcam và cửa sổ ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Face Recognition", 1000, 800)

print("Camera đang chạy... Nhấn 'q' để thoát.")

# === 5. Vòng lặp nhận diện khuôn mặt ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = detector.detect_faces(frame)

    for result in results:
        x, y, w, h = result['box']
        try:
            # Trích xuất khuôn mặt
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (160, 160))

            emb = embedder.embeddings([face])[0]

            label, score = match_face(emb, threshold=0.7)
            color = (0, 255, 0) if label != "unknown" else (0, 0, 255)

            if label != "unknown":
                output_dir = os.path.join(BASE_DIR, "captured")
                os.makedirs(output_dir, exist_ok=True)

                today = datetime.datetime.utcnow().strftime("%d_%m_%Y_%H_%M_%S")
                filename = f"{label}_{today}.jpg"
                filepath = os.path.join(output_dir, filename)

                if not os.path.exists(filepath):
                    cv2.imwrite(filepath, frame)
                    print(f"Đã lưu hình: {filepath}")
                    upload_image_to_firestore(filepath, label)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        except Exception as e:
            print("Lỗi nhận diện:", e)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
