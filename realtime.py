import cv2
import numpy as np
import os
import pickle
import datetime
import base64
import time
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity
from mtcnn import MTCNN
import firebase_admin
from firebase_admin import credentials, firestore

# === THÊM: Màn hình ST7735 ===
import st7735
from PIL import Image, ImageDraw, ImageFont

# === THÊM: Loa GPIO ===
import RPi.GPIO as GPIO

# === THIẾT LẬP THIẾT BỊ NGOẠI VI ===
# Màn hình ST7735
disp = st7735.ST7735(
    port=0,
    cs=0,
    dc=24,
    rst=25,
    rotation=0,
    width=128,
    height=160,
    invert=False
)
disp.begin()

# GPIO Buzzer
BUZZER_PIN = 18
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)

def display_name_on_st7735(name: str):
    WIDTH, HEIGHT = disp.width, disp.height
    image = Image.new('RGB', (WIDTH, HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((10, 70), name, font=font, fill=(0, 255, 0))  # Chữ xanh lá
    disp.display(image)

def play_beep():
    for _ in range(2):
        GPIO.output(BUZZER_PIN, GPIO.HIGH)
        time.sleep(0.1)
        GPIO.output(BUZZER_PIN, GPIO.LOW)
        time.sleep(0.1)

# === THIẾT LẬP NHẬN DIỆN KHUÔN MẶT ===
embedder = FaceNet()
detector = MTCNN()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, "Code", "embeddings", "face_cosine_data.pkl"), "rb") as f:
    face_data = pickle.load(f)  

cred = credentials.Certificate(os.path.join(BASE_DIR, "Code", "serviceAccountKey.json"))
firebase_admin.initialize_app(cred)
db = firestore.client()

def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

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

def upload_image_to_firestore(image_path, label="unknown"):
    try:
        base64_img = image_to_base64(image_path)
        timestamp = datetime.datetime.utcnow()
        doc_ref = db.collection("tracking").document()
        doc_ref.set({
            "img": base64_img,
            "location": "Văn phòng 1",
            "timestamp": timestamp,
            "full_name": label
        })
        print(f"Đã upload: {os.path.basename(image_path)}")
    except Exception as e:
        print(f"Lỗi upload ảnh {image_path}:", e)

# === CHẠY CAMERA ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Face Recognition", 1000, 800)

print("Camera đang chạy... Nhấn 'q' để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = detector.detect_faces(frame)

    for result in results:
        x, y, w, h = result['box']
        try:
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

                    # === GỌI THÊM: hiển thị và phát âm thanh
                    display_name_on_st7735(label)
                    play_beep()

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        except Exception as e:
            print("Lỗi nhận diện:", e)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
GPIO.cleanup()
