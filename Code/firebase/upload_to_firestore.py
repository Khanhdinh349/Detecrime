import os
import base64
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAPTURED_DIR = os.path.join(BASE_DIR, "..", "..", "captured")
CAPTURED_DIR = os.path.abspath(CAPTURED_DIR)

cred = credentials.Certificate(os.path.join(BASE_DIR, "serviceAccountKey.json"))
firebase_admin.initialize_app(cred)

db = firestore.client()

def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

def upload_image(image_path, location="Unknown"):
    try:
        base64_img = image_to_base64(image_path)

        timestamp = datetime.datetime.now().strftime("%B %d, %Y at %I:%M:%S") + " " + datetime.datetime.now().strftime("%p") + " UTC+7"

        doc_ref = db.collection("tracking").document()
        doc_ref.set({
            "img": base64_img,
            "location": location,
            "timestamp": timestamp
        })
        print(f"Đã upload: {os.path.basename(image_path)}")

    except Exception as e:
        print(f"Lỗi upload ảnh {image_path}:", e)

def upload_all_images(location="Unknown"):
    if not os.path.exists(CAPTURED_DIR):
        print("Thư mục captured/ không tồn tại.")
        return

    for filename in os.listdir(CAPTURED_DIR):
        if filename.lower().endswith((".jpg", ".jpeg")):
            full_path = os.path.join(CAPTURED_DIR, filename)
            upload_image(full_path, location)

if __name__ == "__main__":
    upload_all_images(location="Văn phòng 1")
