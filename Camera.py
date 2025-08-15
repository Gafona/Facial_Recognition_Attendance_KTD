import os
import cv2
import torch
import numpy as np
from datetime import datetime
from prepare_data import extract_feature_from_image_bgr, IMG_SIZE
from model_ngu import build_gallery_from_new_dataset  # Hàm update gallery

MODEL_PATH = "siamese_ml_torch.pth"
DATAUPDATE_PATH = r"path_newdata"  # Nơi lưu ảnh người mới
NUM_FRAMES_TO_CAPTURE = 40
THRESHOLD = 0.15  # Ngưỡng nhận diện

def run_webcam_recognition(model_checkpoint_path=MODEL_PATH, threshold=THRESHOLD):
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    selector = checkpoint["selector"]
    scaler = checkpoint["scaler"]
    pca = checkpoint["pca"]
    gallery_embeddings = checkpoint["gallery_embeddings"]
    gallery_names = checkpoint["gallery_names"]

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        min_dist_overall = float('inf')
        nearest_name = "Unknown"

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, IMG_SIZE)
            feat = extract_feature_from_image_bgr(face_img)
            X_sel = selector.transform([feat])
            X_scaled = scaler.transform(X_sel)
            X_emb = pca.transform(X_scaled)

            emb_tensor = torch.tensor(X_emb, dtype=torch.float32)
            gallery_tensor = torch.tensor(gallery_embeddings, dtype=torch.float32)
            dists = torch.cdist(emb_tensor, gallery_tensor, p=2)
            min_dist, min_idx = torch.min(dists, dim=1)

            # Cập nhật người gần nhất trong frame
            if min_dist.item() < min_dist_overall:
                min_dist_overall = min_dist.item()
                if min_dist.item() < threshold:
                    nearest_name = gallery_names[min_idx.item()]
                else:
                    nearest_name = "Unknown"

            # Vẽ khung và tên
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{gallery_names[min_idx.item()]} ({min_dist.item():.2f})",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # In ra console người gần nhất
        print(f"Nearest face in frame: {nearest_name} (distance={min_dist_overall:.2f})")

        # Nhấn Space để đăng ký người mới
        key = cv2.waitKey(1) & 0xFF
        if key == 32 and len(faces) > 0:  # Space
            name = input("Nhập tên người mới: ").strip()
            user_folder = os.path.join(DATAUPDATE_PATH, name)
            os.makedirs(user_folder, exist_ok=True)

            print(f"📸 Bắt đầu chụp {NUM_FRAMES_TO_CAPTURE} frame cho {name}...")
            frames_captured = 0
            while frames_captured < NUM_FRAMES_TO_CAPTURE:
                ret, frame_cap = cap.read()
                if not ret:
                    continue
                gray_cap = cv2.cvtColor(frame_cap, cv2.COLOR_BGR2GRAY)
                faces_cap = face_cascade.detectMultiScale(gray_cap, scaleFactor=1.1, minNeighbors=5)
                for (x_c, y_c, w_c, h_c) in faces_cap:
                    face_img_cap = frame_cap[y_c:y_c+h_c, x_c:x_c+w_c]
                    face_img_cap = cv2.resize(face_img_cap, IMG_SIZE)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    img_path = os.path.join(user_folder, f"{timestamp}.jpg")
                    cv2.imwrite(img_path, face_img_cap)
                    frames_captured += 1
                    print(f"✅ Đã lưu {frames_captured}/{NUM_FRAMES_TO_CAPTURE} frame")
                    if frames_captured >= NUM_FRAMES_TO_CAPTURE:
                        break
            print(f"🎉 Hoàn tất đăng ký người mới: {name}")
            
            # Update gallery ngay sau khi đăng ký xong
            build_gallery_from_new_dataset(DATAUPDATE_PATH)
            checkpoint = torch.load(MODEL_PATH, map_location="cpu")
            selector = checkpoint["selector"]
            scaler = checkpoint["scaler"]
            pca = checkpoint["pca"]
            gallery_embeddings = checkpoint["gallery_embeddings"]
            gallery_names = checkpoint["gallery_names"]
            print("🔄 Gallery đã được cập nhật.")

        if key == ord("q"):  # Thoát
            break

        cv2.imshow("Face Recognition", frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam_recognition()
1
