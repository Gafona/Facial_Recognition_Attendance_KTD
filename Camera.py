import cv2
import numpy as np
import os
import time
import joblib
from skimage.feature import hog
from prepare_data import extract_feature_from_image_bgr, IMG_SIZE

print("Đang tải mô hình ML đã train...")
model_ml = joblib.load('siamese_ml_model.pkl')
selector = model_ml['selector']
scaler = model_ml['scaler']
pca = model_ml['pca']
pair_clf = model_ml['pair_clf']
gallery_embeddings = model_ml['gallery_embeddings']
gallery_names = model_ml['gallery_names']

# Load Haar cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Thư mục lưu ảnh điểm danh
if not os.path.exists('captures'):
    os.makedirs('captures')

# Ngưỡng xác suất để nhận diện giống (có thể điều chỉnh)
PROB_THRESHOLD = 0.85

# Hàm tạo feature cặp cho MLP
def make_pair_feature(e1, e2):
    e1 = np.asarray(e1).ravel()
    e2 = np.asarray(e2).ravel()
    diff = np.abs(e1 - e2)
    return np.concatenate([e1, e2, diff]).astype(np.float32)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

face_detected_time = None
capture_delay = 2.0
captured_for_this_person = False

print("Bắt đầu camera, nhấn 'q' để thoát...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không đọc được frame từ webcam")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64))

    if len(faces) > 0:
        # Lấy mặt to nhất
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])

        # Tính trung tâm khuôn mặt detected
        center_x = x + w // 2
        center_y = y + h // 2

        # Kích thước khung cố định bạn muốn
        box_w, box_h = 250, 300

        # Tính tọa độ khung cố định, giữ tâm khuôn mặt
        new_x = max(center_x - box_w // 2, 0)
        new_y = max(center_y - box_h // 2, 0)
        new_x2 = min(new_x + box_w, frame.shape[1])  # không vượt quá frame
        new_y2 = min(new_y + box_h, frame.shape[0])

        # Cắt face_roi theo khung mới để trích feature
        face_roi = frame[new_y:new_y2, new_x:new_x2]

        # Trích feature giống khi train
        feat = extract_feature_from_image_bgr(face_roi)
        if feat is not None:
            # Xử lý pipeline: selector, scaler, pca
            feat_sel = selector.transform(feat.reshape(1, -1))
            feat_scaled = scaler.transform(feat_sel)
            embedding = pca.transform(feat_scaled)[0]

            # So sánh với gallery
            best_prob = 0
            best_person = "Unknown"
            for i, gallery_emb in enumerate(gallery_embeddings):
                pair_feat = make_pair_feature(embedding, gallery_emb).reshape(1, -1)
                prob = pair_clf.predict_proba(pair_feat)[0][1]
                if prob > best_prob:
                    best_prob = prob
                    best_person = gallery_names[i]

            if best_prob >= PROB_THRESHOLD:
                predicted_id = best_person
            else:
                predicted_id = "Unknown"

            # Xử lý lưu ảnh điểm danh
            current_time = time.time()
            if predicted_id != "Unknown":
                if face_detected_time is None:
                    face_detected_time = current_time
                    captured_for_this_person = False

                if not captured_for_this_person and (current_time - face_detected_time >= capture_delay):
                    timestamp = time.strftime('%Y%m%d-%H%M%S')
                    filename = f'captures/{predicted_id}_{timestamp}.jpg'
                    cv2.imwrite(filename, frame)
                    print(f"[✓] Đã điểm danh và lưu ảnh: {filename}")
                    captured_for_this_person = True
            else:
                face_detected_time = None
                captured_for_this_person = False

            label = f"{predicted_id} ({best_prob:.2f})"
            # Vẽ khung cố định trên frame
            cv2.rectangle(frame, (new_x, new_y), (new_x2, new_y2), (0, 255, 0), 2)
            # Hiển thị nhãn ở trên khung mới
            cv2.putText(frame, label, (new_x, new_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            face_detected_time = None
            captured_for_this_person = False

    else:
        face_detected_time = None
        captured_for_this_person = False

    cv2.imshow("Hệ thống nhận diện khuôn mặt", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
