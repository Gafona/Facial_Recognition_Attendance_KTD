# 👤 Face Attendance System 📸

Hệ thống điểm danh tự động bằng nhận diện khuôn mặt, sử dụng **Haar Cascade** để phát hiện khuôn mặt và **MLP Classifier** để nhận dạng dựa trên embedding đã huấn luyện.

---

## 🚀 Tính năng chính

✅ Phát hiện khuôn mặt trong **thời gian thực** bằng Haar Cascade.  
✅ Trích xuất đặc trưng khuôn mặt và **giảm chiều** bằng PCA.  
✅ Phân loại khuôn mặt bằng **MLP Classifier**, dự đoán xác suất nhận dạng.  
✅ Điểm danh tự động nếu xác suất nhận dạng **vượt ngưỡng**.  
✅ Hỗ trợ lưu ảnh và log kết quả điểm danh.

---

## 🔍 Hướng dẫn sử dụng

### 1️⃣ Chuẩn bị dữ liệu và mô hình
Hệ thống sử dụng file duy nhất **`siamese_ml_model.pkl`** chứa:

- Bộ chọn đặc trưng (**selector**)
- Bộ chuẩn hoá (**scaler**)
- PCA giảm chiều dữ liệu (**pca**)
- Bộ phân loại cặp khuôn mặt (**pair_clf**)
- Embedding gallery và tên người (**gallery_embeddings**, **gallery_names**)

> ⚠️ **Lưu ý:**  
> Bạn cần chạy lại code:
> ```bash
> python prepare_data.py
> python siamese_ml_train.py
> ```
> để chuẩn bị dữ liệu và huấn luyện mô hình **riêng của mình** vì ngoại hình mỗi người là khác nhau.

---

### 2️⃣ Chạy hệ thống điểm danh
- Kết nối camera.
- Chạy file:
```bash
python camera.py
```
Khi khuôn mặt được phát hiện và xác suất nhận dạng ≥ 0.85, hệ thống sẽ:

+ Lưu ảnh khuôn mặt.
+ Ghi tên + thời gian vào log điểm danh.
+ ⏳ Thời gian chờ giữa hai lần ghi nhận cùng một người: 1.0 giây.

📦 Thư viện sử dụng
```
opencv-python    # Phát hiện và xử lý ảnh từ camera
numpy            # Xử lý dữ liệu số
scikit-learn     # Chuẩn hoá, PCA, MLP Classifier
joblib           # Lưu / tải mô hình .pkl
```
Cài đặt nhanh:
```
pip install opencv-python numpy scikit-learn joblib
```
🤝 Đóng góp
+ Mọi đóng góp đều được hoan nghênh!
+ Hãy fork repo, tạo pull request hoặc mở issue nếu có đề xuất cải thiện. ❤️❤️❤️❤️❤️❤️
