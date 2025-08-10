# 👤 Face Attendance System 📸

Dự án này là hệ thống điểm danh tự động bằng nhận diện khuôn mặt, sử dụng kết hợp Haar Cascade để phát hiện khuôn mặt và MLP Classifier để nhận dạng dựa trên embedding đã huấn luyện.

## 🚀 Tính năng chính

✅ Phát hiện khuôn mặt trong thời gian thực bằng Haar Cascade.

✅ Trích xuất đặc trưng khuôn mặt và giảm chiều bằng PCA.

✅ Phân loại khuôn mặt bằng MLP Classifier, dự đoán xác suất nhận dạng.

✅ Điểm danh tự động nếu xác suất nhận dạng vượt ngưỡng.

✅ Hỗ trợ lưu lại ảnh và log kết quả điểm danh.

## 🔍 Hướng dẫn sử dụng

1️⃣ Chuẩn bị dữ liệu và mô hình
Hệ thống sử dụng một file duy nhất siamese_ml_model.pkl chứa:

+ Bộ chọn đặc trưng (selector)
+ Bộ chuẩn hoá (scaler)
+ PCA giảm chiều dữ liệu (pca)
+ Bộ phân loại cặp khuôn mặt (pair_clf)
+ Embedding gallery và tên người (gallery_embeddings, gallery_names)

Nhưng bạn cần chạy lại code prepare_data.py để có được model vì khuôn mặt người không phải ai cũng giống nhau đúng không nè!, nên là hãy tại ra 1 data của riêng mình và chạy chúng nha

2️⃣ Chạy hệ thống điểm danh
Kết nối camera.

Chạy file camera.py (hoặc file chính của bạn).

Khi khuôn mặt được phát hiện và xác suất nhận dạng ≥ 0.95, hệ thống sẽ:

+ Lưu ảnh khuôn mặt.

+ Ghi tên + thời gian vào log điểm danh.

## 🔥 Lưu ý:

- Thời gian chờ giữa hai lần ghi nhận cùng một người: 2 giây (capture_delay = 2.0).

## 📦 Các thư viện đã sử dụng
- opencv-python – Phát hiện và xử lý ảnh từ camera.
- numpy – Xử lý dữ liệu số.
- scikit-learn – Chuẩn hoá, PCA và MLP Classifier.
- joblib – Lưu / tải mô hình .pkl.

## 🤝 Đóng góp
- Mọi đóng góp đều được hoan nghênh!
- Hãy fork repo, tạo pull request hoặc mở issue nếu có đề xuất cải thiện. ❤️❤️❤️

