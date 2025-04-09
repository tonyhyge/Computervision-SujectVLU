import torch
import cv2
import os
from datetime import datetime

# Định nghĩa đường dẫn
MODEL_PATH = "yolov5/runs/train/exp4/weights/best.pt"  # Cập nhật đường dẫn đúng
OUTPUT_DIR = "outputs"  # Thư mục lưu kết quả

# Tạo thư mục outputs nếu chưa có
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load mô hình YOLOv5 đã huấn luyện
model = torch.hub.load("yolov5", "custom", path=MODEL_PATH, source="local")

def detect_image(image_path):
    """Nhận diện sách từ ảnh và lưu kết quả"""
    img = cv2.imread(image_path)
    results = model(img)
    
    # Lưu ảnh kết quả
    output_path = os.path.join(OUTPUT_DIR, f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
    results.save(output_path)
    print(f"✔ Nhận diện xong! Ảnh lưu tại: {output_path}")

def detect_video(video_path):
    """Nhận diện sách từ video"""
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    # Định dạng video đầu ra
    output_video_path = os.path.join(OUTPUT_DIR, f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        out.write(results.render()[0])  # Lưu từng frame đã nhận diện

        # Hiển thị video real-time (bỏ nếu chạy server)
        cv2.imshow("Nhận diện Sách", results.render()[0])
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✔ Nhận diện xong! Video lưu tại: {output_video_path}")

if __name__ == "__main__":
    # Chạy nhận diện
    mode = input("Chọn chế độ (image/video): ").strip().lower()
    file_path = input("Nhập đường dẫn file: ").strip()

    if mode == "image":
        detect_image(file_path)
    elif mode == "video":
        detect_video(file_path)
    else:
        print("❌ Chế độ không hợp lệ. Chỉ chấp nhận 'image' hoặc 'video'.")
