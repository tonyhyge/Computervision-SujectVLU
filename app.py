import os
import torch
import cv2
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, send_from_directory

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load YOLOv5 model
MODEL_PATH = "yolov5/runs/train/exp4/weights/best.pt"
model = torch.hub.load("yolov5", "custom", path=MODEL_PATH, source="local")

def detect_image(image_path):
    """Nhận diện đối tượng trên ảnh và lưu kết quả"""
    img = cv2.imread(image_path)
    results = model(img)
    
    output_filename = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    cv2.imwrite(output_path, results.render()[0])
    return output_filename

def detect_video(video_path):
    """Nhận diện đối tượng trong video và lưu video kết quả"""
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    output_filename = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        out.write(results.render()[0])

    cap.release()
    out.release()
    return output_filename

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return "No file uploaded", 400
    
    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400
    
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    if file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        result_filename = detect_image(filepath)
        result_type = "image"
    elif file.filename.lower().endswith((".mp4", ".avi", ".mov")):
        result_filename = detect_video(filepath)
        result_type = "video"
    else:
        return "Unsupported file type", 400

    return render_template("result.html", result_file=result_filename, result_type=result_type)

@app.route("/static/outputs/<filename>")
def get_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
