
import os
os.environ["WANDB_DISABLED"] = "true"


# Đường dẫn thư mục YOLOv5
YOLOV5_PATH = os.path.join(os.getcwd(), "yolov5")

# Huấn luyện YOLOv5
os.system(f"python {YOLOV5_PATH}/train.py --img 640 --batch 16 --epochs 50 --data bookshelf-data/data.yaml --weights yolov5s.pt --device cpu")
