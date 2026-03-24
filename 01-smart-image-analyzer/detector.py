from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def detect_objects(img_path):
    results = model(img_path)
    return results[0].plot()