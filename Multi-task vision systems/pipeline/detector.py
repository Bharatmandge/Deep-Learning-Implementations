import cv2
import numpy as np 
from ultralytics import YOLO

class Detector:
    MODEL_SIZES = {'n', 's', 'm', 'l', 'x'}
    
    def __init__(self, model_size='n', confidence_threshold=0.5):
        if model_size not in self.MODEL_SIZES:
            raise ValueError(f"Invalid model size '{model_size}'. Choose from {self.MODEL_SIZES}.")
        self.conf_thresh = confidence_threshold
        model_file = f"yolov8{model_size}.pt"
 
        # Ultralytics auto-downloads the weights file on first run
        print(f"[Detector] Loading YOLOv8-{model_size} (downloads if not cached)...")
        self.model = YOLO(model_file)
        print(f"[Detector] Ready. Confidence threshold: {confidence_threshold}")
        
    def detect(self, image_bgr):
        results = self.model(image_bgr, conf=self.conf_thresh, verbose=False)
        results = results[0]
        
        boxes_tensor = results.boxes.xyxy.cpu().numpy()
        confs_tensor = results.boxes.conf.cpu().numpy()
        cls_tensor = results.boxes.cls.cpu().numpy()
        
        class_names = [results.names[int(c)] for c in cls_tensor]
        
        annotated = image_bgr.copy()
        for i, box in enumerate(boxes_tensor):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated, (x1, y1), (x2, y2),
                          color=(0, 200, 0), thickness=2)
            
            label = f"{class_names[i]} {confs_tensor[i]:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            lw, lh = label_size
            
            cv2.rectangle(annotated, 
                          (x1, y1 -lh - 8), (x1 + lw + 4, y1),
                          color=(0, 0, 0), thickness=-1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        n = len(boxes_tensor)
        print(f"[Detector] Found {n}  object(s): {class_names}")
        if n == 0:
            print("[Detector] no objects detected")
            print( "  try a lower confidence threshold or a different image")
            
        return boxes_tensor, confs_tensor, class_names, confs_tensor.tolist()
    
    def get_primary_box(self, boxes, strategy='largest'):
        if len(boxes) == 0:
            return None
        if strategy == 'largest':
            areas = [(b[2]-b[0]) * (b[3]-b[1]) for b in boxes]
            idx = int(np.argmax(areas))
            
        elif strategy == 'most_confident':
            idx = 0
        else:
            idx = 0
        return list(map(int, boxes[idx]))   