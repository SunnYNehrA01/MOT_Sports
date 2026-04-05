import cv2
from ultralytics import YOLO
import yaml
import os

class SportsDetector:
    def __init__(self, model_weights='yolo11n.pt', config_path='config/sports_map.yaml'):
        """
        Initializes the YOLOv11 detector with class filtering capabilities.
        """
        self.model = YOLO(model_weights)
        with open(config_path, 'r') as f:
            self.sports_config = yaml.safe_load(f)

    def get_classes_for_sport(self, sport_type):
        """Returns the list of COCO class IDs for the selected sport."""
        sport_data = self.sports_config.get(sport_type.lower(), self.sports_config['general'])
        return sport_data['classes']

    def detect(self, frame, sport_type='general', conf_threshold=0.25):
        """
        Performs inference and returns detections filtered by sport.
        Returns: list of [x1, y1, x2, y2, confidence, class_id]
        """
        target_classes = self.get_classes_for_sport(sport_type)
        
        # Run YOLO inference
        results = self.model.predict(
            source=frame, 
            conf=conf_threshold, 
            classes=target_classes, 
            iou=0.45,
            imgsz=1280, # IMPORTANT: Increase from default 640
            verbose=False,
            augment=True # Test-time augmentation for small objects
        )
        
        detections = []
        for r in results:
            boxes = r.boxes.cpu().numpy()
            for box in boxes:
                # Format: [x1, y1, x2, y2, conf, cls]
                detection = [*box.xyxy[0], box.conf[0], int(box.cls[0])]
                detections.append(detection)
                
        return detections