# detection.py
import cv2
from ultralytics import YOLO
from config import class_names

def detect_and_crop_objects(yolo_model, image_path, confidence_threshold=0.25):
    """Detect objects using YOLO and return cropped regions"""
    results = yolo_model(image_path, conf=confidence_threshold, verbose=False)
    
    if len(results) == 0 or results[0].boxes is None:
        return [], None
    
    # Get the original image
    original_image = cv2.imread(image_path)
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    detections = []
    boxes = results[0].boxes
    
    for i, box in enumerate(boxes):
        # Get bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = class_names.get(class_id, "Unknown")
        
        # Convert to integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Crop the object from original image
        cropped_object = original_image_rgb[y1:y2, x1:x2]
        
        # Add padding to ensure minimum size
        h, w = cropped_object.shape[:2]
        if h < 50 or w < 50:
            # Skip very small detections
            continue
        
        detection_info = {
            'bbox': (x1, y1, x2, y2),
            'confidence': confidence,
            'class_id': class_id,
            'class_name': class_name,
            'cropped_image': cropped_object,
            'detection_id': i
        }
        
        detections.append(detection_info)
    
    return detections, original_image_rgb