# utils.py
import numpy as np

def _bbox_iou(box1, box2):
    """Compute IoU between two boxes (x1, y1, x2, y2)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h

    if inter == 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def deduplicate_objects_by_iou(classified_objects, iou_threshold: float = 0.7):
    """Remove duplicate detections that heavily overlap.

    Keeps the highest-confidence detection (based on final confidence) when multiple boxes with essentially the
    same class overlap each other. Two detections are considered the same product if either their YOLO class names
    OR their final class names match.
    """
    if not classified_objects:
        return classified_objects

    # Sort by final_confidence descending
    sorted_objs = sorted(
        classified_objects,
        key=lambda o: o.get('final_confidence', 0.0),
        reverse=True,
    )

    kept = []
    for obj in sorted_objs:
        bbox = obj['bbox']
        yolo_name = obj['class_name']
        final_name = obj.get('final_class_name')
        duplicate = False
        for k in kept:
            same_yolo = (k['class_name'] == yolo_name)
            same_final = (k.get('final_class_name') == final_name)
            if not (same_yolo or same_final):
                continue
            if _bbox_iou(bbox, k['bbox']) >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(obj)

    return kept