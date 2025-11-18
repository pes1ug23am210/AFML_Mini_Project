# classification.py
import torch
from PIL import Image
import cv2
from models import predict_classifier, classifier_transform
from ocr import OCRProcessor
from config import class_names

def classify_objects(classifier_model, ocr_processor, detections):
    """Classify cropped objects using classifier model AND OCR"""
    classified_objects = []
    
    for detection in detections:
        cropped_image = detection['cropped_image']
        
        # Convert numpy array to PIL Image for classifier
        pil_image = Image.fromarray(cropped_image)
        
        # Get classifier prediction
        classifier_class, classifier_conf = predict_classifier(
            classifier_model, pil_image, classifier_transform
        )
        classifier_class_name = class_names.get(classifier_class, "Unknown")
        
        # Determine final class and confidence: higher between YOLO and Classifier
        yolo_conf = detection['confidence']
        if classifier_conf > yolo_conf:
            final_class_name = classifier_class_name
            final_confidence = classifier_conf
        else:
            final_class_name = detection['class_name']
            final_confidence = yolo_conf
        
        # Get OCR text extraction and matching
        ocr_text, ocr_confidence = ocr_processor.extract_text_from_object(cropped_image)
        ocr_matched_class, ocr_match_confidence = ocr_processor.match_text_to_class(ocr_text)

        # If OCR suggests a brand that conflicts with a very confident
        # final prediction, trust the final and drop the OCR match.
        # This prevents cases like Himalaya face wash being labeled as 'sprite'.
        if (ocr_matched_class is not None
                and ocr_matched_class != final_class_name
                and final_confidence >= 0.8):
            ocr_matched_class = None
            ocr_match_confidence = 0.0

        # Fallback: if OCR couldn't confidently match any class but
        # final class is determined, treat that as an OCR-supported match with modest confidence.
        if ocr_matched_class is None:
            ocr_matched_class = final_class_name
            # Use a conservative fallback confidence if OCR text exists,
            # otherwise keep it very low to signal weak OCR support.
            ocr_match_confidence = max(ocr_confidence, 0.5 if ocr_text else 0.3)

        # Combine detection and classification info
        object_info = {
            **detection,
            'classifier_class_id': classifier_class,
            'classifier_class_name': classifier_class_name,
            'classifier_confidence': classifier_conf,
            'final_class_name': final_class_name,
            'final_confidence': final_confidence,
            'ocr_text': ocr_text,
            'ocr_confidence': ocr_confidence,
            'ocr_matched_class': ocr_matched_class,
            'ocr_match_confidence': ocr_match_confidence,
            'ocr_final_match': final_class_name == ocr_matched_class if ocr_matched_class else False
        }
        
        classified_objects.append(object_info)
    
    return classified_objects