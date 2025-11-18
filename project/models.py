# models.py
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# Define image transformations for classifier
classifier_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_classifier_model(model_path, num_classes):
    """Load the EfficientNet-B0 classifier model"""
    model = efficientnet_b0()
    model.classifier = nn.Sequential(
        nn.Dropout(0.2), 
        nn.Linear(model.classifier[1].in_features, num_classes)
    )
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint)
    model.eval()
    
    return model

def load_yolo_model(model_path):
    """Load YOLO model"""
    model = YOLO(model_path)
    return model

def predict_classifier(model, image, transform):
    """Make prediction using classifier model on PIL Image"""
    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        image_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        return predicted.item(), confidence.item()
    except Exception as e:
        print(f"Classifier error: {e}")
        return -1, 0.0