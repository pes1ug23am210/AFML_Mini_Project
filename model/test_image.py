import torch
import cv2
import numpy as np
import os
from pathlib import Path
import yaml
from PIL import Image
import torchvision.transforms as transforms
from ultralytics import YOLO
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data.yaml configuration
with open('data.yaml', 'r') as file:
    data_config = yaml.safe_load(file)

# Get class names from the config
class_names = {int(k): v for k, v in data_config['names'].items()}
num_classes = data_config['nc']

print(f"Testing {num_classes} classes: {list(class_names.values())}")

# Define image transformations for classifier (must match training)
classifier_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_classifier_model(model_path, num_classes):
    """Load the EfficientNet-B0 classifier model"""
    import torch
    from torchvision.models import efficientnet_b0
    import torch.nn as nn
    
    # Load model architecture
    model = efficientnet_b0()
    model.classifier = nn.Sequential(
        nn.Dropout(0.2), 
        nn.Linear(model.classifier[1].in_features, num_classes)
    )
    
    # Load trained weights with weights_only=True to avoid warning
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()
    
    return model

def load_yolo_model(model_path):
    """Load YOLO model"""
    model = YOLO(model_path)
    return model

def predict_classifier(model, image_path, transform):
    """Make prediction using classifier model"""
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        return predicted.item(), confidence.item()
    except Exception as e:
        print(f"Classifier error on {image_path}: {e}")
        return -1, 0.0

def predict_yolo(model, image_path):
    """Make prediction using YOLO model"""
    try:
        results = model(image_path, verbose=False)
        
        if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
            # Get the detection with highest confidence
            boxes = results[0].boxes
            best_idx = torch.argmax(boxes.conf).item()
            class_id = int(boxes.cls[best_idx])
            confidence = float(boxes.conf[best_idx])
            return class_id, confidence
        else:
            return None, 0.0
    except Exception as e:
        print(f"YOLO error on {image_path}: {e}")
        return None, 0.0

def get_true_label_from_yolo_label(image_path):
    """Get true label from corresponding YOLO label file in test/labels/"""
    image_path = Path(image_path)
    
    # The label file should be in test/labels/ with same name but .txt extension
    labels_dir = Path('test/labels')
    label_file = labels_dir / f"{image_path.stem}.txt"
    
    if label_file.exists():
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    # Get the first object's class (YOLO format: class x_center y_center width height)
                    first_line = lines[0].strip()
                    if first_line:
                        class_id = int(first_line.split()[0])
                        return class_id
        except Exception as e:
            print(f"Error reading label file {label_file}: {e}")
            return -1
    
    # If no label file found, try to extract from filename as fallback
    print(f"Warning: No label file found for {image_path.name}")
    return extract_label_from_filename(image_path.name)

def extract_label_from_filename(filename):
    """Extract label from filename as fallback (for images without label files)"""
    name_without_ext = Path(filename).stem
    
    # Remove RoboFlow suffix if present
    if '.rf.' in name_without_ext:
        name_without_ext = name_without_ext.split('.rf.')[0]
    
    # Try to match with class names
    for class_id, class_name in class_names.items():
        clean_class_name = class_name.lower().replace('-', '').replace('_', '')
        clean_filename = name_without_ext.lower().replace('-', '').replace('_', '')
        
        # Check if class name appears in filename
        if clean_class_name in clean_filename:
            return class_id
    
    return -1

def test_models_on_folder(test_folder_path, yolo_model, classifier_model):
    """Test both models on all images in the test folder"""
    test_folder = Path(test_folder_path)
    
    # Find all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']
    image_files = []
    for ext in image_extensions:
        image_files.extend(test_folder.glob(ext))
    
    # Remove duplicates and sort
    image_files = list(set(image_files))
    image_files.sort()
    
    results = []
    
    print(f"Found {len(image_files)} images in test folder")
    
    if len(image_files) == 0:
        print("No images found! Please check the test folder path.")
        return results
    
    print("Testing models...")
    
    # Count label files found
    label_files_found = 0
    total_images = len(image_files)
    
    for i, image_path in enumerate(image_files):
        if i % 10 == 0:
            print(f"Processed {i}/{total_images} images...")
        
        # Get true label from YOLO label file
        true_label = get_true_label_from_yolo_label(image_path)
        true_class_name = class_names.get(true_label, "Unknown")
        
        # Count if we found a valid label
        if true_label != -1:
            label_files_found += 1
        
        # YOLO prediction
        yolo_class, yolo_conf = predict_yolo(yolo_model, str(image_path))
        yolo_class_name = class_names.get(yolo_class, "No detection") if yolo_class is not None else "No detection"
        
        # Classifier prediction
        classifier_class, classifier_conf = predict_classifier(classifier_model, str(image_path), classifier_transform)
        classifier_class_name = class_names.get(classifier_class, "Unknown") if classifier_class != -1 else "Error"
        
        # Determine if predictions are correct
        yolo_correct = (yolo_class == true_label) if yolo_class is not None and true_label != -1 else False
        classifier_correct = (classifier_class == true_label) if classifier_class != -1 and true_label != -1 else False
        
        result = {
            'image_path': str(image_path),
            'image_name': image_path.name,
            'true_label': true_label,
            'true_class': true_class_name,
            'yolo_pred': yolo_class,
            'yolo_class': yolo_class_name,
            'yolo_conf': yolo_conf,
            'yolo_correct': yolo_correct,
            'classifier_pred': classifier_class,
            'classifier_class': classifier_class_name,
            'classifier_conf': classifier_conf,
            'classifier_correct': classifier_correct,
            'has_label_file': true_label != -1
        }
        
        results.append(result)
    
    print(f"\nLabel files found: {label_files_found}/{total_images} images")
    
    return results

def calculate_metrics(results):
    """Calculate performance metrics for both models"""
    # Filter out images where we couldn't determine true label
    valid_results = [r for r in results if r['true_label'] != -1]
    
    if not valid_results:
        print("No valid results with known true labels!")
        print("This means label files are missing in test/labels/ folder.")
        return 0, 0
    
    print(f"\nAnalyzing {len(valid_results)} images with known labels...")
    
    # Extract predictions and true labels
    yolo_preds = [r['yolo_pred'] for r in valid_results if r['yolo_pred'] is not None]
    yolo_trues = [r['true_label'] for r in valid_results if r['yolo_pred'] is not None]
    yolo_correct = [r for r in valid_results if r['yolo_correct'] and r['yolo_pred'] is not None]
    
    classifier_preds = [r['classifier_pred'] for r in valid_results if r['classifier_pred'] != -1]
    classifier_trues = [r['true_label'] for r in valid_results if r['classifier_pred'] != -1]
    classifier_correct = [r for r in valid_results if r['classifier_correct'] and r['classifier_pred'] != -1]
    
    # Calculate accuracies
    yolo_accuracy = len(yolo_correct) / len(yolo_preds) * 100 if yolo_preds else 0
    classifier_accuracy = len(classifier_correct) / len(classifier_preds) * 100 if classifier_preds else 0
    
    print(f"\n=== PERFORMANCE SUMMARY ===")
    print(f"YOLO Model:")
    print(f"  - Accuracy: {yolo_accuracy:.2f}% ({len(yolo_correct)}/{len(yolo_preds)} correct)")
    print(f"  - Detection rate: {len(yolo_preds)/len(valid_results)*100:.2f}%")
    
    print(f"\nClassifier Model:")
    print(f"  - Accuracy: {classifier_accuracy:.2f}% ({len(classifier_correct)}/{len(classifier_preds)} correct)")
    print(f"  - Success rate: {len(classifier_preds)/len(valid_results)*100:.2f}%")
    
    # Detailed classification reports
    if yolo_preds and yolo_trues:
        print(f"\n=== YOLO DETAILED REPORT ===")
        print(classification_report(yolo_trues, yolo_preds, 
                                  target_names=[class_names[i] for i in range(num_classes)], 
                                  labels=list(range(num_classes)), zero_division=0))
    
    if classifier_preds and classifier_trues:
        print(f"\n=== CLASSIFIER DETAILED REPORT ===")
        print(classification_report(classifier_trues, classifier_preds, 
                                  target_names=[class_names[i] for i in range(num_classes)], 
                                  labels=list(range(num_classes)), zero_division=0))
    
    return yolo_accuracy, classifier_accuracy

def plot_confusion_matrices(results):
    """Plot confusion matrices for both models"""
    valid_results = [r for r in results if r['true_label'] != -1]
    
    if not valid_results:
        print("No valid results to plot confusion matrices.")
        return
    
    # YOLO confusion matrix
    yolo_preds = [r['yolo_pred'] for r in valid_results if r['yolo_pred'] is not None]
    yolo_trues = [r['true_label'] for r in valid_results if r['yolo_pred'] is not None]
    
    # Classifier confusion matrix
    classifier_preds = [r['classifier_pred'] for r in valid_results if r['classifier_pred'] != -1]
    classifier_trues = [r['true_label'] for r in valid_results if r['classifier_pred'] != -1]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    if yolo_preds and yolo_trues:
        cm_yolo = confusion_matrix(yolo_trues, yolo_preds, labels=list(range(num_classes)))
        sns.heatmap(cm_yolo, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=[class_names[i] for i in range(num_classes)],
                   yticklabels=[class_names[i] for i in range(num_classes)])
        ax1.set_title('YOLO Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('True')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax1.get_yticklabels(), rotation=0)
    
    if classifier_preds and classifier_trues:
        cm_classifier = confusion_matrix(classifier_trues, classifier_preds, labels=list(range(num_classes)))
        sns.heatmap(cm_classifier, annot=True, fmt='d', cmap='Blues', ax=ax2,
                   xticklabels=[class_names[i] for i in range(num_classes)],
                   yticklabels=[class_names[i] for i in range(num_classes)])
        ax2.set_title('Classifier Confusion Matrix')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('True')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax2.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    plt.savefig('model_comparison_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_detailed_results(results, filename='model_test_results.csv'):
    """Save detailed results to CSV"""
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"\nDetailed results saved to {filename}")

def check_label_files():
    """Check if label files exist in test/labels/"""
    labels_dir = Path('test/labels')
    if not labels_dir.exists():
        print(f"❌ Labels directory not found: {labels_dir}")
        return False
    
    label_files = list(labels_dir.glob('*.txt'))
    print(f"Found {len(label_files)} label files in test/labels/")
    
    if len(label_files) == 0:
        print("No label files found! Please check if test/labels/ contains .txt files")
        return False
    
    return True

def main():
    # Check if label files exist
    if not check_label_files():
        print("Continuing with filename-based label extraction as fallback...")
    
    # Load models
    print("Loading models...")
    
    # Load YOLO model
    yolo_model = load_yolo_model('best.pt')
    print("✓ YOLO model loaded")
    
    # Load classifier model
    classifier_model = load_classifier_model('classifier_best.pth', num_classes)
    print("✓ Classifier model loaded")
    
    # Test on test/images folder
    test_folder = Path('test/images')
    if not test_folder.exists():
        print("❌ Test images folder not found: test/images/")
        return
    
    print(f"Testing models on: {test_folder}")
    results = test_models_on_folder(test_folder, yolo_model, classifier_model)
    
    if not results:
        print("No results to analyze!")
        return
    
    # Calculate and display metrics
    yolo_acc, classifier_acc = calculate_metrics(results)
    
    # Plot confusion matrices
    plot_confusion_matrices(results)
    
    # Save detailed results
    save_detailed_results(results)
    
    # Print some example results
    valid_results = [r for r in results if r['true_label'] != -1]
    if valid_results:
        print(f"\n=== SAMPLE PREDICTIONS (first 10 with labels) ===")
        for i, result in enumerate(valid_results[:10]):
            print(f"\nImage: {result['image_name']}")
            print(f"True: {result['true_class']} (ID: {result['true_label']})")
            print(f"YOLO: {result['yolo_class']} (conf: {result['yolo_conf']:.3f}) - {'✓' if result['yolo_correct'] else '✗'}")
            print(f"Classifier: {result['classifier_class']} (conf: {result['classifier_conf']:.3f}) - {'✓' if result['classifier_correct'] else '✗'}")

if __name__ == '__main__':
    main()