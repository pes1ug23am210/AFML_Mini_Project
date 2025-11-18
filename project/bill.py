# bill.py - Integrated Billing System with One-Class Model Verification
import os
from pathlib import Path
from multiprocessing import freeze_support
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
from datetime import datetime
import pandas as pd

from config import class_names, num_classes, get_product_price
from models import load_yolo_model, load_classifier_model
from ocr import OCRProcessor
from detection import detect_and_crop_objects
from classification import classify_objects
from utils import deduplicate_objects_by_iou
from visualization import create_detection_visualization, create_object_grid_visualization

# One-Class Model Architecture
class SVDDNet(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.backbone = models.efficientnet_b0(pretrained=True)
        self.backbone.classifier = nn.Identity()
        for param in self.backbone.features[:4].parameters():
            param.requires_grad = False
        self.fc = nn.Linear(1280, latent_dim)
        self.center = nn.Parameter(torch.zeros(1, latent_dim))
    
    def forward(self, x):
        feats = self.backbone(x)
        z = self.fc(feats)
        dist = torch.sum((z - self.center) ** 2, dim=1)
        return dist, z

class OneClassModelManager:
    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.load_all_models()
    
    def load_all_models(self):
        """Load all one-class models from models directory"""
        if not self.models_dir.exists():
            print(f"⚠️  Models directory not found: {self.models_dir}")
            return
        
        model_files = list(self.models_dir.glob("*.pth"))
        if not model_files:
            print(f"⚠️  No .pth files found in {self.models_dir}")
            return
        
        print(f"🔍 Searching for one-class models in {self.models_dir}...")
        
        for model_path in model_files:
            class_name = model_path.stem
            
            try:
                ckpt = torch.load(model_path, map_location='cpu')
                model = SVDDNet()
                model.load_state_dict(ckpt['model_state'])
                model.center.data = ckpt['center']
                model.eval()
                
                self.models[class_name] = {
                    'model': model,
                    'threshold': ckpt['threshold']
                }
                print(f"✅ Loaded one-class model: {class_name}")
                
            except Exception as e:
                print(f"❌ Failed to load model {class_name}: {e}")
        
        print(f"📦 Total one-class models loaded: {len(self.models)}")
    
    def predict_single_image(self, image, model_name):
        """Run single image through specific one-class model"""
        if model_name not in self.models:
            return None, 0.0
        
        model_info = self.models[model_name]
        model = model_info['model']
        threshold = model_info['threshold']
        
        try:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            tensor = self.transform(image).unsqueeze(0)
            
            with torch.no_grad():
                dist, _ = model(tensor)
                dist_value = dist.item()
                
                if dist_value < threshold:
                    conf = 1 - (dist_value / threshold)
                else:
                    conf = np.exp(- (dist_value - threshold) / threshold)
                
                conf = max(0, min(1, conf))
                return dist_value, conf
                
        except Exception as e:
            print(f"❌ Prediction error for {model_name}: {e}")
            return None, 0.0
    
    def predict_all_models(self, image):
        """Run image through all available one-class models"""
        results = {}
        
        for model_name in self.models.keys():
            distance, confidence = self.predict_single_image(image, model_name)
            if distance is not None:
                results[model_name] = {
                    'distance': distance,
                    'confidence': confidence,
                    'is_match': confidence > 0.35
                }
        
        return results

def process_image_for_billing(image_path, yolo_model, classifier_model, ocr_processor, oneclass_manager):
    """Process single image and return objects with one-class verification"""
    print(f"\n{'='*60}")
    print(f"PROCESSING: {Path(image_path).name}")
    print(f"{'='*60}")
    
    # Step 1: YOLO Object Detection
    print("🔍 Step 1: YOLO Object Detection...")
    detections, original_image = detect_and_crop_objects(yolo_model, image_path)
    
    if not detections:
        print("❌ No objects detected!")
        return None
    
    print(f"✅ Detected {len(detections)} objects")
    
    # Display YOLO detection results
    print("\n📦 YOLO DETECTION RESULTS:")
    print("-" * 50)
    for i, det in enumerate(detections):
        print(f"Object {i+1}: {det['class_name']} (conf: {det['confidence']:.3f}) "
              f"at {det['bbox']}")
    
    # Step 2: Classifier + OCR Processing
    print("\n🎯 Step 2: Classifier & OCR Processing...")
    classified_objects = classify_objects(classifier_model, ocr_processor, detections)
    classified_objects = deduplicate_objects_by_iou(classified_objects)
    
    print(f"✅ Processed {len(classified_objects)} unique objects with classifier + OCR")
    
    # Display enhanced classification results
    print("\n🎓 CLASSIFICATION & OCR RESULTS:")
    print("-" * 50)
    for i, obj in enumerate(classified_objects):
        print(f"🔵 Object {i+1}:")
        print(f"  Detected: {obj['final_class_name']} (conf: {obj['final_confidence']:.3f})")
        
        if obj['ocr_text']:
            print(f"  📝 OCR Text: {obj['ocr_text']}")
        else:
            print("  📝 OCR Text: None")
        print()
    
    # Step 3: One-Class Model Verification
    print("\n🔬 Step 3: One-Class Model Verification...")
    print("-" * 50)
    
    for i, obj in enumerate(classified_objects):
        print(f"\n🔄 Testing Object {i+1} ({obj['final_class_name']}) against one-class models:")
        
        # Run through all one-class models
        oneclass_results = oneclass_manager.predict_all_models(obj['cropped_image'])
        
        if not oneclass_results:
            print("   No one-class models available for testing")
            obj['oneclass_matches'] = []
            continue
        
        # Find all matches
        matches = []
        for model_name, result in oneclass_results.items():
            if result['is_match']:
                matches.append({
                    'model_name': model_name,
                    'confidence': result['confidence'],
                    'distance': result['distance']
                })
                print(f"   ✅ MATCH with {model_name} (conf: {result['confidence']:.3f})")
        
        obj['oneclass_matches'] = matches
        
        if not matches:
            print("   ❌ No one-class matches found")
    
    # Step 4: Show visualizations (but don't save files)
    if classified_objects:
        print("\n🖼️  Showing detection visualizations...")
        
        # Create and show detection visualization (temporary display only)
        create_detection_visualization(original_image, classified_objects, None)
        
        # Create and show object grid visualization (temporary display only)
        create_object_grid_visualization(classified_objects, None)
    
    return classified_objects

def generate_excel_bill(classified_objects, image_filename, output_dir="bill_image"):
    """Generate bill with confirmation system and save as Excel"""
    
    # Create bill directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n🧾 GENERATING BILL FOR: {image_filename}")
    print("=" * 60)
    
    # Collect billable items
    bill_items = {}
    BILL_CONF_THRESHOLD = 0.7
    
    for i, obj in enumerate(classified_objects):
        final_name = obj['final_class_name']
        final_conf = obj['final_confidence']
        
        # Skip low confidence detections
        if final_conf < BILL_CONF_THRESHOLD:
            continue
        
        # Get one-class confirmation
        oneclass_confirmation = ""
        if obj['oneclass_matches']:
            best_match = max(obj['oneclass_matches'], key=lambda x: x['confidence'])
            oneclass_confirmation = f" (possible product: {best_match['model_name']})"
        
        # Create display name
        display_name = f"{final_name}{oneclass_confirmation}"
        
        price = get_product_price(final_name)
        
        # Correct checking needed logic
        needs_checking = "Yes" if oneclass_confirmation else "No"
        
        if display_name not in bill_items:
            bill_items[display_name] = {
                "base_class": final_name,
                "oneclass_confirm": oneclass_confirmation,
                "count": 0, 
                "price": price,
                "needs_checking": needs_checking
            }
        bill_items[display_name]["count"] += 1
    
    if not bill_items:
        print("❌ No billable objects detected!")
        return None
    
    # Prepare data for Excel
    excel_data = []
    total_amount = 0.0
    
    print(f"\n{'Product':40s} {'Qty':>5s} {'Price':>8s} {'Total':>10s} {'Check Needed':>12s}")
    print("-" * 85)
    
    for display_name, info in bill_items.items():
        qty = info["count"]
        price = info["price"]
        line_total = qty * price
        total_amount += line_total
        check_needed = info["needs_checking"]
        
        print(f"{display_name:40s} {qty:5d} {price:8.2f} {line_total:10.2f} {check_needed:>12s}")
        
        # Store for Excel
        excel_data.append({
            'Product Name': display_name,
            'Quantity': qty,
            'Price (₹)': price,
            'Total (₹)': line_total,
            'Checking Needed': check_needed
        })
    
    print("-" * 85)
    print(f"{'GRAND TOTAL':40s} {'':5s} {'':8s} {total_amount:10.2f} {'':>12s}")
    
    # Add grand total to Excel data
    excel_data.append({
        'Product Name': 'GRAND TOTAL',
        'Quantity': '',
        'Price (₹)': '',
        'Total (₹)': total_amount,
        'Checking Needed': ''
    })
    
    # Create DataFrame
    df = pd.DataFrame(excel_data)
    
    # Save as Excel file only (no CSV)
    excel_filename = f"{output_dir}/{Path(image_filename).stem}_bill.xlsx"
    
    # Create Excel writer with formatting
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Bill', index=False)
        
        # Get workbook and worksheet for formatting
        workbook = writer.book
        worksheet = writer.sheets['Bill']
        
        # Auto-adjust column widths
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = (max_length + 2)
            worksheet.column_dimensions[column_letter].width = adjusted_width
        
        # Add header formatting
        for cell in worksheet[1]:
            cell.font = cell.font.copy(bold=True)
        
        # Format currency columns
        for row in range(2, worksheet.max_row + 1):
            worksheet[f'C{row}'].number_format = '#,##0.00'
            worksheet[f'D{row}'].number_format = '#,##0.00'
    
    print(f"\n💾 Excel bill saved as: {excel_filename}")
    
    return df

def main():
    """Main billing system"""
    print("🧾 INTEGRATED BILLING SYSTEM")
    print("YOLO + Classifier + OCR + One-Class Verification")
    print("=" * 70)
    
    # Load all models
    print("📦 Loading models...")
    yolo_model = load_yolo_model('best.pt')
    classifier_model = load_classifier_model('classifier_best.pth', num_classes)
    ocr_processor = OCRProcessor()
    
    # Load one-class models
    print("\n🔬 Loading One-Class Models...")
    oneclass_manager = OneClassModelManager("models")
    
    print("✅ All models loaded successfully!")
    
    # Find images in bill_image folder
    bill_folder = Path('bill_image')
    if not bill_folder.exists():
        print(f"❌ Bill image folder not found: {bill_folder}")
        print("💡 Creating bill_image folder...")
        bill_folder.mkdir(exist_ok=True)
        print("📁 Please place your image in the 'bill_image' folder and run again.")
        return
    
    # Find images in bill_image folder
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']
    image_files = []
    for ext in image_extensions:
        image_files.extend(bill_folder.glob(ext))
    
    if not image_files:
        print("❌ No images found in bill_image folder!")
        print("💡 Please place your image in the 'bill_image' folder.")
        return
    
    print(f"📁 Found {len(image_files)} images in bill_image folder")
    
    # Process each image
    all_bills = []
    
    for i, image_path in enumerate(image_files):
        print(f"\n📸 Processing image {i+1}/{len(image_files)}: {image_path.name}")
        
        # Process image through pipeline
        classified_objects = process_image_for_billing(
            str(image_path), yolo_model, classifier_model, ocr_processor, oneclass_manager
        )
        
        if not classified_objects:
            print("❌ No objects to bill!")
            continue
        
        # Generate Excel bill
        bill_df = generate_excel_bill(classified_objects, image_path.name)
        
        if bill_df is not None:
            all_bills.append(bill_df)
            print(f"\n🎉 Excel bill generated successfully for {image_path.name}!")
            
            # Show summary
            total_items = len([item for item in bill_df.to_dict('records') if item['Product Name'] != 'GRAND TOTAL'])
            items_needing_check = len([item for item in bill_df.to_dict('records') 
                                     if item.get('Checking Needed') == 'Yes' and item['Product Name'] != 'GRAND TOTAL'])
            
            print(f"\n📊 SUMMARY:")
            print(f"  Total items: {total_items}")
            print(f"  Items needing checking: {items_needing_check}")
            print(f"  Confirmed by one-class models: {total_items - items_needing_check}")
        
        # Ask to continue if multiple images
        if i < len(image_files) - 1:
            response = input(f"\n⏭️  Process next image? (y/n): ").lower().strip()
            if response != 'y':
                print("⏹️  Billing stopped by user.")
                break
    
    # Create combined bill if multiple images processed
    if len(all_bills) > 1:
        print(f"\n📋 Creating combined bill for all {len(all_bills)} images...")
        combined_df = pd.concat(all_bills, ignore_index=True)
        combined_filename = f"bill_image/combined_bill_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        combined_df.to_excel(combined_filename, index=False)
        print(f"💾 Combined bill saved as: {combined_filename}")
    
    # Final summary
    if all_bills:
        print(f"\n🎉 BILLING PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"📈 Total images processed: {len(all_bills)}")
        print(f"🧾 All bills saved in 'bill_image/' folder")

if __name__ == '__main__':
    freeze_support()
    main()