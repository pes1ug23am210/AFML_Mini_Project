# main.py
import os
from pathlib import Path
from multiprocessing import freeze_support
from config import class_names, num_classes
from models import load_yolo_model, load_classifier_model
from ocr import OCRProcessor
from pipeline import process_single_image

def main():
    """Main pipeline execution"""
    print("🚀 ENHANCED OBJECT DETECTION & CLASSIFICATION PIPELINE")
    print("YOLO + Classifier + OCR Integration")
    print("=" * 70)
    
    # Load all models
    print("📦 Loading models...")
    yolo_model = load_yolo_model('best.pt')
    classifier_model = load_classifier_model('classifier_best.pth', num_classes)
    ocr_processor = OCRProcessor()
    print("✅ All models loaded successfully!")
    
    # Find images in cluster folder
    cluster_folder = Path('cluster')
    if not cluster_folder.exists():
        print(f"❌ Cluster folder not found: {cluster_folder}")
        return
    
    # Find all images in cluster folder
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']
    image_files = []
    for ext in image_extensions:
        image_files.extend(cluster_folder.glob(ext))
    
    if not image_files:
        print("❌ No images found in cluster folder!")
        return
    
    print(f"📁 Found {len(image_files)} images in cluster folder")
    
    # Process each image
    all_results = []
    
    for i, image_path in enumerate(image_files):
        print(f"\n📸 Processing image {i+1}/{len(image_files)}")
        results = process_single_image(str(image_path), yolo_model, classifier_model, ocr_processor)
        if results:
            all_results.extend(results)
        
        # Ask user if they want to continue after each image
        if i < len(image_files) - 1:
            response = input(f"\n⏭️  Process next image? (y/n): ").lower().strip()
            if response != 'y':
                print("⏹️  Pipeline stopped by user.")
                break
    
    # Final enhanced summary
    if all_results:
        print(f"\n🎉 ENHANCED PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"📈 Total objects processed: {len(all_results)}")
        
        # Calculate overall statistics
        objects_with_ocr = sum(1 for obj in all_results if obj['ocr_text'])
        
        ocr_success_rate = (objects_with_ocr / len(all_results)) * 100
        
        print(f"🔤 Overall OCR Success Rate: {ocr_success_rate:.1f}%")
        
        print(f"\n💾 All results saved in 'pipeline_results/' folder")
        
        # Show final statistics
        print(f"\n📊 FINAL ENHANCED STATISTICS:")
        print("-" * 35)
        print(f"Total images processed: {min(i+1, len(image_files))}")
        print(f"Total objects detected: {len(all_results)}")

if __name__ == '__main__':
    freeze_support()
    main()