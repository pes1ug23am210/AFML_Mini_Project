# pipeline.py
import os
from pathlib import Path
import cv2
from detection import detect_and_crop_objects
from classification import classify_objects
from utils import deduplicate_objects_by_iou
from visualization import create_detection_visualization, create_object_grid_visualization
from config import get_product_price

def process_single_image(image_path, yolo_model, classifier_model, ocr_processor, output_dir="pipeline_results"):
    """Process a single image through the complete pipeline"""
    print(f"\n{'='*60}")
    print(f"PROCESSING: {Path(image_path).name}")
    print(f"{'='*60}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    image_stem = Path(image_path).stem
    output_path = os.path.join(output_dir, image_stem)
    
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
    
    # Step 2: Classifier + OCR Object Processing
    print("\n🎯 Step 2: Classifier & OCR Object Processing...")
    classified_objects = classify_objects(classifier_model, ocr_processor, detections)

    # Deduplicate heavily overlapping detections of the same class to
    # avoid counting the same product twice.
    classified_objects = deduplicate_objects_by_iou(classified_objects)
    
    print(f"✅ Processed {len(classified_objects)} unique objects with classifier + OCR")
    
    # Display enhanced classification results
    print("\n🎓 CLASSIFICATION & OCR RESULTS:")
    print("-" * 50)
    for i, obj in enumerate(classified_objects):
        print(f"🔵 Object {i+1}:")
        print(f"  Detected: {obj['final_class_name']} (conf: {obj['final_confidence']:.3f})")
        
        # Display OCR text only
        if obj['ocr_text']:
            print(f"  📝 OCR Text: {obj['ocr_text']}")
        else:
            print("  📝 OCR Text: None")
        print()
    
    # Step 3: Generate Enhanced Summary
    print("\n📊 ENHANCED PIPELINE SUMMARY:")
    print("-" * 50)
    objects_with_ocr = sum(1 for obj in classified_objects if obj['ocr_text'])
    
    ocr_success_rate = (objects_with_ocr/len(classified_objects))*100 if classified_objects else 0
    
    print(f"Total Objects: {len(classified_objects)}")
    print(f"OCR Success Rate: {ocr_success_rate:.1f}%")

    # Step 3b: Per-image billing summary - FIXED VERSION
    print("\n🧾 BILLING SUMMARY FOR THIS IMAGE:")
    print("-" * 50)
    bill_items = {}

    # Billing rules - FIXED VERSION:
    # 1. Use final_class_name and final_confidence
    # 2. Require final_confidence >= 0.7 (reduced from 0.8)
    # 3. OCR disagreement only matters if OCR is VERY confident (>= 0.8)
    # 4. Always use the final_class_name for billing (trust the model more)
    BILL_CONF_THRESHOLD = 0.7  # Reduced threshold
    OCR_STRONG_THRESHOLD = 0.8  # Increased threshold for conflicts

    # Debug: Show all objects and their status
    print("\n🔍 OBJECT ANALYSIS FOR BILLING:")
    for i, obj in enumerate(classified_objects):
        final_name = obj['final_class_name']
        final_conf = obj['final_confidence']
        ocr_name = obj.get('ocr_matched_class')
        ocr_conf = obj.get('ocr_match_confidence', 0)
        
        billable = final_conf >= BILL_CONF_THRESHOLD
        conflict = (ocr_name and ocr_name != final_name and ocr_conf >= OCR_STRONG_THRESHOLD)
        
        status = "✅ BILLABLE" if (billable and not conflict) else "❌ NOT BILLABLE"
        if conflict:
            status += f" (OCR conflict: {ocr_name})"
        elif not billable:
            status += f" (Low confidence: {final_conf:.3f})"
        
        print(f"  Object {i+1}: {final_name} - {status}")

    # Calculate billing
    for obj in classified_objects:
        final_name = obj['final_class_name']
        final_conf = obj['final_confidence']
        ocr_name = obj.get('ocr_matched_class')
        ocr_match_conf = obj.get('ocr_match_confidence', 0.0)

        # 1) Require reasonable final confidence
        if final_conf < BILL_CONF_THRESHOLD:
            continue

        # 2) Only reject if OCR is VERY confident AND disagrees
        # This gives more weight to the model predictions
        if (ocr_name is not None and 
            ocr_name != final_name and 
            ocr_match_conf >= OCR_STRONG_THRESHOLD):
            print(f"  ⚠️  Skipping {final_name} due to strong OCR conflict: {ocr_name}")
            continue

        # 3) Bill the object
        price = get_product_price(final_name)
        if final_name not in bill_items:
            bill_items[final_name] = {"count": 0, "price": price}
        bill_items[final_name]["count"] += 1

    total_amount = 0.0
    if not bill_items:
        print("No billable objects detected.")
    else:
        print(f"\n{'Product':25s} {'Qty':>5s} {'Price':>8s} {'Total':>10s}")
        print("-" * 50)
        for pname, info in bill_items.items():
            qty = info["count"]
            price = info["price"]
            line_total = qty * price
            total_amount += line_total
            print(f"{pname:25s} {qty:5d} {price:8.2f} {line_total:10.2f}")
        print("-" * 50)
        print(f"{'GRAND TOTAL':25s} {'':5s} {'':8s} {total_amount:10.2f}")
    
    # Step 4: Visualization (only if we have objects)
    if classified_objects:
        print("\n🖼️  Generating visualizations...")
        
        # Create detection visualization
        create_detection_visualization(original_image, classified_objects, output_path)
        
        # Create object grid visualization
        create_object_grid_visualization(classified_objects, output_path)
        
        # Save individual cropped objects
        print("\n💾 Saving individual objects...")
        for i, obj in enumerate(classified_objects):
            obj_filename = f"{output_path}_object_{i+1}_{obj['final_class_name']}.png"
            # Convert RGB to BGR for OpenCV saving
            cv2.imwrite(obj_filename, cv2.cvtColor(obj['cropped_image'], cv2.COLOR_RGB2BGR))
            print(f"  💿 Saved: {Path(obj_filename).name}")
    
    return classified_objects