# Grocery Recognition Project - Phase 1: Multi-Class YOLOv8 Training (Fixed Per-Class Metrics)
# Key Fixes:
# - Per-class metrics: Use 'val_results.box.maps' (per-class mAP@0.5:0.95) instead of 'mp' (scalar overall).
#   - If no per-class array, gracefully skip with a message.
# - Retained all prior fixes: if __name__, workers=0, paths, plots=True.
# - Added optional inference conf threshold tuning note.
# Other: Training succeeded! Your model hit excellent metrics (98% mAP@0.5)—great for testing. Proceed to export/mobile.
# Prerequisites: pip install ultralytics
# Hardware: RTX 3050 or equivalent GPU

import os
import yaml
import torch
from multiprocessing import freeze_support  # For Windows compatibility
from ultralytics import YOLO

def main():
    # Step 1: Configure Dataset Path
    data_path = 'data.yaml'  # Points to your Roboflow-exported data.yaml in current dir
    dataset_root = os.path.dirname(os.path.abspath(__file__))  # Use script dir as root for consistency

    # Verify dataset structure
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"data.yaml not found at {data_path}. Ensure you've unzipped dataset-yolov8.zip to the current directory.")

    # Load and print dataset info from YAML
    with open(data_path, 'r') as f:
        dataset_config = yaml.safe_load(f)

    # Handle 'names' as list or dict (Roboflow sometimes uses list: ['class0', 'class1'])
    names_raw = dataset_config.get('names', [])
    if isinstance(names_raw, list):
        names = {i: name for i, name in enumerate(names_raw)}  # Convert to dict for Ultralytics compatibility
        dataset_config['names'] = names  # Update config for consistency
        class_names = names_raw
    else:
        class_names = list(names_raw.values())

    print("Dataset Config Loaded:")
    print(f"Path: {dataset_config.get('path', 'N/A')}")
    print(f"Train: {dataset_config.get('train', 'N/A')}")
    print(f"Val: {dataset_config.get('val', 'N/A')}")
    print(f"Test: {dataset_config.get('test', 'N/A')}")
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")  # Safe print for list or dict values

    # Fix relative paths: Set absolute root and adjust splits to absolute if needed
    if dataset_config.get('path') is None or dataset_config.get('path') == 'N/A':
        dataset_config['path'] = dataset_root  # Set to script dir (adjust if your splits are elsewhere, e.g., '../dataset')

    # Verify key paths exist post-fix
    train_img_path = os.path.join(dataset_config['path'], dataset_config.get('train', '').replace('/images', ''))
    val_img_path = os.path.join(dataset_config['path'], dataset_config.get('val', '').replace('/images', ''))
    print(f"\nPath Verification:")
    print(f"Train images dir: {train_img_path} (exists: {os.path.exists(train_img_path)})")
    print(f"Val images dir: {val_img_path} (exists: {os.path.exists(val_img_path)})")

    if not os.path.exists(train_img_path) or not os.path.exists(val_img_path):
        print("Warning: Paths may still be incorrect. Manually adjust dataset_config['path'] to your dataset root.")

    # Save updated YAML if modified (for reuse)
    with open(data_path, 'w') as f:
        yaml.dump(dataset_config, f)
    print(f"Updated data.yaml saved (with fixed paths and names as dict).")

    # Step 2: Load Pre-trained YOLOv8n Model
    model = YOLO('yolov8n.pt')  # Load nano variant (fast for mobile)

    # Step 3: Train the Model
    # Hyperparameters tuned for general detection (adjust based on your test dataset size/classes)
    # FIXED: workers=0 to avoid multiprocessing issues on Windows; reduce batch if OOM
    results = model.train(
        data=data_path,           # Path to your Roboflow YAML
        epochs=150,               # More epochs for fine-tuning; reduce for quick tests (e.g., 50)
        imgsz=640,                # Image size (balance speed/accuracy)
        batch=16,                 # Batch size (adjust for GPU memory; lower if OOM, e.g., 8)
        lr0=0.01,                 # Initial learning rate
        patience=20,              # Early stopping patience
        device=0 if torch.cuda.is_available() else 'cpu',  # Use GPU if available
        project='grocery_yolo_runs',  # Save dir (reuse for project later)
        name='roboflow_test_v1',  # Run name (for your test dataset)
        augment=True,             # Enable augmentations (mosaic, flip, etc.)
        mosaic=1.0,               # Mosaic augmentation probability
        mixup=0.1,                # Mixup for class mixing
        copy_paste=0.1,           # Copy-paste for occlusions
        fliplr=0.5,               # Horizontal flip
        degrees=15.0,             # Rotation degrees
        translate=0.1,            # Translation
        scale=0.5,                # Scale
        shear=2.0,                # Shear
        perspective=0.0001,       # Perspective
        hsv_h=0.015,              # HSV hue
        hsv_s=0.7,                # HSV saturation
        hsv_v=0.4,                # HSV value
        plots=True,               # Generate training plots
        save_period=10,           # Save checkpoint every 10 epochs
        workers=0                  # Disable multiprocessing dataloader for Windows stability
    )

    # Step 4: Validate the Model
    val_results = model.val()  # Run validation on val set (uses data.yaml)
    print("Validation Results:")
    print(f"mAP@0.5: {val_results.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {val_results.box.map:.4f}")
    print(f"Precision: {val_results.box.p.mean():.4f}")
    print(f"Recall: {val_results.box.r.mean():.4f}")

    # Per-class metrics (for testing insights) - FIXED: Use .maps for per-class mAP@0.5:0.95 array
    print("\nPer-Class mAP@0.5:0.95:")
    if hasattr(val_results.box, 'maps') and val_results.box.maps is not None and len(val_results.box.maps) > 1:
        for i, mp in enumerate(val_results.box.maps):
            class_name = class_names[i]
            print(f"  {class_name}: {mp:.4f}")
    else:
        print("  Per-class metrics unavailable (scalar only); check val() output above for details.")

    # Step 5: Test on Sample Images (Optional: Quick End-to-End Check)
    # Auto-find a test/valid image if available
    test_image_candidates = []
    for split in ['test/images', 'valid/images', 'train/images']:
        split_path = os.path.join(dataset_root, split)
        if os.path.exists(split_path):
            images = [f for f in os.listdir(split_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            if images:
                test_image_candidates.append(os.path.join(split_path, images[0]))
    if test_image_candidates:
        test_image_path = test_image_candidates[0]
        print(f"\nRunning inference on sample: {test_image_path}")
        results_test = model(test_image_path, save=True, conf=0.25)  # Save annotated image to runs/detect/
        print("Detections:")
        for r in results_test:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = model.names[cls]
                    print(f"  - {class_name} (ID: {cls}) conf: {conf:.2f}")
            else:
                print("  No detections found (try lowering conf=0.1).")
    else:
        print("\nNo sample images found in splits. Add a path to test_image_path manually.")

    # Step 6: Export Model for Mobile (ONNX for ONNX Runtime; TFLite optional for Android)
    success = model.export(format='onnx', dynamic=False, simplify=True, imgsz=640)
    print(f"\nModel exported: {success}")
    # Optional: Export to TensorFlow Lite for direct Android integration
    # tflite_path = model.export(format='tflite', imgsz=640, int8=True)  # Quantized for speed
    # print(f"TFLite exported: {tflite_path}")

    # Next Steps:
    # - Monitor training in TensorBoard: tensorboard --logdir grocery_yolo_runs
    # - If mixed segments warning persists: Re-export dataset from Roboflow as "Object Detection" only.
    # - For actual project: Use grocery classes, then retrain.
    # - Proceed to Phase 2: Android integration with ONNX Runtime.
    # - If errors: Reduce batch=8, or check dataset for corrupt labels.
    # - Tune inference: Lower conf=0.1 in predict() for more detections.

    print("YOLO Training Complete! Check 'grocery_yolo_runs/roboflow_test_v1/' for weights, plots, and logs.")

if __name__ == '__main__':
    freeze_support()  # For Windows multiprocessing compatibility
    main()