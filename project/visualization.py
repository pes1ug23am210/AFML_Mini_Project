# visualization.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def create_detection_visualization(original_image, classified_objects, output_path=None):
    """Create visualization of detection results with OCR information"""
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.imshow(original_image)
    ax.set_title('YOLO Object Detection with Classification & OCR Results', fontsize=16, fontweight='bold')
    ax.axis('off')
    
    # Draw bounding boxes on original image
    for obj in classified_objects:
        x1, y1, x2, y2 = obj['bbox']
        
        # Always blue bounding box
        color = 'blue'
        
        # Draw bounding box
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=3, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        # Label with predicted class name only
        label = obj['final_class_name']
        
        ax.text(x1, y1-10, label, fontsize=8, color=color, weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    if output_path:
        plt.savefig(f"{output_path}_detection_results.png", dpi=300, bbox_inches='tight')
        print(f"  📄 Saved: {Path(output_path).name}_detection_results.png")
    plt.show()
    plt.close()

def create_object_grid_visualization(classified_objects, output_path=None):
    """Create grid visualization of classified objects with OCR info"""
    num_objects = len(classified_objects)
    if num_objects == 0:
        return
    
    # Calculate grid size
    cols = min(3, num_objects)
    rows = (num_objects + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    fig.suptitle('Individual Object Classification & OCR Results', fontsize=16, fontweight='bold')
    
    # Handle different subplot configurations
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes
    elif cols == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    for idx, obj in enumerate(classified_objects):
        if idx < len(axes):
            # Display cropped object
            axes[idx].imshow(obj['cropped_image'])
            
            # Title with predicted class name only
            color = "blue"
            
            title = f"Object {idx+1}: {obj['final_class_name']}"
            
            axes[idx].set_title(title, fontsize=10, weight='bold', color=color)
            axes[idx].axis('off')
    
    # Hide empty subplots
    for idx in range(num_objects, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    if output_path:
        plt.savefig(f"{output_path}_classified_objects.png", dpi=300, bbox_inches='tight')
        print(f"  📄 Saved: {Path(output_path).name}_classified_objects.png")
    plt.show()
    plt.close()