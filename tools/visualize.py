"""
Visualization Tool for SAR Ship Detection - 6 Classes
可视化6类别模型预测结果
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import cv2
import numpy as np
from tqdm import tqdm

from config.sar_ship_config import cfg
from models.detectors.denodet_enhanced import EnhancedDenoDet
from data.datasets import SARShipDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize 6-Class SAR Ship Detection')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', default=None, help='Data directory')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples')
    parser.add_argument('--output_dir', default='visualizations', help='Output directory')
    parser.add_argument('--conf_threshold', type=float, default=0.3, help='Confidence threshold')
    parser.add_argument('--show_gt', action='store_true', help='Show ground truth')
    parser.add_argument('--save_crops', action='store_true', help='Save detection crops')
    return parser.parse_args()


def load_model(checkpoint_path, device):
    """Load model from checkpoint"""
    print(f"📥 Loading model from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = EnhancedDenoDet(use_attention=True, use_frequency=True).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    if 'epoch' in checkpoint:
        print(f"   Epoch: {checkpoint['epoch']}")
    if 'best_loss' in checkpoint:
        print(f"   Best loss: {checkpoint['best_loss']:.4f}")
    
    print("✅ Model loaded successfully")
    return model


def draw_bbox(img, bbox, label, score, color, thickness=2):
    """Draw bounding box with class label"""
    x1, y1, x2, y2 = map(int, bbox)
    
    # Draw rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    
    # Get class name
    class_name = cfg.CLASSES[label]
    
    # Draw label background
    text = f"{class_name}: {score:.2f}"
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(img, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), color, -1)
    
    # Draw text
    cv2.putText(img, text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, (255, 255, 255), 2)
    
    return img


def visualize_prediction(img, predictions, gt_boxes=None, gt_labels=None, 
                        conf_threshold=0.3, show_gt=False):
    """Visualize predictions with 6 classes"""
    # Convert to BGR
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img = img.copy()
    
    H, W = img.shape[:2]
    
    # Draw ground truth (white boxes)
    if show_gt and gt_boxes is not None:
        for bbox, label in zip(gt_boxes, gt_labels):
            # Convert YOLO to absolute
            x_center, y_center, w, h = bbox
            x1 = int((x_center - w/2) * W)
            y1 = int((y_center - h/2) * H)
            x2 = int((x_center + w/2) * W)
            y2 = int((y_center + h/2) * H)
            
            class_name = cfg.CLASSES[int(label)]
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)  # White for GT
            cv2.putText(img, f"GT:{class_name}", (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Draw predictions (colored by class)
    pred_bboxes = predictions['bboxes'].cpu().numpy()
    pred_scores = predictions['scores'].cpu().numpy()
    pred_labels = predictions['labels'].cpu().numpy()
    
    for bbox, score, label in zip(pred_bboxes, pred_scores, pred_labels):
        if score >= conf_threshold:
            # Get class-specific color
            class_name = cfg.CLASSES[int(label)]
            color = cfg.CLASS_COLORS.get(class_name, (0, 255, 0))
            img = draw_bbox(img, bbox, int(label), float(score), color, thickness=2)
    
    # Add legend
    legend_y = 30
    cv2.putText(img, f"Detections: {len(pred_scores[pred_scores >= conf_threshold])}", 
               (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add class legend
    legend_y = H - 20 * len(cfg.CLASSES) - 10
    for i, class_name in enumerate(cfg.CLASSES):
        color = cfg.CLASS_COLORS[class_name]
        y_pos = legend_y + i * 20
        cv2.rectangle(img, (10, y_pos-10), (30, y_pos+5), color, -1)
        cv2.putText(img, class_name, (35, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img


def save_detection_crops(img, predictions, output_dir, img_name, conf_threshold=0.3):
    """Save individual detection crops"""
    crop_dir = output_dir / 'crops' / img_name
    crop_dir.mkdir(parents=True, exist_ok=True)
    
    pred_bboxes = predictions['bboxes'].cpu().numpy()
    pred_scores = predictions['scores'].cpu().numpy()
    pred_labels = predictions['labels'].cpu().numpy()
    
    count = 0
    for i, (bbox, score, label) in enumerate(zip(pred_bboxes, pred_scores, pred_labels)):
        if score >= conf_threshold:
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
            
            crop = img[y1:y2, x1:x2]
            if crop.size > 0:
                class_name = cfg.CLASSES[int(label)]
                crop_path = crop_dir / f"{class_name}_{i}_score{score:.3f}.jpg"
                cv2.imwrite(str(crop_path), crop)
                count += 1
    
    return count


def main():
    args = parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Override threshold
    orig_conf = cfg.CONF_THRESHOLD
    cfg.CONF_THRESHOLD = args.conf_threshold
    
    print("="*80)
    print("SAR Ship Detection - 6 Classes Visualization")
    print("="*80)
    print(f"Classes: {', '.join(cfg.CLASSES)}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {output_dir}")
    print(f"Confidence threshold: {args.conf_threshold}")
    print("="*80)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.checkpoint, device)
    
    # Load dataset
    print("\n📦 Loading dataset...")
    if args.data_dir:
        img_dir = Path(args.data_dir) / 'images'
        label_dir = Path(args.data_dir) / 'labels'
    else:
        img_dir = cfg.VAL_IMG_DIR
        label_dir = cfg.VAL_LABEL_DIR
    
    dataset = SARShipDataset(
        img_dir=img_dir,
        label_dir=label_dir,
        img_size=cfg.IMG_SIZE,
        augment=False
    )
    
    print(f"✅ Loaded {len(dataset)} images")
    
    # Statistics
    num_samples = min(args.num_samples, len(dataset))
    stats = {
        'total_detections': 0,
        'class_counts': {cls: 0 for cls in cfg.CLASSES},
        'total_gt': 0
    }
    
    print(f"\n📊 Processing {num_samples} samples...")
    
    for i in tqdm(range(num_samples), desc="Visualizing"):
        data = dataset[i]
        
        # Predict
        with torch.no_grad():
            img_tensor = data['img'].unsqueeze(0).to(device)
            inputs = {'img': img_tensor}
            results = model(inputs, mode='test')[0]
        
        # Load image
        img = cv2.imread(str(data['img_path']), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, cfg.IMG_SIZE)
        
        # Get GT
        gt_boxes = data['gt_bboxes'].numpy() if args.show_gt else None
        gt_labels = data['gt_labels'].numpy() if args.show_gt else None
        
        # Visualize
        vis_img = visualize_prediction(
            img, results, gt_boxes, gt_labels,
            conf_threshold=args.conf_threshold,
            show_gt=args.show_gt
        )
        
        # Save
        img_name = Path(data['img_path']).stem
        output_path = output_dir / f"pred_{i:04d}_{img_name}.jpg"
        cv2.imwrite(str(output_path), vis_img)
        
        # Save crops
        if args.save_crops:
            save_detection_crops(img, results, output_dir, img_name, args.conf_threshold)
        
        # Update stats
        pred_labels = results['labels'].cpu().numpy()
        pred_scores = results['scores'].cpu().numpy()
        valid_mask = pred_scores >= args.conf_threshold
        
        stats['total_detections'] += valid_mask.sum()
        for label in pred_labels[valid_mask]:
            class_name = cfg.CLASSES[int(label)]
            stats['class_counts'][class_name] += 1
        
        if gt_labels is not None:
            stats['total_gt'] += len(gt_labels)
    
    # Print statistics
    print("\n" + "="*80)
    print("📊 Detection Statistics")
    print("="*80)
    print(f"Images processed: {num_samples}")
    print(f"Total detections: {stats['total_detections']}")
    print(f"Avg detections/image: {stats['total_detections']/num_samples:.2f}")
    print(f"\nPer-class detections:")
    for class_name, count in stats['class_counts'].items():
        pct = count / stats['total_detections'] * 100 if stats['total_detections'] > 0 else 0
        print(f"  {class_name:<12} {count:>4} ({pct:>5.1f}%)")
    if stats['total_gt'] > 0:
        print(f"\nTotal ground truth: {stats['total_gt']}")
        print(f"Avg GT/image: {stats['total_gt']/num_samples:.2f}")
    print("="*80)
    
    print(f"\n✅ Visualizations saved to: {output_dir}")
    if args.save_crops:
        print(f"✅ Detection crops saved to: {output_dir}/crops/")
    
    # Restore config
    cfg.CONF_THRESHOLD = orig_conf


if __name__ == '__main__':
    main()