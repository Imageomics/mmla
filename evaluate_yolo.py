#!/usr/bin/env python3
"""
Evaluate YOLO model performance on downsampled frames with metrics calculation.
This script processes frames with ground truth annotations and calculates:
- Precision, Recall, F1-score
- mAP50
- Inference times
"""

import os
import time
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
import pandas as pd
import argparse
from collections import defaultdict


def calculate_metrics_per_frame(predictions, ground_truth, iou_threshold=0.5):
    """
    Calculate precision, recall, F1 for a single frame
    
    Args:
        predictions: List of predicted boxes [class_id, x, y, w, h, conf]
        ground_truth: List of ground truth boxes [class_id, x, y, w, h]
        iou_threshold: IoU threshold for matching
        
    Returns:
        Dict with precision, recall, F1 score
    """
    # No predictions or ground truth
    if not predictions and not ground_truth:
        return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
    elif not predictions:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    elif not ground_truth:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    # Track matches
    tp = 0
    fp = 0
    matched_gt = set()
    
    # For each prediction, find best matching ground truth
    for pred in predictions:
        pred_class = pred[0]
        pred_box = pred[1:5]
        
        best_iou = 0.0
        best_gt_idx = -1
        
        # Check against all ground truths
        for gt_idx, gt in enumerate(ground_truth):
            gt_class = gt[0]
            gt_box = gt[1:5]
            
            # Skip if class doesn't match or already matched
            if gt_class != pred_class or gt_idx in matched_gt:
                continue
            
            # Calculate IoU
            iou = calculate_iou(pred_box, gt_box)
            
            # Update best match
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_gt_idx = gt_idx
        
        # Count true positives and false positives
        if best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1
    
    # Calculate false negatives
    fn = len(ground_truth) - len(matched_gt)
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {'precision': precision, 'recall': recall, 'f1': f1}


def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes in [x, y, w, h] format
    
    Args:
        box1: First box [x, y, w, h] (normalized coordinates)
        box2: Second box [x, y, w, h] (normalized coordinates)
        
    Returns:
        IoU score
    """
    # Convert to [x1, y1, x2, y2] format
    box1_x1 = box1[0] - box1[2]/2
    box1_y1 = box1[1] - box1[3]/2
    box1_x2 = box1[0] + box1[2]/2
    box1_y2 = box1[1] + box1[3]/2
    
    box2_x1 = box2[0] - box2[2]/2
    box2_y1 = box2[1] - box2[3]/2
    box2_x2 = box2[0] + box2[2]/2
    box2_y2 = box2[1] + box2[3]/2
    
    # Calculate intersection area
    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union area
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union = box1_area + box2_area - intersection
    
    # Calculate IoU
    iou = intersection / union if union > 0 else 0.0
    
    return iou


def calculate_ap(precisions, recalls):
    """
    Calculate Average Precision using the 11-point interpolation
    
    Args:
        precisions: List of precision values
        recalls: List of recall values
        
    Returns:
        AP value
    """
    # Sort by recall
    combined = sorted(zip(recalls, precisions), key=lambda x: x[0])
    recalls = [r for r, _ in combined]
    precisions = [p for _, p in combined]
    
    # Add sentinel values
    recalls = [0.0] + recalls + [1.0]
    precisions = [0.0] + precisions + [0.0]
    
    # Make precision monotonically decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    # Calculate AP using 11-point interpolation
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        for i in range(len(recalls) - 1):
            if recalls[i] <= t <= recalls[i + 1]:
                ap += precisions[i]
                break
    
    return ap / 11.0


def evaluate_yolov8(
    model_path,
    image_dir,
    annotation_dir,
    output_dir=None,
    conf_threshold=0.25,
    iou_threshold=0.5,
    device="auto"
):
    """
    Evaluate YOLOv8 model on images with annotations
    
    Args:
        model_path: Path to YOLOv8 model (.pt file)
        image_dir: Directory with images
        annotation_dir: Directory with annotation files
        output_dir: Directory to save results (optional)
        conf_threshold: Confidence threshold for predictions
        iou_threshold: IoU threshold for matching
        device: Device to run model on ("cpu", "cuda", or "auto")
        
    Returns:
        Dict with evaluation results
    """
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load YOLOv8 model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # Get list of image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(list(Path(image_dir).glob(f"*{ext}")))
    
    print(f"Found {len(image_files)} images")
    
    # Initialize results storage
    results = {
        'per_frame': [],
        'overall': {},
        'class_metrics': defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    }
    
    # Overall metrics calculation variables # get these for Juan :) 
    all_predictions = []
    all_confidences = []
    all_tp_fp = []  # 1 for TP, 0 for FP
    all_gt_count = 0
    
    # Process each image
    for img_path in tqdm(image_files, desc="Evaluating frames"):
        img_name = img_path.stem
        
        # Load ground truth annotations
        ann_path = Path(annotation_dir) / f"{img_name}.txt"
        
        ground_truth = []
        if ann_path.exists():
            with open(ann_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x, y, w, h = map(float, parts[1:5])
                        ground_truth.append([class_id, x, y, w, h])
        
        # Update class-wise ground truth counts
        for gt in ground_truth:
            results['class_metrics'][gt[0]]['fn'] += 1  # Initialize as FN, will decrement if matched
        
        all_gt_count += len(ground_truth)
        
        # Run model inference and time it
        start_time = time.time()
        predictions = model(str(img_path), conf=conf_threshold)[0]
        inference_time = time.time() - start_time
        
        # Process predictions
        pred_boxes = []
        for pred in predictions.boxes.data.cpu().numpy():
            x1, y1, x2, y2 = pred[:4]  # Predictions are in xyxy format
            conf = pred[4]
            class_id = int(pred[5])
            
            # Convert to xywh format (normalized)
            img_height, img_width = predictions.orig_shape
            x = ((x1 + x2) / 2) / img_width
            y = ((y1 + y2) / 2) / img_height
            w = (x2 - x1) / img_width
            h = (y2 - y1) / img_height
            
            pred_boxes.append([class_id, x, y, w, h, conf])
            
            # Store for AP calculation
            all_predictions.append(class_id)
            all_confidences.append(conf)
        
        # Calculate metrics for this frame
        frame_metrics = calculate_metrics_per_frame(pred_boxes, ground_truth, iou_threshold)
        frame_metrics['inference_time'] = inference_time
        frame_metrics['frame'] = img_name
        frame_metrics['num_predictions'] = len(pred_boxes)
        frame_metrics['num_ground_truth'] = len(ground_truth)
        
        # Process TP/FP for AP calculation
        for pred in pred_boxes:
            pred_class = pred[0]
            pred_box = pred[1:5]
            
            # Check if prediction matches any ground truth
            matched = False
            for gt_idx, gt in enumerate(ground_truth):
                gt_class = gt[0]
                gt_box = gt[1:5]
                
                # Skip if class doesn't match
                if gt_class != pred_class:
                    continue
                
                # Calculate IoU
                iou = calculate_iou(pred_box, gt_box)
                
                # If IoU above threshold, count as TP
                if iou >= iou_threshold:
                    matched = True
                    # Decrement FN for this class (was initialized to full count)
                    results['class_metrics'][gt_class]['fn'] -= 1
                    # Increment TP for this class
                    results['class_metrics'][gt_class]['tp'] += 1
                    break
            
            # If not matched, count as FP
            if not matched:
                results['class_metrics'][pred_class]['fp'] += 1
                all_tp_fp.append(0)  # FP
            else:
                all_tp_fp.append(1)  # TP
        
        # Store per-frame results
        results['per_frame'].append(frame_metrics)
    
    # Calculate overall metrics
    if all_predictions:
        # Sort predictions by confidence (descending)
        combined = sorted(zip(all_confidences, all_tp_fp, all_predictions), 
                          key=lambda x: x[0], reverse=True)
        
        # Calculate cumulative TP and FP
        tp_cumsum = np.cumsum([x[1] for x in combined])
        fp_cumsum = np.cumsum([1 - x[1] for x in combined])
        
        # Calculate precision and recall at each threshold
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        recalls = tp_cumsum / all_gt_count if all_gt_count > 0 else np.zeros_like(tp_cumsum)
        
        # Calculate mAP
        overall_map = calculate_ap(precisions.tolist(), recalls.tolist())
        
        # Calculate overall precision, recall, F1
        tp_total = sum(x[1] for x in combined)
        fp_total = len(combined) - tp_total
        fn_total = all_gt_count - tp_total
        
        overall_precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        overall_recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
        
        results['overall'] = {
            'precision': overall_precision,
            'recall': overall_recall,
            'f1': overall_f1,
            'mAP50': overall_map,
            'avg_inference_time': np.mean([r['inference_time'] for r in results['per_frame']])
        }
    else:
        results['overall'] = {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'mAP50': 0.0,
            'avg_inference_time': 0.0
        }
    
    # Calculate per-class metrics
    for class_id, metrics in results['class_metrics'].items():
        tp = metrics['tp']
        fp = metrics['fp']
        fn = metrics['fn']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        results['class_metrics'][class_id] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    # Convert class metrics defaultdict to regular dict for JSON serialization
    results['class_metrics'] = dict(results['class_metrics'])
    
    # Save results if output directory is specified
    if output_dir:
        # Save overall metrics
        with open(os.path.join(output_dir, 'overall_metrics.json'), 'w') as f:
            json.dump(results['overall'], f, indent=4)
        
        # Save per-class metrics
        with open(os.path.join(output_dir, 'class_metrics.json'), 'w') as f:
            json.dump(results['class_metrics'], f, indent=4)
        
        # Save per-frame metrics as CSV
        per_frame_df = pd.DataFrame(results['per_frame'])
        per_frame_df.to_csv(os.path.join(output_dir, 'per_frame_metrics.csv'), index=False)
        
        print(f"Results saved to {output_dir}")
    
    # Print overall metrics
    print("\nOverall Metrics:")
    print(f"Precision: {results['overall']['precision']:.4f}")
    print(f"Recall: {results['overall']['recall']:.4f}")
    print(f"F1 Score: {results['overall']['f1']:.4f}")
    print(f"mAP@0.5: {results['overall']['mAP50']:.4f}")
    print(f"Average Inference Time: {results['overall']['avg_inference_time']*1000:.2f} ms")
    
    # Print per-class metrics
    print("\nPer-Class Metrics:")
    for class_id, metrics in results['class_metrics'].items():
        print(f"Class {class_id}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        print(f"  TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 on downsampled frames")
    parser.add_argument("--model", required=True, help="Path to YOLOv8 model (.pt file)")
    parser.add_argument("--images", required=True, help="Directory with images")
    parser.add_argument("--annotations", required=True, help="Directory with annotations")
    parser.add_argument("--output", help="Directory to save results")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold")
    parser.add_argument("--device", default="auto", help="Device to run on (cpu, cuda, auto)")
    
    args = parser.parse_args()
    
    evaluate_yolov8(
        args.model,
        args.images,
        args.annotations,
        args.output,
        args.conf,
        args.iou,
        args.device
    )


if __name__ == "__main__":
    main()