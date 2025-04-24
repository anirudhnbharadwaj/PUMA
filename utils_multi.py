import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import cv2
import json
from shapely.geometry import Polygon
import os
from matplotlib.patches import Patch

sns.set_theme(style="darkgrid")
plt.rcParams['image.cmap'] = 'rocket'

def compute_iou(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return intersection / union if union > 0 else 0.0

def compute_metrics(preds, gts, num_classes, task='nuclei', class_gts=None):
    if task != 'nuclei':
        raise ValueError("Only nuclei task is supported")
    
    batch_pq = []
    batch_aji = []
    batch_f1 = []
    
    for b in range(preds.shape[0]):
        # Extract predicted instances
        pred_instances = np.unique(preds[b])[1:]  # Exclude background
        pred_classes = []
        for inst_id in pred_instances:
            class_id = inst_id // 1000
            if class_id >= num_classes:
                logging.warning(f"Invalid class_id {class_id} in pred instance {inst_id}")
                class_id = 0
            pred_classes.append(class_id)
        
        # Extract GT instances from instance mask
        gt_instances = np.unique(gts[b])[1:]  # Exclude background
        gt_classes = []
        for inst_id in gt_instances:
            if class_gts is None:
                logging.warning(f"Batch {b}: No class GT provided, assuming class 1 for instance {inst_id}")
                gt_classes.append(1)
                continue
            instance_mask = (gts[b] == inst_id)
            class_ids = class_gts[b][instance_mask]
            if class_ids.size == 0:
                logging.debug(f"Empty instance {inst_id} in GT batch {b}")
                gt_classes.append(0)
                continue
            class_id = np.bincount(class_ids).argmax()
            gt_classes.append(class_id)
        
        logging.debug(f"Batch {b}: {len(pred_instances)} pred instances, {len(gt_instances)} GT instances")
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(pred_instances), len(gt_instances)))
        for i, pred_id in enumerate(pred_instances):
            pred_mask = (preds[b] == pred_id)
            for j, gt_id in enumerate(gt_instances):
                gt_mask = (gts[b] == gt_id)
                iou_matrix[i, j] = compute_iou(pred_mask, gt_mask)
        
        # PQ: Match instances with IoU > 0.5 and same class
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        tp = 0
        iou_sum = 0.0
        matches = []
        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] > 0.5 and pred_classes[r] == gt_classes[c]:
                tp += 1
                iou_sum += iou_matrix[r, c]
                matches.append((r, c))
        
        fp = len(pred_instances) - tp
        fn = len(gt_instances) - tp
        pq = iou_sum / (tp + 0.5 * fp + 0.5 * fn) if (tp + 0.5 * fp + 0.5 * fn) > 0 else 0.0
        batch_pq.append(pq)
        
        # AJI: Match GT to best pred with same class
        matched_pred = set()
        intersection_sum = 0.0
        union_sum = 0.0
        for j, gt_id in enumerate(gt_instances):
            gt_mask = (gts[b] == gt_id)
            if len(pred_instances) == 0:
                union_sum += gt_mask.sum()
                continue
            best_pred_idx = np.argmax(iou_matrix[:, j]) if iou_matrix.shape[0] > 0 else -1
            best_iou = iou_matrix[best_pred_idx, j] if best_pred_idx >= 0 else 0.0
            if best_iou > 0 and best_pred_idx >= 0 and pred_classes[best_pred_idx] == gt_classes[j]:
                pred_mask = (preds[b] == pred_instances[best_pred_idx])
                intersection = np.logical_and(pred_mask, gt_mask).sum()
                union = np.logical_or(pred_mask, gt_mask).sum()
                intersection_sum += intersection
                union_sum += union
                matched_pred.add(best_pred_idx)
        
        for i, pred_id in enumerate(pred_instances):
            if i not in matched_pred:
                pred_mask = (preds[b] == pred_id)
                union_sum += pred_mask.sum()
        
        aji = intersection_sum / union_sum if union_sum > 0 else 0.0
        batch_aji.append(aji)
        
        # Macro F1: Instance-based
        if len(matches) == 0 and (len(pred_instances) == 0 or len(gt_instances) == 0):
            batch_f1.append(0.0)
            logging.debug(f"Batch {b}: No matches, setting F1=0")
            continue
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        batch_f1.append(f1)
    
    pq = np.mean(batch_pq) if batch_pq else 0.0
    aji = np.mean(batch_aji) if batch_aji else 0.0
    macro_f1 = np.mean(batch_f1) if batch_f1 else 0.0
    
    logging.info(f"Metrics - PQ: {pq:.4f}, AJI: {aji:.4f}, Macro F1: {macro_f1:.4f}")
    return pq, aji, macro_f1

def save_sample_image(images, nuclei_instance_gt, nuclei_pred, fold, epoch, output_dir, idx=0, suffix="", pretrain=False, image_name="Unknown", nuclei_class_gt=None):
    # Sample colors from rocket palette
    rocket_colors = sns.color_palette("rocket", n_colors=40)
    class_colors = {
        'Background': rocket_colors[0],   # Light grayish (index 0)
        'Tumor': rocket_colors[20],       # Reddish (index 20)
        'TILs': rocket_colors[10],        # Orange-like (index 10)
        'Others': rocket_colors[30]       # Darker blueish (index 30)
    }
    class_hex = {
        'Background': '#F5E8E5',  # Approx match to rocket[0]
        'Tumor': '#A8244F',       # Approx match to rocket[20]
        'TILs': '#E57C6F',        # Approx match to rocket[10]
        'Others': '#4F2A44'       # Approx match to rocket[30]
    }
    class_rgb = {
        'Background': np.array([245, 232, 229]),  # RGB for #F5E8E5
        'Tumor': np.array([168, 36, 79]),         # RGB for #A8244F
        'TILs': np.array([229, 124, 111]),        # RGB for #E57C6F
        'Others': np.array([79, 42, 68])          # RGB for #4F2A44
    }
    
    # Set up figure
    fig, axes = plt.subplots(1, 3 if not pretrain else 2, figsize=(15 if not pretrain else 10, 5))
    fig.suptitle(f"{image_name}", fontsize=16, fontweight='bold', fontfamily='DejaVu Sans')
    
    # Process image
    img = images[idx].permute(1, 2, 0).cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img * std + mean)
    img = (img * 255).astype(np.uint8)
    
    # Original Image
    axes[0].imshow(img)
    axes[0].set_title("Original Image", fontsize=12, fontweight='bold', fontfamily='DejaVu Sans')
    axes[0].axis('off')
    
    # Ground Truth
    nuclei_gt_rgb = np.zeros((nuclei_instance_gt.shape[1], nuclei_instance_gt.shape[2], 3), dtype=np.uint8)
    if not pretrain:
        if nuclei_class_gt is None:
            raise ValueError("nuclei_class_gt must be provided when pretrain=False")
        # Map instance IDs to class IDs
        instance_ids = np.unique(nuclei_instance_gt[idx])[1:]  # Exclude background
        logging.debug(f"Ground Truth instance IDs: {instance_ids}")
        for instance_id in instance_ids:
            instance_mask = (nuclei_instance_gt[idx] == instance_id)
            class_ids = nuclei_class_gt[idx][instance_mask]
            if class_ids.size == 0:
                logging.warning(f"Instance {instance_id} has no corresponding class pixels")
                continue
            class_id = np.bincount(class_ids.flatten()).argmax()  # Most common class ID
            logging.debug(f"Instance {instance_id} mapped to class {class_id}")
            nuclei_gt_rgb[nuclei_instance_gt[idx] == 0] = class_rgb['Background']
            if class_id == 1:
                nuclei_gt_rgb[instance_mask] = class_rgb['Tumor']
            elif class_id == 2:
                nuclei_gt_rgb[instance_mask] = class_rgb['TILs']
            elif class_id == 3:
                nuclei_gt_rgb[instance_mask] = class_rgb['Others']
            else:
                logging.warning(f"Invalid class ID {class_id} for instance {instance_id}")
    else:
        # Pretraining: nuclei_instance_gt is actually nuclei_class_gt with raw class IDs
        nuclei_gt_rgb[nuclei_instance_gt[idx] == 0] = class_rgb['Background']
        nuclei_gt_rgb[nuclei_instance_gt[idx] == 1] = class_rgb['Tumor']
        nuclei_gt_rgb[nuclei_instance_gt[idx] == 2] = class_rgb['TILs']
        nuclei_gt_rgb[nuclei_instance_gt[idx] == 3] = class_rgb['Others']
    
    axes[1].imshow(nuclei_gt_rgb)
    axes[1].set_title("Ground Truth", fontsize=12, fontweight='bold', fontfamily='DejaVu Sans')
    axes[1].axis('off')
    
    # Predictions (if not pretrain)
    if not pretrain:
        nuclei_pred_rgb = np.zeros((nuclei_pred.shape[1], nuclei_pred.shape[2], 3), dtype=np.uint8)
        nuclei_pred_rgb[nuclei_pred[idx] == 0] = class_rgb['Background']
        nuclei_pred_rgb[(nuclei_pred[idx] // 1000) == 1] = class_rgb['Tumor']
        nuclei_pred_rgb[(nuclei_pred[idx] // 1000) == 2] = class_rgb['TILs']
        nuclei_pred_rgb[(nuclei_pred[idx] // 1000) == 3] = class_rgb['Others']
        
        axes[2].imshow(nuclei_pred_rgb)
        axes[2].set_title("Instance Prediction", fontsize=12, fontweight='bold', fontfamily='DejaVu Sans')
        axes[2].axis('off')
    
    # Add legend with color patches
    legend_elements = [
        Patch(facecolor=class_hex['Background'], edgecolor='black', label='Background'),
        Patch(facecolor=class_hex['Tumor'], edgecolor='black', label='Tumor'),
        Patch(facecolor=class_hex['TILs'], edgecolor='black', label='TILs'),
        Patch(facecolor=class_hex['Others'], edgecolor='black', label='Others')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.05))
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.15)
    
    # Save figure
    plt.savefig(os.path.join(output_dir, f"fold_{fold}_epoch_{epoch}_{suffix}.png"), bbox_inches='tight')
    plt.close()

def plot_metrics_and_loss(metrics, fold, output_dir, classes, train_losses=None, val_losses=None):
    if fold is not None:
        # Box plot for fold metrics
        fig, ax = plt.subplots(figsize=(10, 6))
        data = [
            [m['pq'] for m in metrics],
            [m['aji'] for m in metrics],
            [m['macro_f1'] for m in metrics]
        ]
        sns.boxplot(data=data, ax=ax, palette='rocket')
        ax.set_xticklabels(['PQ', 'AJI', 'Macro F1'])
        ax.set_title(f"Performance Metrics for Fold {fold+1}", fontsize=14, fontweight='bold', fontfamily='DejaVu Sans')
        ax.set_ylabel("Score", fontsize=12)
        plt.savefig(os.path.join(output_dir, f"metrics_fold_{fold}.png"), bbox_inches='tight')
        plt.close()
    else:
        # Line plots for all folds
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        rocket_colors = sns.color_palette("rocket", n_colors=len(train_losses))
        for fold_idx, t_losses in enumerate(train_losses):
            ax1.plot(range(1, len(t_losses)+1), t_losses, label=f"Fold {fold_idx+1}", color=rocket_colors[fold_idx], linewidth=2)
        ax1.set_title("Training Loss Across Folds", fontsize=14, fontweight='bold', fontfamily='DejaVu Sans')
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Loss", fontsize=12)
        ax1.legend(fontsize=10)
        
        for fold_idx, v_losses in enumerate(val_losses):
            ax2.plot(range(1, len(v_losses)+1), v_losses, label=f"Fold {fold_idx+1}", color=rocket_colors[fold_idx], linewidth=2)
        ax2.set_title("Validation Loss Across Folds", fontsize=14, fontweight='bold', fontfamily='DejaVu Sans')
        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel("Loss", fontsize=12)
        ax2.set_ylim(bottom=0)  # Ensure non-negative loss
        ax2.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "loss_curves.png"), bbox_inches='tight')
        plt.close()

def save_predictions_to_geojson(predictions, image_name, output_dir, suffix):
    geojson_data = {
        "type": "FeatureCollection",
        "features": []
    }
    
    for batch_idx in range(predictions.shape[0]):
        pred_mask = predictions[batch_idx]
        instance_ids = np.unique(pred_mask)[1:]  # Exclude background
        for inst_id in instance_ids:
            class_id = inst_id // 1000
            if class_id == 0:
                continue
            mask = (pred_mask == inst_id).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                coords = [[int(x), int(y)] for x, y in contour.squeeze().tolist()]
                if len(coords) < 3:
                    continue
                polygon = Polygon(coords)
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [coords]
                    },
                    "properties": {
                        "classification": {
                            "name": {1: "Tumor", 2: "TILs", 3: "Others"}[class_id],
                            "class_id": int(class_id),
                            "instance_id": int(inst_id % 1000)
                        }
                    }
                }
                geojson_data["features"].append(feature)
    
    output_path = os.path.join(output_dir, f"{image_name}_{suffix}.geojson")
    with open(output_path, 'w') as f:
        json.dump(geojson_data, f)
    logging.info(f"Saved GeoJSON to {output_path}")