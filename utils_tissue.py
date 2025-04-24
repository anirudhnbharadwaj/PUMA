import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from matplotlib.patches import Patch

sns.set_theme(style="darkgrid")
plt.rcParams['image.cmap'] = 'rocket'

def compute_metrics(preds, gts, num_classes):
    preds = torch.argmax(preds, dim=1).cpu().numpy()  # Convert logits to class indices
    gts = gts.cpu().numpy()
    
    # Micro Dice per class
    micro_dice_per_class = []
    for class_id in range(num_classes):
        pred_class = (preds == class_id).astype(np.float32)
        gt_class = (gts == class_id).astype(np.float32)
        
        intersection = (pred_class * gt_class).sum()
        union = pred_class.sum() + gt_class.sum()
        dice = 2 * intersection / (union + 1e-6) if union > 0 else 1.0
        micro_dice_per_class.append(dice)
    
    # Macro Dice (average of per-class Dice)
    macro_dice = np.mean(micro_dice_per_class)
    
    logging.info(f"Metrics - Macro Dice: {macro_dice:.4f}")
    logging.info(f"Micro Dice per class: {micro_dice_per_class}")
    return macro_dice, micro_dice_per_class

def save_sample_image(images, tissue_gt, tissue_pred=None, fold=None, epoch=None, output_dir=None, idx=0, suffix="", pretrain=False, image_name="Unknown"):
    class_colors = {
        'White Background': sns.color_palette("rocket", n_colors=40)[0],   # Light grayish
        'Stroma': sns.color_palette("rocket", n_colors=40)[10],             # Orange-like
        'Blood Vessel': sns.color_palette("rocket", n_colors=40)[15],       # Slightly darker
        'Tumor': sns.color_palette("rocket", n_colors=40)[20],              # Reddish
        'Epidermis': sns.color_palette("rocket", n_colors=40)[25],          # Darker reddish
        'Necrosis': sns.color_palette("rocket", n_colors=40)[30]            # Dark blueish
    }
    class_hex = {
        'White Background': '#F5E8E5',
        'Stroma': '#E57C6F',
        'Blood Vessel': '#D95A5C',
        'Tumor': '#A8244F',
        'Epidermis': '#8B1E3F',
        'Necrosis': '#4F2A44'
    }
    class_rgb = {
        'White Background': np.array([245, 232, 229]),
        'Stroma': np.array([229, 124, 111]),
        'Blood Vessel': np.array([217, 90, 92]),
        'Tumor': np.array([168, 36, 79]),
        'Epidermis': np.array([139, 30, 63]),
        'Necrosis': np.array([79, 42, 68])
    }
    
    # Set up figure: 2 panels for pretrain (Image, GT), 3 panels for post-train (Image, GT, Pred)
    fig, axes = plt.subplots(1, 2 if pretrain else 3, figsize=(10 if pretrain else 15, 5))
    fig.suptitle(f"{image_name}", fontsize=16, fontweight='bold', fontfamily='DejaVu Sans')
    
    # Process image
    img = images[idx].permute(1, 2, 0).cpu().numpy() * 255.0  # Already in [0, 1], scale back to [0, 255]
    img = img.astype(np.uint8)
    
    # Original Image
    axes[0].imshow(img)
    axes[0].set_title("Original Image", fontsize=12, fontweight='bold', fontfamily='DejaVu Sans')
    axes[0].axis('off')
    
    # Ground Truth
    tissue_gt_rgb = np.zeros((tissue_gt.shape[1], tissue_gt.shape[2], 3), dtype=np.uint8)
    tissue_gt_rgb[tissue_gt[idx] == 0] = class_rgb['White Background']
    tissue_gt_rgb[tissue_gt[idx] == 1] = class_rgb['Stroma']
    tissue_gt_rgb[tissue_gt[idx] == 2] = class_rgb['Blood Vessel']
    tissue_gt_rgb[tissue_gt[idx] == 3] = class_rgb['Tumor']
    tissue_gt_rgb[tissue_gt[idx] == 4] = class_rgb['Epidermis']
    tissue_gt_rgb[tissue_gt[idx] == 5] = class_rgb['Necrosis']
    
    axes[1].imshow(tissue_gt_rgb)
    axes[1].set_title("Ground Truth", fontsize=12, fontweight='bold', fontfamily='DejaVu Sans')
    axes[1].axis('off')
    
    # Predictions (only for post-train)
    if not pretrain and tissue_pred is not None:
        tissue_pred = torch.argmax(tissue_pred, dim=1).cpu().numpy()
        tissue_pred_rgb = np.zeros((tissue_pred.shape[1], tissue_pred.shape[2], 3), dtype=np.uint8)
        tissue_pred_rgb[tissue_pred[idx] == 0] = class_rgb['White Background']
        tissue_pred_rgb[tissue_pred[idx] == 1] = class_rgb['Stroma']
        tissue_pred_rgb[tissue_pred[idx] == 2] = class_rgb['Blood Vessel']
        tissue_pred_rgb[tissue_pred[idx] == 3] = class_rgb['Tumor']
        tissue_pred_rgb[tissue_pred[idx] == 4] = class_rgb['Epidermis']
        tissue_pred_rgb[tissue_pred[idx] == 5] = class_rgb['Necrosis']
        
        axes[2].imshow(tissue_pred_rgb)
        axes[2].set_title("Prediction", fontsize=12, fontweight='bold', fontfamily='DejaVu Sans')
        axes[2].axis('off')
    elif not pretrain and tissue_pred is None:
        logging.warning("No prediction provided for post-train visualization")
    
    # Add legend
    legend_elements = [
        Patch(facecolor=class_hex['White Background'], edgecolor='black', label='White Background'),
        Patch(facecolor=class_hex['Stroma'], edgecolor='black', label='Stroma'),
        Patch(facecolor=class_hex['Blood Vessel'], edgecolor='black', label='Blood Vessel'),
        Patch(facecolor=class_hex['Tumor'], edgecolor='black', label='Tumor'),
        Patch(facecolor=class_hex['Epidermis'], edgecolor='black', label='Epidermis'),
        Patch(facecolor=class_hex['Necrosis'], edgecolor='black', label='Necrosis')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=6, fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.05))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.15)
    
    plt.savefig(os.path.join(output_dir, f"fold_{fold}_epoch_{epoch}_{suffix}.png"), bbox_inches='tight')
    plt.close()

def plot_metrics_and_loss(metrics, fold, output_dir, classes, train_losses=None, val_losses=None):
    if fold is not None:
        # Box plot for fold metrics
        fig, ax = plt.subplots(figsize=(10, 6))
        data = [
            [m['macro_dice'] for m in metrics],
            [m['micro_dice'][0] for m in metrics],  # White Background
            [m['micro_dice'][1] for m in metrics],  # Stroma
            [m['micro_dice'][2] for m in metrics],  # Blood Vessel
            [m['micro_dice'][3] for m in metrics],  # Tumor
            [m['micro_dice'][4] for m in metrics],  # Epidermis
            [m['micro_dice'][5] for m in metrics]   # Necrosis
        ]
        sns.boxplot(data=data, ax=ax, palette='rocket')
        ax.set_xticklabels(['Macro Dice', 'White Bg', 'Stroma', 'Blood Vessel', 'Tumor', 'Epidermis', 'Necrosis'], rotation=45)
        ax.set_title(f"Performance Metrics for Fold {fold+1}", fontsize=14, fontweight='bold', fontfamily='DejaVu Sans')
        ax.set_ylabel("Dice Score", fontsize=12)
        plt.tight_layout()
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
        ax2.set_ylim(bottom=0)
        ax2.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "loss_curves.png"), bbox_inches='tight')
        plt.close()