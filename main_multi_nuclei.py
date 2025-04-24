import os
import logging
import argparse
import time
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import random

from data_multi import PUMADatasetMulti
from nuclei_multi import NucleiModelMulti
from utils_multi import compute_metrics, save_sample_image, plot_metrics_and_loss, save_predictions_to_geojson

sns.set_theme(style="darkgrid")
plt.rcParams['image.cmap'] = 'rocket'

class Params:
    def __init__(self):
        self.root = "dataset/01_training_dataset_tif_ROIs"
        self.nuclei_root = "dataset/01_training_dataset_geojson_nuclei"
        self.output_dir = "puma_output/multi_nuclei"
        self.patch_size = 256
        self.stride = 128
        self.batch_size = 1
        self.num_epochs = 2
        self.num_folds = 5
        self.val_interval = 1
        self.lr = 0.02
        self.weight_decay = 1e-4
        self.warmup_epochs = 5
        self.num_workers = 1
        self.reduced_load = True
        self.reduced_size = 31
        self.batch_val_interval = 800

def setup_logging(debug=False):
    os.makedirs(Params().output_dir, exist_ok=True)
    level = logging.DEBUG if debug else logging.INFO
    # if level == logging.DEBUG:
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(Params().output_dir, 'multi_nuclei_debug.log')),
        ]
    )
    # else:
    #     logging.basicConfig(
    #         level=level,
    #         format='%(asctime)s [%(levelname)s] %(message)s',
    #         handlers=[
    #             logging.FileHandler(os.path.join(Params().output_dir, 'multi_nuclei.log')),
    #         ]
    #     )

def train_one_epoch(model, loader, optimizer, device, params, args, fold, epoch, inter, val, sample_indices, dataset):
    model.train()
    epoch_loss = 0
    nuclei_loss_sum = 0
    batch_count = 0
        
    progress_bar = tqdm(loader, desc=f"Fold {fold+1}, Epoch {epoch+1}: Loss=0.0000", leave=False, total=len(loader), colour="green")
    
    for batch_idx, batch in enumerate(progress_bar):
        images = batch['image'].to(device)
        nuclei_instance_gt = batch['nuclei_instance_mask'].to(device)
        nuclei_class_gt = batch['nuclei_class_mask'].to(device)
        
        if nuclei_instance_gt.max() == 0:
            logging.debug(f"Skipping batch {batch_idx} with empty nuclei mask")
            continue
        
        maskrcnn_targets = []
        for b in range(images.size(0)):
            nuclei_instance_gt_b = nuclei_instance_gt[b]
            nuclei_class_gt_b = nuclei_class_gt[b]
            boxes = []
            instance_ids = torch.unique(nuclei_instance_gt_b)
            instance_ids = instance_ids[instance_ids > 0]
            instance_masks = []
            labels = []
            
            for instance_id in instance_ids:
                instance_mask = (nuclei_instance_gt_b == instance_id).float()
                instance_array = instance_mask.cpu().numpy().astype(np.uint8)
                contours, hierarchy = cv2.findContours(instance_array, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                for contour_idx, contour in enumerate(contours):
                    if hierarchy[0][contour_idx][3] != -1:
                        continue
                    x, y, w, h = cv2.boundingRect(contour)
                    if w > 0 and h > 0 and (x + w) <= params.patch_size and (y + h) <= params.patch_size:
                        boxes.append([x, y, x + w, y + h])
                        instance_masks.append(instance_mask)
                        instance_pixels = nuclei_class_gt_b[nuclei_instance_gt_b == instance_id]
                        if instance_pixels.numel() > 0:
                            class_id = torch.mode(instance_pixels)[0].item()
                            labels.append(class_id)
                        else:
                            labels.append(0)
            
            boxes = torch.tensor(boxes, dtype=torch.float32).to(device) if boxes else torch.zeros((0, 4), dtype=torch.float32).to(device)
            labels = torch.tensor(labels, dtype=torch.int64).to(device) if labels else torch.zeros((0,), dtype=torch.int64).to(device)
            instance_masks = torch.stack(instance_masks, dim=0).to(device) if instance_masks else torch.zeros((0, params.patch_size, params.patch_size), dtype=torch.float32).to(device)
            maskrcnn_targets.append({
                'labels': labels,
                'masks': instance_masks,
                'boxes': boxes
            })
            logging.debug(f"Batch {batch_idx}, Image {b}: {len(boxes)} boxes, {len(labels)} labels, {instance_masks.shape[0]} masks")
        
        optimizer.zero_grad()
        nuclei_outputs, nuclei_instance_mask = model(images, maskrcnn_targets, output_dir=params.output_dir if args.debug else None, batch_idx=batch_idx)
        
        nuclei_loss = torch.tensor(0.0, device=device)
        maskrcnn_loss = torch.tensor(0.0, device=device)
        if isinstance(nuclei_outputs, dict):
            loss_components = {}
            for loss_name, loss_value in nuclei_outputs.items():
                if torch.isfinite(loss_value):
                    if loss_name == 'loss_mask':
                        loss_value = 1.0 * loss_value
                    elif loss_name == 'loss_objectness':
                        loss_value = 1.0 * loss_value
                    elif loss_name == 'loss_box_reg':
                        loss_value = 1.0 * loss_value
                    elif loss_name == 'loss_rpn_box_reg':
                        loss_value = 0.7 * loss_value
                    elif loss_name == 'loss_classifier':
                        loss_value = 1.0 * loss_value
                    loss_components[loss_name] = loss_value.item()
                    maskrcnn_loss += loss_value
            if batch_idx % 200 == 0:
                logging.info(f"Training Batch {batch_idx}: Mask R-CNN Losses: {loss_components}")
                logging.info(f"Training Batch {batch_idx}: Total Mask R-CNN Loss: {maskrcnn_loss.item():.4f}")
            nuclei_loss += maskrcnn_loss
        
        loss = nuclei_loss
        if not torch.isfinite(loss):
            logging.warning(f"NaN loss detected in batch {batch_idx}, skipping")
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        epoch_loss += loss.item()
        nuclei_loss_sum += nuclei_loss.item()
        batch_count += 1
        
        avg_nuclei_loss = nuclei_loss_sum / batch_count if batch_count > 0 else 0
        progress_bar.set_description(f"Fold {fold+1}, Epoch {epoch+1}: Loss={avg_nuclei_loss:.4f}")
    
        if (batch_idx + 1) % inter == 0:
            val_loss, pq, aji, macro_f1 = validate(model, val, device, params, args, fold, epoch, sample_indices, dataset)
            logging.info(f"Fold {fold+1}, Epoch {epoch+1}, Batch {batch_idx+1} - Val Loss: {val_loss:.4f}, PQ: {pq:.4f}, AJI: {aji:.4f}, Macro F1: {macro_f1:.4f}")
            model.train()
        
    avg_loss = epoch_loss / max(batch_count, 1)
    logging.info(f"Fold {fold+1}, Epoch {epoch+1} - Training Loss: {avg_loss:.4f}")
    return avg_loss

def validate(model, loader, device, params, args, fold, epoch, sample_indices, dataset):
    random_idx = random.randint(0, len(loader) - 1)
    model.eval()
    total_nuclei_loss = 0.0
    nuclei_metrics = []
    class_gts = []
    with torch.no_grad():
        progress_bar = tqdm(loader, desc=f"Fold {fold+1}, Epoch {epoch+1}: Validating", leave=False, colour="blue", total=len(loader))
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(device)
            nuclei_instance_gt = batch['nuclei_instance_mask'].to(device)
            nuclei_class_gt = batch['nuclei_class_mask'].to(device)
            
            logging.debug(f"Batch {batch_idx}: Class GT unique values: {np.unique(nuclei_class_gt.cpu().numpy())}")
            
            maskrcnn_targets = []
            for b in range(images.size(0)):
                nuclei_instance_gt_b = nuclei_instance_gt[b]
                nuclei_class_gt_b = nuclei_class_gt[b]
                boxes = []
                instance_ids = torch.unique(nuclei_instance_gt_b)
                instance_ids = instance_ids[instance_ids > 0]
                instance_masks = []
                labels = []
                
                for instance_id in instance_ids:
                    instance_mask = (nuclei_instance_gt_b == instance_id).float()
                    instance_array = instance_mask.cpu().numpy().astype(np.uint8)
                    contours, hierarchy = cv2.findContours(instance_array, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                    for contour_idx, contour in enumerate(contours):
                        if hierarchy[0][contour_idx][3] != -1:
                            continue
                        x, y, w, h = cv2.boundingRect(contour)
                        if w > 0 and h > 0 and (x + w) <= params.patch_size and (y + h) <= params.patch_size:
                            boxes.append([x, y, x + w, y + h])
                            instance_masks.append(instance_mask)
                            instance_pixels = nuclei_class_gt_b[nuclei_instance_gt_b == instance_id]
                            if instance_pixels.numel() > 0:
                                class_id = torch.mode(instance_pixels)[0].item()
                                labels.append(class_id)
                            else:
                                labels.append(0)
                
                boxes = torch.tensor(boxes, dtype=torch.float32).to(device) if boxes else torch.zeros((0, 4), dtype=torch.float32).to(device)
                labels = torch.tensor(labels, dtype=torch.int64).to(device) if labels else torch.zeros((0,), dtype=torch.int64).to(device)
                instance_masks = torch.stack(instance_masks, dim=0).to(device) if instance_masks else torch.zeros((0, params.patch_size, params.patch_size), dtype=torch.float32).to(device)
                maskrcnn_targets.append({
                    'labels': labels,
                    'masks': instance_masks,
                    'boxes': boxes
                })
                logging.debug(f"Val Batch {batch_idx}, Image {b}: {len(boxes)} boxes, {len(labels)} labels, {instance_masks.shape[0]} masks")
            
            model.train()
            nuclei_outputs, _ = model(images, maskrcnn_targets)
            model.eval()
            
            maskrcnn_loss = 0.0
            loss_components = {}
            for loss_name, loss_value in nuclei_outputs.items():
                if torch.isfinite(loss_value):
                    if loss_name == 'loss_mask':
                        loss_value = 1.5 * loss_value
                    elif loss_name == 'loss_objectness':
                        loss_value = 1.0 * loss_value
                    elif loss_name == 'loss_box_reg':
                        loss_value = 1.0 * loss_value
                    elif loss_name == 'loss_rpn_box_reg':
                        loss_value = 0.7 * loss_value
                    loss_components[loss_name] = loss_value.item()
                    if batch_idx % 200 == 0:
                        logging.info(f"Validation Batch {batch_idx}: Mask R-CNN Losses: {loss_components}")
                    maskrcnn_loss += loss_value
            
            total_nuclei_loss += maskrcnn_loss.item()
            
            _, nuclei_instance_pred = model(images, output_dir=params.output_dir if args.debug else None, batch_idx=batch_idx)
            logging.debug(f"Batch {batch_idx}: GT instances: {len(np.unique(nuclei_instance_gt.cpu().numpy()[0])[1:])}, Pred instances: {len(np.unique(nuclei_instance_pred[0])[1:])}")
            nuclei_metrics.append((nuclei_instance_pred, nuclei_instance_gt.cpu().numpy()))
            class_gts.append(nuclei_class_gt.cpu().numpy())
            
            if batch_idx == random_idx or batch_idx == 7:
                save_sample_image(
                    images,
                    nuclei_instance_gt.cpu(),
                    nuclei_instance_pred,
                    fold=fold,
                    epoch=epoch,
                    output_dir=params.output_dir,
                    idx=0,
                    suffix=f"val_batch_{batch_idx}",
                    pretrain=False,
                    image_name=f"val_batch_{batch_idx}",
                    nuclei_class_gt=nuclei_class_gt.cpu()  # Pass the class ground truth
                )
    
    avg_nuclei_loss = total_nuclei_loss / len(loader)
    
    all_preds = np.concatenate([pred for pred, gt in nuclei_metrics], axis=0)
    all_gts = np.concatenate([gt for pred, gt in nuclei_metrics], axis=0)
    all_class_gts = np.concatenate(class_gts, axis=0)
    pq, aji, macro_f1 = compute_metrics(all_preds, all_gts, num_classes=4, task='nuclei', class_gts=all_class_gts)
    
    logging.info(f"Fold {fold+1}, Epoch {epoch+1} - Validation Loss: {avg_nuclei_loss:.4f}, PQ: {pq:.4f}, AJI: {aji:.4f}, Macro F1: {macro_f1:.4f}")
    return avg_nuclei_loss, pq, aji, macro_f1

def main():
    parser = argparse.ArgumentParser(description="Train multi-class nuclei segmentation model")
    parser.add_argument('--root', default="dataset/01_training_dataset_tif_ROIs", help="Path to image directory")
    parser.add_argument('--nuclei_root', default="dataset/01_training_dataset_geojson_nuclei", help="Path to nuclei GeoJSON directory")
    parser.add_argument('--output_dir', default="puma_output/multi_nuclei", help="Output directory")
    parser.add_argument('--patch_size', type=int, default=256, help="Patch size")
    parser.add_argument('--stride', type=int, default=128, help="Stride for patch extraction")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size")
    parser.add_argument('--num_epochs', type=int, default=2, help="Number of epochs")
    parser.add_argument('--num_folds', type=int, default=5, help="Number of folds")
    parser.add_argument('--val_interval', type=int, default=1, help="Validation interval (epochs)")
    parser.add_argument('--lr', type=float, default=0.002, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="Weight decay")
    parser.add_argument('--debug', action='store_false', help="Enable debug logging and images")
    args = parser.parse_args()
    
    setup_logging(args.debug)
    
    params = Params()
    params.root = args.root
    params.nuclei_root = args.nuclei_root
    params.output_dir = args.output_dir
    params.patch_size = args.patch_size
    params.stride = args.stride
    params.batch_size = args.batch_size
    params.num_epochs = args.num_epochs
    params.num_folds = args.num_folds
    params.val_interval = args.val_interval
    params.lr = args.lr
    params.weight_decay = args.weight_decay
    
    os.makedirs(params.output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    start_time = time.time()
    logging.info("Loading dataset...")
    dataset = PUMADatasetMulti(
        params.root,
        params.nuclei_root,
        patch_size=params.patch_size,
        stride=params.stride,
        mode='train',
        maskrcnn_transforms=NucleiModelMulti().maskrcnn_transforms,
        reduced_load=params.reduced_load,
        reduced_size=params.reduced_size
    )
    logging.info(f"Dataset loaded in {time.time() - start_time:.2f} seconds")
    logging.info(f"Dataset size - 7x7 patches per image * number of images: {len(dataset)}")
    
    sample_indices = np.random.choice(len(dataset), 3, replace=False)
    for sample_idx in sample_indices:
        sample = dataset[sample_idx]
        images = sample['image'].unsqueeze(0)
        nuclei_gt = sample['nuclei_class_mask'].unsqueeze(0)
        save_sample_image(
            images,
            nuclei_gt,
            nuclei_gt,
            fold=0,
            epoch=0,
            output_dir=params.output_dir,
            idx=0,
            suffix=f"pretrain_sample_{sample_idx}",
            pretrain=True,
            image_name=sample['image_name'],
            nuclei_class_gt=None  # Not needed for pretrain
        )
    
    kf = KFold(n_splits=params.num_folds, shuffle=True, random_state=42)
    indices = list(range(len(dataset)))
    
    all_train_losses = []
    all_val_losses = []
    all_metrics = []
    
    logging.info("Starting training...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        logging.info(f"Starting Fold {fold + 1}/{params.num_folds}")
        
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=params.batch_size, shuffle=False, num_workers=params.num_workers, pin_memory=True)
        
        total_iterations = len(train_loader) * params.num_epochs
        dynamic_warmup_epochs = min(max(1, params.num_epochs // 2), params.num_epochs)
        warmup_iterations = len(train_loader) * dynamic_warmup_epochs
        start_factor = max(0.01, 0.1 / (len(train_loader) / 1000))
        cosine_iterations = max(1, total_iterations - warmup_iterations)
        t_max = max(1, cosine_iterations // len(train_loader))
        
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(NucleiModelMulti().to(device))
            model.freeze_backbone(freeze=False)
            optimizer = torch.optim.AdamW(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=start_factor, end_factor=1.0, total_iters=warmup_iterations)
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=params.lr * 0.01)
            scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_iterations])
        else:
            model = NucleiModelMulti().to(device)
            model.freeze_backbone(freeze=False)
            optimizer = torch.optim.AdamW(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=start_factor, end_factor=1.0, total_iters=warmup_iterations)
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=params.lr * 0.01)
            scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_iterations])
        
        logging.info(f"Fold {fold+1} Scheduler: Warmup {dynamic_warmup_epochs} epochs ({warmup_iterations} iterations, start_factor={start_factor:.3f}), "
                     f"Cosine T_max={t_max} epochs, eta_min={params.lr * 0.01:.6f}, val_interval={params.val_interval} epochs")
        
        best_val_loss = float('inf')
        fold_train_losses = []
        fold_val_losses = []
        fold_metrics = []
        
        for epoch in tqdm(range(params.num_epochs), desc=f"Fold {fold+1} Epochs", leave=True, total=params.num_epochs, colour="red"):
            logging.info(f"Fold {fold+1}, Epoch {epoch + 1}/{params.num_epochs}")
            train_loss = train_one_epoch(model, train_loader, optimizer, device, params, args, fold, epoch, params.batch_val_interval, val_loader, sample_indices, dataset)
            fold_train_losses.append(train_loss)
            
            if (epoch + 1) % params.val_interval == 0:
                logging.info(f"Validating at Fold {fold+1}, Epoch {epoch+1}")
                val_loss, pq, aji, macro_f1 = validate(model, val_loader, device, params, args, fold, epoch, sample_indices, dataset)
                logging.info(f"Fold {fold+1}, Epoch {epoch+1} - Val Loss: {val_loss:.4f}, PQ: {pq:.4f}, AJI: {aji:.4f}, Macro F1: {macro_f1:.4f}")
                fold_val_losses.append(val_loss)
                fold_metrics.append({'pq': pq, 'aji': aji, 'macro_f1': macro_f1})
            
            scheduler.step()
            logging.info(f"Fold {fold+1}, Epoch {epoch+1} - Learning rate: {scheduler.get_last_lr()}")
            
            if len(fold_val_losses) > 0 and fold_val_losses[-1] < best_val_loss:
                best_val_loss = fold_val_losses[-1]
                torch.save(model.state_dict(), os.path.join(params.output_dir, f"best_model_fold_{fold}.pth"))
                logging.info(f"Saved best model for Fold {fold+1} with validation loss: {best_val_loss:.4f}")
        
        final_val_loss, pq, aji, macro_f1 = validate(model, val_loader, device, params, args, fold, params.num_epochs, sample_indices, dataset)
        logging.info(f"Fold {fold+1} Final Validation - Loss: {final_val_loss:.4f}, PQ: {pq:.4f}, AJI: {aji:.4f}, Macro F1: {macro_f1:.4f}")
        fold_val_losses.append(final_val_loss)
        fold_metrics.append({'pq': pq, 'aji': aji, 'macro_f1': macro_f1})
        
        all_train_losses.append(fold_train_losses)
        all_val_losses.append(fold_val_losses)
        all_metrics.append(fold_metrics)
        
        plot_metrics_and_loss(fold_metrics, fold, params.output_dir, classes=['Tumor', 'TILs', 'Others'])
        
        with torch.no_grad():
            for sample_idx in sample_indices:
                sample = dataset[sample_idx]
                images = sample['image'].unsqueeze(0).to(device)
                _, nuclei_pred = model(images)
                save_predictions_to_geojson(nuclei_pred, sample['image_name'], params.output_dir, f"fold_{fold}_sample_{sample_idx}")
    
    plot_metrics_and_loss(all_metrics, None, params.output_dir, classes=['Tumor', 'TILs', 'Others'], train_losses=all_train_losses, val_losses=all_val_losses)

if __name__ == "__main__":
    main()