import os
import logging
import argparse
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import random

from data_tissue import PUMADatasetTissue
from tissue import TissueModel
from utils_tissue import compute_metrics, save_sample_image, plot_metrics_and_loss
from monai.losses import DiceFocalLoss

sns.set_theme(style="darkgrid")
plt.rcParams['image.cmap'] = 'rocket'

class Params:
    def __init__(self):
        self.root = "dataset/01_training_dataset_tif_ROIs"
        self.tissue_root = "dataset/01_training_dataset_geojson_tissue"
        self.output_dir = "puma_output/multi_tissue"
        self.patch_size = 256
        self.stride = 128
        self.batch_size = 16
        self.num_epochs = 16
        self.num_folds = 5
        self.val_interval = 2
        self.lr = 0.002
        self.weight_decay = 1e-4
        self.warmup_epochs = 5
        self.num_workers = 62
        self.reduced_load = False
        self.reduced_size = 31

def setup_logging(debug=False):
    os.makedirs(Params().output_dir, exist_ok=True)
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(Params().output_dir, 'multi_tissue_debug.log')),
        ]
    )

def train_one_epoch(model, loader, optimizer, criterion, device, params, args, fold, epoch, inter, val, sample_indices, dataset):
    model.train()
    epoch_loss = 0
    batch_count = 0
    
    progress_bar = tqdm(loader, desc=f"Fold {fold+1}, Epoch {epoch+1}: Loss=0.0000", leave=False, total=len(loader), colour="green")
    
    for batch_idx, batch in enumerate(progress_bar):
        images = batch['image'].to(device)
        tissue_class_gt = batch['tissue_class_mask'].to(device)
        
        # Add channel dimension to target for DiceFocalLoss
        tissue_class_gt = tissue_class_gt.unsqueeze(1)  # Shape: [batch_size, 1, height, width]
        
        optimizer.zero_grad()
        tissue_pred = model(images)
        
        # Handle deep supervision output
        if not isinstance(tissue_pred, (list, tuple)):
            tissue_pred = [tissue_pred]  # Wrap single tensor in a list for consistency
        
        main_pred = tissue_pred[0]  # Main output
        aux_preds = tissue_pred[1:]  # Auxiliary outputs
        
        # Log shapes for debugging
        logging.debug(f"Batch {batch_idx} - main_pred shape: {main_pred.shape}, tissue_class_gt shape: {tissue_class_gt.shape}")
        
        # Compute main loss
        main_loss = criterion(main_pred, tissue_class_gt)
        logging.debug(f"Batch {batch_idx} - main_loss: {main_loss.item():.4f}")
        
        # Compute auxiliary losses
        aux_weight = 0.4
        aux_loss = 0
        for i, aux_pred in enumerate(aux_preds):
            logging.debug(f"Batch {batch_idx} - aux_pred {i} shape: {aux_pred.shape}")
            aux_loss += criterion(aux_pred, tissue_class_gt)
        if len(aux_preds) > 0:
            aux_loss /= len(aux_preds)
            logging.debug(f"Batch {batch_idx} - avg aux_loss: {aux_loss.item():.4f}")
        
        loss = main_loss + aux_weight * aux_loss
        
        if not torch.isfinite(loss):
            logging.warning(f"NaN loss detected in batch {batch_idx}, skipping")
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        epoch_loss += loss.item()
        batch_count += 1
        
        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
        progress_bar.set_description(f"Fold {fold+1}, Epoch {epoch+1}: Loss={avg_loss:.4f}")
    
    avg_loss = epoch_loss / max(batch_count, 1)
    logging.info(f"Fold {fold+1}, Epoch {epoch+1} - Training Loss: {avg_loss:.4f}")
    return avg_loss

def validate(model, loader, criterion, device, params, args, fold, epoch, sample_indices, dataset):
    random_idx = random.randint(0, len(loader) - 1)
    model.eval()
    total_loss = 0.0
    metrics = []
    
    with torch.no_grad():
        progress_bar = tqdm(loader, desc=f"Fold {fold+1}, Epoch {epoch+1}: Validating", leave=False, colour="blue", total=len(loader))
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(device)
            tissue_class_gt = batch['tissue_class_mask'].to(device)
            
            tissue_class_gt = tissue_class_gt.unsqueeze(1)
            
            tissue_pred = model(images)
            
            # Handle deep supervision output
            if not isinstance(tissue_pred, (list, tuple)):
                tissue_pred = [tissue_pred]  # Wrap single tensor in a list for consistency
            
            main_pred = tissue_pred[0]
            
            # Log shapes for debugging
            logging.debug(f"Validation Batch {batch_idx} - main_pred shape: {main_pred.shape}, tissue_class_gt shape: {tissue_class_gt.shape}")
            
            loss = criterion(main_pred, tissue_class_gt)
            total_loss += loss.item()
            
            metrics.append((main_pred, tissue_class_gt.squeeze(1)))
            
            if batch_idx % 100 == 0:
                save_sample_image(
                    images,
                    tissue_class_gt.squeeze(1).cpu(),
                    main_pred,
                    fold=fold,
                    epoch=epoch,
                    output_dir=params.output_dir,
                    idx=0,
                    suffix=f"val_batch_{batch_idx}",
                    pretrain=False,
                    image_name=f"val_batch_{batch_idx}"
                )
    
    avg_loss = total_loss / len(loader)
    
    all_preds = torch.cat([pred for pred, gt in metrics], dim=0)
    all_gts = torch.cat([gt for pred, gt in metrics], dim=0)
    macro_dice, micro_dice = compute_metrics(all_preds, all_gts, num_classes=6)
    
    logging.info(f"Fold {fold+1}, Epoch {epoch+1} - Validation Loss: {avg_loss:.4f}, Macro Dice: {macro_dice:.4f}")
    return avg_loss, macro_dice, micro_dice

def main():
    parser = argparse.ArgumentParser(description="Train multi-class tissue segmentation model")
    parser.add_argument('--root', default="dataset/01_training_dataset_tif_ROIs", help="Path to image directory")
    parser.add_argument('--tissue_root', default="dataset/01_training_dataset_geojson_tissue", help="Path to tissue GeoJSON directory")
    parser.add_argument('--output_dir', default="puma_output/multi_tissue", help="Output directory")
    parser.add_argument('--patch_size', type=int, default=256, help="Patch size")
    parser.add_argument('--stride', type=int, default=128, help="Stride for patch extraction")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    parser.add_argument('--num_epochs', type=int, default=52, help="Number of epochs")
    parser.add_argument('--num_folds', type=int, default=5, help="Number of folds")
    parser.add_argument('--val_interval', type=int, default=4, help="Validation interval (epochs)")
    parser.add_argument('--lr', type=float, default=0.002, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="Weight decay")
    parser.add_argument('--debug', action='store_true', help="Disable debug logging and images")
    args = parser.parse_args()
    
    setup_logging(args.debug)
    
    params = Params()
    params.root = args.root
    params.tissue_root = args.tissue_root
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
    dataset = PUMADatasetTissue(
        params.root,
        params.tissue_root,
        patch_size=params.patch_size,
        stride=params.stride,
        mode='train',
        reduced_load=params.reduced_load,
        reduced_size=params.reduced_size
    )
    logging.info(f"Dataset loaded in {time.time() - start_time:.2f} seconds")
    logging.info(f"Dataset size - 7x7 patches per image * number of images: {len(dataset)}")
    
    sample_indices = np.random.choice(len(dataset), 3, replace=False)
    for sample_idx in sample_indices:
        sample = dataset[sample_idx]
        images = sample['image'].unsqueeze(0)
        tissue_gt = sample['tissue_class_mask'].unsqueeze(0)
        save_sample_image(
            images,
            tissue_gt,
            tissue_pred=None,
            fold=0,
            epoch=0,
            output_dir=params.output_dir,
            idx=0,
            suffix=f"pretrain_sample_{sample_idx}",
            pretrain=True,
            image_name=sample['image_name']
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
        
        train_loader = DataLoader(
            train_subset,
            batch_size=params.batch_size,
            sampler=dataset.get_sampler(subset_indices=train_idx),
            num_workers=params.num_workers,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=params.batch_size,
            shuffle=False,
            num_workers=params.num_workers,
            pin_memory=True
        )
        
        total_iterations = len(train_loader) * params.num_epochs
        dynamic_warmup_epochs = min(max(1, params.num_epochs // 2), params.num_epochs)
        warmup_iterations = len(train_loader) * dynamic_warmup_epochs
        start_factor = max(0.01, 0.1 / (len(train_loader) / 1000))
        cosine_iterations = max(1, total_iterations - warmup_iterations)
        t_max = max(1, cosine_iterations // len(train_loader))
        

        model = TissueModel().to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=start_factor, end_factor=1.0, total_iters=warmup_iterations)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=params.lr * 0.01)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_iterations])
        
        criterion = DiceFocalLoss(
            include_background=True,
            softmax=True,
            to_onehot_y=True,
            lambda_dice=0.7,
            lambda_focal=0.3,
            weight=torch.tensor(dataset.class_weights, dtype=torch.float32).to(device)
        )
        
        logging.info(f"Fold {fold+1} Scheduler: Warmup {dynamic_warmup_epochs} epochs ({warmup_iterations} iterations, start_factor={start_factor:.3f}), "
                     f"Cosine T_max={t_max} epochs, eta_min={params.lr * 0.01:.6f}, val_interval={params.val_interval} epochs")
        
        best_val_loss = float('inf')
        fold_train_losses = []
        fold_val_losses = []
        fold_metrics = []
        
        for epoch in tqdm(range(params.num_epochs), desc=f"Fold {fold+1} Epochs", leave=True, total=params.num_epochs, colour="red"):
            logging.info(f"Fold {fold+1}, Epoch {epoch + 1}/{params.num_epochs}")
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, params, args, fold, epoch, 400, val_loader, sample_indices, dataset)
            fold_train_losses.append(train_loss)
            
            if (epoch + 1) % params.val_interval == 0:
                logging.info(f"Validating at Fold {fold+1}, Epoch {epoch+1}")
                val_loss, macro_dice, micro_dice = validate(model, val_loader, criterion, device, params, args, fold, epoch, sample_indices, dataset)
                logging.info(f"Fold {fold+1}, Epoch {epoch+1} - Val Loss: {val_loss:.4f}, Macro Dice: {macro_dice:.4f}")
                fold_val_losses.append(val_loss)
                fold_metrics.append({'macro_dice': macro_dice, 'micro_dice': micro_dice})
            
            scheduler.step()
            logging.info(f"Fold {fold+1}, Epoch {epoch+1} - Learning rate: {scheduler.get_last_lr()}")
            
            if len(fold_val_losses) > 0 and fold_val_losses[-1] < best_val_loss:
                best_val_loss = fold_val_losses[-1]
                torch.save(model.state_dict(), os.path.join(params.output_dir, f"best_model_fold_{fold}.pth"))
                logging.info(f"Saved best model for Fold {fold+1} with validation loss: {best_val_loss:.4f}")
        
        final_val_loss, macro_dice, micro_dice = validate(model, val_loader, criterion, device, params, args, fold, params.num_epochs, sample_indices, dataset)
        logging.info(f"Fold {fold+1} Final Validation - Loss: {final_val_loss:.4f}, Macro Dice: {macro_dice:.4f}")
        fold_val_losses.append(final_val_loss)
        fold_metrics.append({'macro_dice': macro_dice, 'micro_dice': micro_dice})
        
        all_train_losses.append(fold_train_losses)
        all_val_losses.append(fold_val_losses)
        all_metrics.append(fold_metrics)
        
        plot_metrics_and_loss(fold_metrics, fold, params.output_dir, classes=['White Bg', 'Stroma', 'Blood Vessel', 'Tumor', 'Epidermis', 'Necrosis'])
    
    plot_metrics_and_loss(all_metrics, None, params.output_dir, classes=['White Bg', 'Stroma', 'Blood Vessel', 'Tumor', 'Epidermis', 'Necrosis'], train_losses=all_train_losses, val_losses=all_val_losses)

if __name__ == "__main__":
    main()