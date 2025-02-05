import os
import torch
import torch.nn.functional as F  # Add this import
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
from src.data.dataset import MoNuSegDataset
from src.models.unet import UNet
from src.utils.logger import TrainingLogger
from src.data.transforms import CellSegmentationTransforms as transforms
from src.config import Config
from src.utils import (
    save_model,
    upscale_predictions,
    reconstruct_from_patches,
    calculate_dice_coefficient,
    count_cells
)
from src.utils.cell_counter import CellCounter

def get_transforms(split):
    return transforms.get_transforms(split=split)

def get_device():
    """Get the best available device (CUDA, MPS, or CPU)"""
    if torch.cuda.is_available():
        return torch.device('cuda'), True
    elif torch.backends.mps.is_available():
        return torch.device('mps'), False
    else:
        return torch.device('cpu'), False

def train_epoch(model, loader, criterion, optimizer, device, use_amp, scaler, desc="Training"):
    """Optimized training epoch"""
    cell_counter = CellCounter(min_cell_size=50)  # Initialize counter once
    running_loss = 0
    running_dice = 0
    running_cell_diff = 0  # Track cell count difference
    num_batches = 0
    
    for patches, masks in loader:
        try:
            B, N, C, H, W = patches.shape
            patches = patches.view(-1, C, H, W).to(device, non_blocking=True)
            masks = masks.view(-1, 1, H, W).to(device, non_blocking=True)
            
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(patches)
                if isinstance(outputs, tuple):
                    main_output, aux_outputs = outputs
                    # Main loss with label smoothing
                    masks_smooth = masks * 0.9 + 0.05
                    loss = criterion(main_output, masks_smooth)
                    
                    try:
                        # Calculate cell count loss with scaling
                        with torch.no_grad():
                            pred_binary = (main_output.sigmoid() > 0.5).cpu()
                            pred_counts = []
                            true_counts = []
                            
                            for p, m in zip(pred_binary, masks):
                                p_count, _ = cell_counter.count_cells(p.numpy())
                                m_count, _ = cell_counter.count_cells(m.cpu().numpy())
                                pred_counts.append(p_count)
                                true_counts.append(m_count)
                            
                            pred_counts = torch.tensor(pred_counts, device=device).float()
                            true_counts = torch.tensor(true_counts, device=device).float()
                            
                            if pred_counts.numel() > 0 and true_counts.numel() > 0:
                                cell_loss = F.l1_loss(pred_counts, true_counts)
                                loss += 0.2 * cell_loss  # Increased weight for cell count loss
                                
                                # Track absolute difference in cell counts
                                with torch.no_grad():
                                    cell_diff = torch.abs(pred_counts - true_counts).mean()
                                    running_cell_diff += cell_diff.item()
                                    
                                    if num_batches % 10 == 0:  # Log every 10 batches
                                        print(f"\nPred cells: {pred_counts.cpu().numpy().mean():.1f}, "
                                              f"True cells: {true_counts.cpu().numpy().mean():.1f}")
                    except Exception as e:
                        print(f"Warning: Cell counting error: {e}")
                    
                    # Auxiliary losses
                    aux_weights = [0.4, 0.3, 0.2, 0.1]
                    for aux_out, weight in zip(aux_outputs, aux_weights):
                        loss += weight * criterion(aux_out, masks_smooth)
                else:
                    main_output = outputs
                    loss = criterion(main_output, masks)
            
            # Add L2 regularization
            l2_lambda = 0.0001
            l2_reg = torch.tensor(0., device=device)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += l2_lambda * l2_reg

            if model.training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            
            # Calculate cell count difference
            with torch.no_grad():
                pred_binary = (main_output.sigmoid() > 0.5).float()
                pred_cell_counts = torch.tensor([count_cells(p.cpu().numpy()) for p in pred_binary])
                true_cell_counts = torch.tensor([count_cells(m.cpu().numpy()) for m in masks])
                cell_diff = torch.abs(pred_cell_counts - true_cell_counts).float().mean()
                running_cell_diff += cell_diff.item()
            
            running_loss += loss.item()
            running_dice += calculate_dice_coefficient(main_output.detach(), masks)
            num_batches += 1
            
        except RuntimeError as e:
            print(f"Error during {desc.lower()}: {e}")
            continue
    
    metrics = {
        'loss': running_loss / max(1, num_batches),
        'dice': running_dice / max(1, num_batches),
        'cell_diff': running_cell_diff / max(1, num_batches)
    }
    return metrics

def save_epoch_visualizations(epoch, model, val_loader, device, vis_dir, num_samples=4):
    """Save visualization of model predictions"""
    model.eval()
    os.makedirs(vis_dir, exist_ok=True)
    
    # Get a batch of validation data
    val_iter = iter(val_loader)
    patches, masks = next(val_iter)
    
    # Handle the batch dimension
    B, N, C, H, W = patches.shape  # B=batch_size, N=num_patches
    patches = patches.view(-1, C, H, W).to(device)  # Combine batch and patches dimensions
    masks = masks.view(-1, 1, H, W).to(device)
    
    with torch.no_grad():
        # Get predictions
        predictions = model(patches)
        
        # Reshape back to batch form
        patches = patches.view(B, N, C, H, W)
        masks = masks.view(B, N, 1, H, W)
        predictions = predictions.view(B, N, 1, H, W)
        
        # Create visualization grid
        fig, axes = plt.subplots(3, num_samples, figsize=(4*num_samples, 12))
        
        for i in range(min(num_samples, N)):  # Use N instead of batch size
            # Input image
            img = patches[0, i].cpu().permute(1, 2, 0).numpy()  # Take first batch
            img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0,1]
            axes[0, i].imshow(img)
            axes[0, i].set_title(f'Input {i+1}')
            axes[0, i].axis('off')
            
            # Ground truth
            mask = masks[0, i].cpu().squeeze().numpy()
            axes[1, i].imshow(mask, cmap='gray')
            axes[1, i].set_title(f'Ground Truth {i+1}')
            axes[1, i].axis('off')
            
            # Prediction
            pred = predictions[0, i].cpu().squeeze().numpy()
            axes[2, i].imshow(pred, cmap='gray')
            axes[2, i].set_title(f'Prediction {i+1}')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'epoch_{epoch}.png'))
        plt.close()

def train_patch_based(config):
    # Suppress warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    # Get device and AMP capability
    device, use_amp = get_device()
    print(f"Using device: {device}")
    
    # Enable async data loading
    torch.multiprocessing.set_start_method('spawn', force=True)
    
    # Initialize train dataset
    train_dataset = MoNuSegDataset(
        tissue_dir=config.TISSUE_IMAGES_DIR,
        annot_dir=config.ANNOTATIONS_DIR,
        split='train',
        patch_size=config.PATCH_SIZE,
        scale_factor=config.SCALE_FACTOR,
        transform=get_transforms('train'),
        train_ratio=config.TRAIN_SPLIT,
        patches_per_image=config.PATCHES_PER_IMAGE,
        cache_size=config.CACHE_SIZE
    )
    
    # Initialize validation dataset
    val_dataset = MoNuSegDataset(
        tissue_dir=config.TISSUE_IMAGES_DIR,
        annot_dir=config.ANNOTATIONS_DIR,
        split='val',
        patch_size=config.PATCH_SIZE,
        scale_factor=config.SCALE_FACTOR,
        transform=get_transforms('val'),
        train_ratio=config.TRAIN_SPLIT,
        patches_per_image=config.PATCHES_PER_IMAGE,
        cache_size=config.CACHE_SIZE
    )
    
    # Create data loaders with improved settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    # Initialize model for 128x128 input
    model = UNet(
        in_channels=config.UNET_INPUT_CHANNELS,
        out_channels=config.UNET_OUTPUT_CHANNELS,
        features=[32, 64, 128, 256]  # Reduced feature maps for smaller input
    )
    
    # Move model to device
    model = model.to(device)
    
    # Use combined loss function
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Use AdamW with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=0.01
    )
    
    # Use Mixed Precision Training
    scaler = torch.cuda.amp.GradScaler()
    
    # Initialize logger
    logger = TrainingLogger(config)
    
    # Use OneCycle learning rate scheduler with warmup
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.LEARNING_RATE,
        epochs=config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,  # 30% warmup
        div_factor=25.0,
        final_div_factor=1000.0
    )
    
    # Training loop with progress bar
    progress_bar = tqdm(range(config.EPOCHS), desc='Training')
    best_val_dice = 0
    
    for epoch in range(config.EPOCHS):
        # Training phase
        model.train()
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, use_amp, scaler,
            desc=f"Epoch {epoch+1}/{config.EPOCHS}"
        )
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_metrics = train_epoch(
                model, val_loader, criterion, None, device, False, None,
                desc="Validation"
            )
        
        # Update learning rate
        scheduler.step()
        
        # Update progress bar with cell count metrics
        progress_bar.set_postfix({
            'Train Loss': f"{train_metrics['loss']:.4f}",
            'Val Loss': f"{val_metrics['loss']:.4f}",
            'Train Dice': f"{train_metrics['dice']:.4f}",
            'Val Dice': f"{val_metrics['dice']:.4f}",
            'Train Cell Diff': f"{train_metrics['cell_diff']:.1f}",
            'Val Cell Diff': f"{val_metrics['cell_diff']:.1f}",
            'LR': f"{optimizer.param_groups[0]['lr']:.6f}"
        })
        
        # Save best model
        if val_metrics['dice'] > best_val_dice:
            best_val_dice = val_metrics['dice']
            save_model(model, os.path.join(config.CHECKPOINT_DIR, 'best_model.pth'))
            progress_bar.write(f"New best validation Dice: {val_metrics['dice']:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config.CHECKPOINT_FREQ == 0:
            checkpoint_path = os.path.join(
                config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch+1}.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_dice': best_val_dice,
            }, checkpoint_path)
        
        # Log metrics including cell count
        logger.log_metrics(
            epoch, 
            train_metrics['loss'], 
            val_metrics['loss'],
            train_metrics['dice'], 
            val_metrics['dice'],
            train_metrics['cell_diff'],
            val_metrics['cell_diff']
        )
        
        # Save visualizations
        if (epoch + 1) % config.VIS_FREQ == 0:
            save_epoch_visualizations(
                epoch, model, val_loader, device, 
                config.VIS_DIR, num_samples=4
            )
        
        progress_bar.update(1)
    
    progress_bar.close()
    logger.plot_training_history()

def test_with_tta(model, image, transforms):
    """Test-time augmentation"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for transform in transforms:
            # Apply transform
            augmented = transform(image=image)
            input_tensor = augmented['image'].unsqueeze(0)
            
            # Get prediction
            pred = model(input_tensor)
            
            # Reverse augmentation if needed (e.g., for flips and rotations)
            if 'HorizontalFlip' in str(transform):
                pred = torch.flip(pred, dims=[-1])
            elif 'VerticalFlip' in str(transform):

                pred = torch.flip(pred, dims=[-2])
            elif 'Rotate' in str(transform):
                pred = torch.rot90(pred, k=-1, dims=[-2, -1])
            
            predictions.append(pred)
    
    # Average predictions
    return torch.mean(torch.stack(predictions), dim=0)

    main()
def main():
    config = Config()
    train_patch_based(config)

if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()