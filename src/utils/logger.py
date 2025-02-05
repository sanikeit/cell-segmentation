import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import numpy as np
import warnings
from .metrics import calculate_dice_coefficient

class TrainingLogger:
    def __init__(self, config):
        # Create logger
        self.logger = self._setup_logger(config)
        self.config = config
        
        # Initialize metrics history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_dice': [],
            'val_dice': [],
            'train_cell_diff': [],  # Add cell count tracking
            'val_cell_diff': []     # Add cell count tracking
        }
        
        # Best metrics
        self.best_val_dice = 0.0

    def _setup_logger(self, config):
        """Setup logger with file and console handlers"""
        # Create logs directory if it doesn't exist
        os.makedirs(config.LOG_DIR, exist_ok=True)
        
        # Create logger
        logger = logging.getLogger('training')
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        logger.handlers = []
        
        # File handler
        fh = logging.FileHandler(
            os.path.join(config.LOG_DIR, f'training_{datetime.now():%Y%m%d_%H%M%S}.log')
        )
        fh.setLevel(logging.INFO)
        fh_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(fh_formatter)
        logger.addHandler(fh)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch_formatter = logging.Formatter('%(levelname)s - %(message)s')
        ch.setFormatter(ch_formatter)
        logger.addHandler(ch)
        
        return logger

    def log_metrics(self, epoch, train_loss, val_loss, train_dice, val_dice, 
                    train_cell_diff, val_cell_diff):
        """Log metrics for an epoch and return True if model improved"""
        # Update history
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_dice'].append(train_dice)
        self.history['val_dice'].append(val_dice)
        self.history['train_cell_diff'].append(train_cell_diff)
        self.history['val_cell_diff'].append(val_cell_diff)
        
        # Log metrics
        self.logger.info(
            f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train Dice: {train_dice:.4f} | Val Dice: {val_dice:.4f} | "
            f"Train Cell Diff: {train_cell_diff:.1f} | Val Cell Diff: {val_cell_diff:.1f}"
        )
        
        # Check if model improved
        is_best = val_dice > self.best_val_dice
        if is_best:
            self.best_val_dice = val_dice
            self.logger.info(f"New best validation Dice: {val_dice:.4f}")
        
        return is_best

    def save_epoch_visualizations(self, epoch, images, masks, predictions):
        """Save visualization of predictions for multiple patches"""
        if not isinstance(images, torch.Tensor):
            return
            
        # Suppress matplotlib warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Create visualizations directory
            vis_dir = os.path.join(self.config.OUTPUT_DIR, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
            
            # Select first 4 patches for visualization
            num_patches = min(4, images.shape[0])
            fig, axes = plt.subplots(3, num_patches, figsize=(4*num_patches, 12))
            
            for i in range(num_patches):
                # Normalize images to [0, 1] range
                img = images[i].cpu().permute(1, 2, 0).numpy()
                img = (img - img.min()) / (img.max() - img.min())
                
                axes[0, i].imshow(img)
                axes[0, i].set_title(f'Patch {i+1} Input')
                axes[0, i].axis('off')
                
                axes[1, i].imshow(masks[i].cpu().squeeze(), cmap='gray')
                axes[1, i].set_title(f'Patch {i+1} GT')
                axes[1, i].axis('off')
                
                axes[2, i].imshow(predictions[i].cpu().squeeze(), cmap='gray')
                axes[2, i].set_title(f'Patch {i+1} Pred')
                axes[2, i].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f'epoch_{epoch}.png'))
            plt.close()

    def plot_training_history(self):
        """Plot and save training history"""
        # Create plots directory
        plots_dir = os.path.join(self.config.OUTPUT_DIR, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot metrics
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        epochs = range(1, len(self.history['train_loss']) + 1)
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss')
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Dice plot
        ax2.plot(epochs, self.history['train_dice'], 'b-', label='Train Dice')
        ax2.plot(epochs, self.history['val_dice'], 'r-', label='Val Dice')
        ax2.set_title('Training and Validation Dice')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Dice')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'training_history.png'))
        plt.close()
        
        # Add cell count difference plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_cell_diff'], label='Train Cell Count Difference')
        plt.plot(self.history['val_cell_diff'], label='Val Cell Count Difference')
        plt.title('Average Cell Count Difference During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Absolute Cell Count Difference')
        plt.legend()
        plt.savefig(os.path.join(self.config.OUTPUT_DIR, 'plots', 'cell_count_history.png'))
        plt.close()
