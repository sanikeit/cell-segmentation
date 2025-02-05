import torch
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from pathlib import Path
import logging

def save_model(model, filepath):
    """Save the model weights to the specified filepath."""
    torch.save(model.state_dict(), filepath)

def load_model(model, filepath):
    """Load the model weights from the specified filepath."""
    model.load_state_dict(torch.load(filepath))
    model.eval()
    
def count_cells(mask, min_area=50):
    """Count number of cells in binary mask using connected components."""
    # Convert to binary
    mask = (mask > 0.5).astype(np.uint8)
    
    # Find connected components
    num_labels, labels = cv2.connectedComponents(mask)
    
    # Filter small components
    cell_count = 0
    for label in range(1, num_labels):  # Skip background (0)
        component = (labels == label).astype(np.uint8)
        area = np.sum(component)
        if area >= min_area:
            cell_count += 1
            
    return cell_count

def calculate_metrics(y_true, y_pred):
    """Calculate IoU and Dice metrics."""
    y_true = y_true > 0.5
    y_pred = y_pred > 0.5
    
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    
    iou = intersection / (union + 1e-7)
    dice = 2 * intersection / (y_true.sum() + y_pred.sum() + 1e-7)
    
    return {"iou": iou, "dice": dice}

def reconstruct_from_patches(patches, original_size, patch_size, stride):
    """Enhanced reconstruction with proper handling of overlapping regions"""
    h, w = original_size
    reconstructed = np.zeros((h, w))
    weights = np.zeros((h, w))
    
    # Create weight matrix for smooth blending
    patch_weights = np.ones((patch_size, patch_size))
    for i in range(patch_size//4):
        patch_weights[i, :] *= (i / (patch_size/4))
        patch_weights[patch_size-1-i, :] *= (i / (patch_size/4))
        patch_weights[:, i] *= (i / (patch_size/4))
        patch_weights[:, patch_size-1-i] *= (i / (patch_size/4))
    
    idx = 0
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            reconstructed[y:y+patch_size, x:x+patch_size] += patches[idx] * patch_weights
            weights[y:y+patch_size, x:x+patch_size] += patch_weights
            idx += 1
    
    # Normalize by weights
    reconstructed = np.divide(reconstructed, weights, where=weights != 0)
    return reconstructed

def upscale_predictions(predictions, target_size):
    """Upscale predicted patches back to original size"""
    upscaled = []
    for pred in predictions:
        pred_np = pred.cpu().numpy().squeeze()
        upscaled_pred = cv2.resize(pred_np, target_size, interpolation=cv2.INTER_LINEAR)
        upscaled.append(upscaled_pred)
    return upscaled

def visualize_results(image, true_mask, pred_mask, save_path=None):
    """Visualize original image, ground truth and prediction."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    
    axes[1].imshow(true_mask, cmap='gray')
    axes[1].set_title('Ground Truth')
    
    axes[2].imshow(pred_mask, cmap='gray')
    axes[2].set_title('Prediction')
    
    for ax in axes:
        ax.axis('off')
        
    if save_path:
        plt.savefig(save_path)
    plt.close()

def setup_logger(name, log_file, level=logging.INFO):
    """Set up logger with file and console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    logger.addHandler(ch)
    
    return logger

class MetricTracker:
    """Track training metrics over epochs."""
    def __init__(self):
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_iou': [],
            'val_iou': [],
            'train_dice': [],
            'val_dice': []
        }
    
    def update(self, metrics, phase):
        for k, v in metrics.items():
            self.history[f'{phase}_{k}'].append(v)
    
    def plot_metrics(self, save_path=None):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(self.history['train_loss'], label='Train')
        ax1.plot(self.history['val_loss'], label='Validation')
        ax1.set_title('Loss')
        ax1.legend()
        
        # Metrics plot
        ax2.plot(self.history['train_dice'], label='Train Dice')
        ax2.plot(self.history['val_dice'], label='Val Dice')
        ax2.plot(self.history['train_iou'], label='Train IoU')
        ax2.plot(self.history['val_iou'], label='Val IoU')
        ax2.set_title('Metrics')
        ax2.legend()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()