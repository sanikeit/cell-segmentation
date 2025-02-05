from .metrics import calculate_dice_coefficient, calculate_iou, evaluate_segmentation
from typing import Union, Tuple
import torch
import numpy as np
import cv2

def save_model(model, filepath: str) -> None:
    """Save the model weights to the specified filepath."""
    torch.save(model.state_dict(), filepath)

def load_model(model, filepath: str) -> None:
    """Load the model weights from the specified filepath."""
    model.load_state_dict(torch.load(filepath, weights_only=True))
    model.eval()

def upscale_predictions(predictions: list, target_size: tuple) -> list:
    """Upscale predicted patches back to original size"""
    upscaled = []
    for pred in predictions:
        pred_np = pred.cpu().numpy().squeeze()
        upscaled_pred = cv2.resize(pred_np, target_size, interpolation=cv2.INTER_LINEAR)
        upscaled.append(upscaled_pred)
    return upscaled

def reconstruct_from_patches(patches: list, original_size: tuple, 
                           patch_size: int, stride: int) -> 'numpy.ndarray':
    """Reconstruct full image from patches"""
    h, w = original_size
    reconstructed = np.zeros((h, w))
    weights = np.zeros((h, w))
    
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
    
    reconstructed = np.divide(reconstructed, weights, where=weights != 0)
    return reconstructed

def count_cells(mask: np.ndarray, min_area: int = 50) -> int:
    """Count number of cells in binary mask using connected components."""
    from skimage import measure
    
    # Ensure binary mask
    binary_mask = mask > 0.5 if mask.dtype == np.float32 or mask.dtype == np.float64 else mask > 127
    
    # Label connected components
    labels = measure.label(binary_mask)
    
    # Filter small components (noise)
    min_cell_size = 50  # adjust based on your data
    unique_labels = np.unique(labels)
    cell_count = 0
    
    for label in unique_labels[1:]:  # Skip background (0)
        if np.sum(labels == label) >= min_cell_size:
            cell_count += 1
            
    return cell_count

def visualize_results(image: np.ndarray, true_mask: np.ndarray, pred_mask: np.ndarray, 
                     save_path: str = None) -> None:
    """Visualize results with proper shape handling."""
    import matplotlib.pyplot as plt
    
    # Ensure all inputs are in correct format
    if len(image.shape) == 4:
        image = image[0]
    if len(true_mask.shape) == 4:
        true_mask = true_mask[0]
    if len(pred_mask.shape) == 4:
        pred_mask = pred_mask[0]
    
    # Normalize image if needed
    if image.max() > 1:
        image = image / 255.0
    
    # Convert masks to binary
    true_mask = (true_mask > 0.5).astype(np.uint8)
    pred_mask = (pred_mask > 0.5).astype(np.uint8)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    true_count = count_cells(true_mask)
    axes[1].imshow(true_mask, cmap='gray')
    axes[1].set_title(f'Ground Truth\n{true_count} cells')
    axes[1].axis('off')
    
    pred_count = count_cells(pred_mask)
    axes[2].imshow(pred_mask, cmap='gray')
    axes[2].set_title(f'Prediction\n{pred_count} cells')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

__all__ = [
    'calculate_dice_coefficient',
    'calculate_iou',
    'evaluate_segmentation',
    'save_model',
    'load_model',
    'upscale_predictions',
    'reconstruct_from_patches',
    'count_cells',
    'visualize_results'
]