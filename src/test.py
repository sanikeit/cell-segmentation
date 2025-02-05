import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.models.unet import UNet
from src.data.dataset import MoNuSegDataset
from src.data.transforms import CellSegmentationTransforms
from src.utils.metrics import evaluate_segmentation
from src.config import Config

def load_model(model_path, config):
    """Load trained model"""
    model = UNet(
        in_channels=config.UNET_INPUT_CHANNELS,
        out_channels=config.UNET_OUTPUT_CHANNELS,
        features=[32, 64, 128, 256]
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    return model

def test_single_image(model, image, device, tta_transforms):
    """Test single image with TTA"""
    predictions = []
    
    for transform in tta_transforms:
        # Apply transform
        augmented = transform(image=image)['image'].unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            pred = model(augmented)
            
        # Move prediction to CPU and ensure correct dimensions
        pred = pred.cpu().squeeze()  # Remove batch dimension
        predictions.append(pred)
    
    # Average predictions
    final_pred = torch.mean(torch.stack(predictions), dim=0)
    return final_pred

def visualize_results(image, mask, prediction, save_path):
    """Visualize and save test results with patch overlay"""
    fig = plt.figure(figsize=(20, 10))
    
    # Full image visualization
    ax1 = plt.subplot2grid((2, 4), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((2, 4), (0, 2), colspan=2)
    ax3 = plt.subplot2grid((2, 4), (1, 0), colspan=2)
    ax4 = plt.subplot2grid((2, 4), (1, 2), colspan=2)
    
    # Original image with patch grid overlay
    ax1.imshow(image)
    h, w = image.shape[:2]
    patch_size = 128  # Adjust based on your config
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            rect = plt.Rectangle((j, i), patch_size, patch_size, 
                               fill=False, color='white', linewidth=0.5)
            ax1.add_patch(rect)
    ax1.set_title('Input Image with Patch Grid')
    ax1.axis('off')
    
    # Ground truth mask
    ax2.imshow(mask, cmap='gray')
    ax2.set_title(f'Ground Truth (Cells: {count_cells(mask)})')
    ax2.axis('off')
    
    # Prediction
    ax3.imshow(prediction, cmap='gray')
    ax3.set_title(f'Prediction (Cells: {count_cells(prediction)})')
    ax3.axis('off')
    
    # Overlay
    overlay = image.copy()
    overlay[prediction > 0.5] = [255, 0, 0]  # Red overlay for predictions
    ax4.imshow(overlay)
    ax4.set_title('Prediction Overlay')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def test_model(config, model_path):
    """Full testing pipeline"""
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = load_model(model_path, config)
    model = model.to(device)
    model.eval()
    
    # Initialize test dataset
    test_dataset = MoNuSegDataset(
        tissue_dir=config.TISSUE_IMAGES_DIR,
        annot_dir=config.ANNOTATIONS_DIR,
        split='val',
        patch_size=config.PATCH_SIZE,
        scale_factor=config.SCALE_FACTOR,
        transform=None,  # No transforms initially
        train_ratio=config.TRAIN_SPLIT,
        patches_per_image=config.PATCHES_PER_IMAGE
    )
    
    # Get TTA transforms
    tta_transforms = CellSegmentationTransforms.get_tta_transforms()
    
    # Setup output directory
    output_dir = Path(config.OUTPUT_DIR) / 'test_results'
    output_dir.mkdir(exist_ok=True)
    
    # Testing loop
    dice_scores = []
    iou_scores = []
    acc_scores = []
    
    print("Starting testing...")
    for idx in tqdm(range(len(test_dataset))):
        # Get original image and mask
        image, mask = test_dataset.preloaded_data[idx]
        
        # Extract patches with overlap
        stride = config.PATCH_SIZE // 2  # 50% overlap
        h, w = image.shape[:2]
        patches_img = []
        patch_positions = []
        
        for y in range(0, h - config.PATCH_SIZE + 1, stride):
            for x in range(0, w - config.PATCH_SIZE + 1, stride):
                patch = image[y:y + config.PATCH_SIZE, x:x + config.PATCH_SIZE]
                patches_img.append(patch)
                patch_positions.append((y, x))
        
        # Process patches with TTA
        predictions = []
        weights = np.zeros((h, w))
        full_pred = np.zeros((h, w))
        
        for patch, (y, x) in zip(patches_img, patch_positions):
            pred = test_single_image(model, patch, device, tta_transforms)
            pred_np = pred.numpy()
            
            # Ensure pred_np is 2D
            if len(pred_np.shape) > 2:
                pred_np = pred_np.squeeze()
            
            # Apply Gaussian weighting for smooth blending
            weight = np.exp(-((np.arange(config.PATCH_SIZE) - config.PATCH_SIZE/2)**2)/(config.PATCH_SIZE/4)**2)
            weight_matrix = weight[:, np.newaxis] * weight[np.newaxis, :]
            
            # Accumulate predictions and weights
            full_pred[y:y + config.PATCH_SIZE, x:x + config.PATCH_SIZE] += pred_np * weight_matrix
            weights[y:y + config.PATCH_SIZE, x:x + config.PATCH_SIZE] += weight_matrix
        
        # Average overlapping regions
        full_pred = np.divide(full_pred, weights, where=weights > 0)
        
        # Convert to numpy for metrics
        pred_np = full_pred
        mask_np = mask
        
        # Calculate metrics
        dice, iou, acc = evaluate_segmentation(
            torch.from_numpy(full_pred).unsqueeze(0).unsqueeze(0),  # Add batch and channel dims
            torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
        )
        
        dice_scores.append(dice)
        iou_scores.append(iou)
        acc_scores.append(acc)
        
        # Visualize and save results
        visualize_results(
            image,
            mask_np,
            pred_np,
            output_dir / f'test_case_{idx}.png'
        )
    
    # Print final metrics
    print("\nTest Results:")
    print(f"Average Dice Score: {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}")
    print(f"Average IoU Score: {np.mean(iou_scores):.4f} ± {np.std(iou_scores):.4f}")
    print(f"Average Accuracy: {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")
    
    # Save metrics to file
    metrics_path = output_dir / 'test_metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write(f"Average Dice Score: {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}\n")
        f.write(f"Average IoU Score: {np.mean(iou_scores):.4f} ± {np.std(iou_scores):.4f}\n")
        f.write(f"Average Accuracy: {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}\n")

if __name__ == "__main__":
    config = Config()
    model_path = Path(config.CHECKPOINT_DIR) / 'best_model.pth'
    test_model(config, model_path)
