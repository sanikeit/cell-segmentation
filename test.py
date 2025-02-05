import torch
import argparse
import warnings
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.models.unet import UNet
from src.data.dataset import MoNuSegDataset
from src.data.transforms import CellSegmentationTransforms
from src.utils.metrics import evaluate_segmentation
from src.utils import load_model
from torch.utils.data import DataLoader
from src.config import Config

def visualize_prediction(image, true_mask, pred_mask, output_path):
    """Visualize a single prediction with proper normalization"""
    plt.figure(figsize=(15, 5))
    
    # Plot original image
    plt.subplot(131)
    plt.imshow(image.transpose(1, 2, 0))
    plt.title('Input Image')
    plt.axis('off')
    
    # Plot ground truth
    plt.subplot(132)
    plt.imshow(true_mask.squeeze(), cmap='gray')
    plt.title('Ground Truth')
    plt.axis('off')
    
    # Plot prediction
    plt.subplot(133)
    plt.imshow(pred_mask.squeeze(), cmap='gray')
    plt.title('Prediction')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def test_single_image(model, patches, masks, device):
    """Test model on a single image's patches"""
    model.eval()
    with torch.no_grad():
        # Move data to device
        patches = patches.to(device)
        masks = masks.to(device)
        
        # Forward pass
        predictions = model(patches)
        
        # Calculate metrics
        dice, iou, accuracy = evaluate_segmentation(predictions, masks)
        
    return {
        'dice': float(dice),
        'iou': float(iou),
        'accuracy': float(accuracy)
    }, predictions

def run_test(args):
    # Suppress warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # Setup
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    config = Config()
    output_dir = Path('test_results')
    output_dir.mkdir(exist_ok=True)
    
    # Initialize model
    model = UNet(
        in_channels=config.UNET_INPUT_CHANNELS,
        out_channels=config.UNET_OUTPUT_CHANNELS,
        features=[32, 64, 128, 256]
    ).to(device)
    
    # Load model weights
    load_model(model, args.model_path)
    print(f"Loaded model from {args.model_path}")
    
    # Update test directory path
    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    
    # Verify test files exist
    image_files = list(test_dir.glob('*.tif'))
    if not image_files:
        raise ValueError(f"No .tif files found in {test_dir}")
    
    print(f"Found {len(image_files)} test images")
    
    # Initialize test dataset
    test_dataset = MoNuSegDataset(
        tissue_dir=str(test_dir),
        annot_dir=str(test_dir),
        split='test',
        patch_size=config.PATCH_SIZE,
        scale_factor=config.SCALE_FACTOR,
        transform=CellSegmentationTransforms.get_transforms(split='val'),
        train_ratio=1.0  # Use all images for testing
    )
    
    print(f"Found {len(test_dataset)} test images")
    
    if len(test_dataset) == 0:
        raise ValueError("Test dataset is empty!")
    
    # Run tests and collect results
    all_metrics = []
    
    for idx in tqdm(range(len(test_dataset)), desc='Testing'):
        patches, masks = test_dataset[idx]
        
        # Process patches in batches
        batch_size = 4
        all_predictions = []
        
        for i in range(0, len(patches), batch_size):
            batch_patches = patches[i:i + batch_size].to(device)
            with torch.no_grad():
                predictions = model(batch_patches)
                all_predictions.append(predictions.cpu())
        
        # Concatenate predictions
        predictions = torch.cat(all_predictions)
        
        # Calculate metrics for this image
        metrics = {}
        metrics['image'] = test_dataset.image_paths[idx].name
        dice, iou, acc = evaluate_segmentation(predictions, masks)
        metrics['dice'] = float(dice)
        metrics['iou'] = float(iou)
        metrics['accuracy'] = float(acc)
        all_metrics.append(metrics)
        
        # Save visualization
        if idx < 5:  # Save first 5 images
            vis_path = output_dir / f'test_result_{idx}.png'
            visualize_prediction(
                patches[0].cpu().numpy(),
                masks[0].cpu().numpy(),
                predictions[0].cpu().numpy(),
                vis_path
            )
    
    # Create summary
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        summary = {
            'Average Dice': df['dice'].mean(),
            'Average IoU': df['iou'].mean(),
            'Average Accuracy': df['accuracy'].mean()
        }
        
        # Save results
        df.to_csv(output_dir / 'detailed_results.csv', index=False)
        
        # Print and save summary
        print("\nTest Results:")
        print("============")
        for metric, value in summary.items():
            print(f"{metric}: {value:.4f}")
        
        with open(output_dir / 'summary.txt', 'w') as f:
            f.write("Test Results Summary\n")
            f.write("==================\n\n")
            for metric, value in summary.items():
                f.write(f"{metric}: {value:.4f}\n")
    else:
        print("No results to report!")

def main():
    parser = argparse.ArgumentParser(description='Test Cell Segmentation Model')
    parser.add_argument('--model_path', type=str, default='outputs/checkpoints/best_model.pth')
    parser.add_argument('--test_dir', type=str, default='MoNuSegTestData')
    args = parser.parse_args()
    
    try:
        # First ensure we have test data
        if not os.path.exists(args.test_dir):
            print("Test directory not found. Preparing test data...")
            from scripts.prepare_test_data import prepare_test_data
            prepare_test_data()
        
        run_test(args)
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        raise

if __name__ == "__main__":
    main()
