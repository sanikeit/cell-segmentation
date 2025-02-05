import os
import warnings
import argparse
import torch
from src.train import train_patch_based
from src.config import Config
from src.utils.logger import TrainingLogger
from src.utils.metrics import evaluate_segmentation

# Suppress torchvision warning
warnings.filterwarnings(
    "ignore", 
    "Failed to load image Python extension:"
)

def setup_directories(config):
    """Create necessary directories if they don't exist"""
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description='Cell Segmentation Training')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='batch size (auto-calculated if not specified)')
    parser.add_argument('--aug-strategy', type=str, default='aggressive', 
                       choices=['baseline', 'aggressive', 'minimal'],
                       help='augmentation strategy')
    args = parser.parse_args()

    # Initialize configuration
    config = Config()
    
    # Update config with command line arguments
    config.update_from_args(args)

    # Setup directories
    setup_directories(config)

    # Start training
    print("\nStarting training with configuration:")
    print(f"- Epochs: {config.EPOCHS}")
    print(f"- Batch Size: {config.BATCH_SIZE} (with gradient accumulation steps: {config.GRADIENT_ACCUMULATION_STEPS})")
    print(f"- Patch Size: {config.PATCH_SIZE}")
    print(f"- Scale Factor: {config.SCALE_FACTOR}")
    print(f"- Augmentation Strategy: {config.AUGMENTATION_STRATEGY}")
    print(f"- Data Directory: {config.MONUSEG_DIR}")
    print(f"- Memory-optimized settings:")
    print(f"  - Workers: {config.NUM_WORKERS}")
    print(f"  - Prefetch Factor: {config.PREFETCH_FACTOR}")
    print(f"  - Mixed Precision: {config.USE_MIXED_PRECISION}")
    print()

    train_patch_based(config)

def test_metrics():
    # Create dummy data
    pred = torch.randint(0, 2, (1, 1, 64, 64)).float()
    target = torch.randint(0, 2, (1, 1, 64, 64)).float()
    
    # Test metrics
    dice, iou, accuracy = evaluate_segmentation(pred, target)
    print(f"Dice: {dice:.4f}, IoU: {iou:.4f}, Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
    test_metrics()
