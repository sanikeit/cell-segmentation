
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cv2

def visualize_preprocessing(image, mask, patches_img, patches_mask, save_dir, prefix=""):
    """Visualize preprocessing steps and patch extraction"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Original Image and Mask
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    ax1.imshow(image)
    ax1.set_title(f'Original Image\nShape: {image.shape}')
    
    # Draw grid on original image
    h, w = image.shape[:2]
    patch_size = 128
    stride = patch_size // 2
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            rect = plt.Rectangle((x, y), patch_size, patch_size, 
                               fill=False, color='white', linewidth=0.5)
            ax1.add_patch(rect)
    
    # Count and display cells in full image
    from skimage import measure
    binary_mask = mask > 0.5
    labels = measure.label(binary_mask)
    total_cells = len(np.unique(labels)) - 1  # subtract background
    
    ax2.imshow(mask, cmap='gray')
    ax2.set_title(f'Original Mask\nTotal Cells: {total_cells}')
    plt.savefig(save_dir / f'{prefix}original.png')
    plt.close()
    
    # 2. Extracted Patches
    n_patches = len(patches_img)
    cols = min(8, n_patches)
    rows = (n_patches + cols - 1) // cols
    
    fig, axes = plt.subplots(2 * rows, cols, figsize=(cols * 2, rows * 4))
    fig.suptitle(f'Extracted Patches (Total: {n_patches})')
    
    for idx in range(n_patches):
        row = (idx // cols) * 2
        col = idx % cols
        
        # Image patch
        if rows == 1:
            ax_img = axes[0, col]
            ax_mask = axes[1, col]
        else:
            ax_img = axes[row, col]
            ax_mask = axes[row + 1, col]
        
        patch_img = patches_img[idx]
        patch_mask = patches_mask[idx]
        
        # Count cells in patch
        labels = measure.label(patch_mask > 0.5)
        cells_in_patch = len(np.unique(labels)) - 1
        
        ax_img.imshow(patch_img)
        ax_img.set_title(f'Patch {idx+1}\nShape: {patch_img.shape}')
        ax_img.axis('off')
        
        ax_mask.imshow(patch_mask, cmap='gray')
        ax_mask.set_title(f'Mask {idx+1}\nCells: {cells_in_patch}')
        ax_mask.axis('off')
        
    # Fill empty subplots if any
    for idx in range(n_patches, rows * cols):
        row = (idx // cols) * 2
        col = idx % cols
        if rows == 1:
            axes[0, col].axis('off')
            axes[1, col].axis('off')
        else:
            axes[row, col].axis('off')
            axes[row + 1, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_dir / f'{prefix}patches.png')
    plt.close()
    
    # 3. Print statistics
    with open(save_dir / f'{prefix}stats.txt', 'w') as f:
        f.write(f"Original Image Shape: {image.shape}\n")
        f.write(f"Original Mask Shape: {mask.shape}\n")
        f.write(f"Total Cells in Original: {total_cells}\n")
        f.write(f"Number of Patches: {n_patches}\n")
        f.write(f"Patch Size: {patch_size}x{patch_size}\n")
        f.write(f"Stride: {stride}\n")
        
        # Count cells in each patch
        cells_per_patch = []
        for patch_mask in patches_mask:
            labels = measure.label(patch_mask > 0.5)
            cells = len(np.unique(labels)) - 1
            cells_per_patch.append(cells)
        
        f.write(f"Average Cells per Patch: {np.mean(cells_per_patch):.2f}\n")
        f.write(f"Max Cells in a Patch: {max(cells_per_patch)}\n")
        f.write(f"Min Cells in a Patch: {min(cells_per_patch)}\n")
        f.write(f"Total Cells in All Patches: {sum(cells_per_patch)}\n")

# Add to MoNuSegDataset class