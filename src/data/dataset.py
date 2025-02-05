import os
import numpy as np
import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
from pathlib import Path
import cv2

class MoNuSegDataset(Dataset):
    def __init__(self, tissue_dir, annot_dir, split='train', patch_size=128, 
                 scale_factor=0.5, transform=None, train_ratio=0.85, 
                 patches_per_image=32, cache_size=100):
        self.tissue_dir = Path(tissue_dir)
        self.annot_dir = Path(annot_dir)
        
        # Verify directories exist
        if not self.tissue_dir.exists():
            raise FileNotFoundError(f"Tissue directory not found: {self.tissue_dir}")
        if not self.annot_dir.exists():
            raise FileNotFoundError(f"Annotation directory not found: {self.annot_dir}")
            
        # Get all image paths
        self.image_paths = sorted(list(self.tissue_dir.glob('*.tif')))
        if not self.image_paths:
            raise ValueError(f"No .tif files found in {self.tissue_dir}")
            
        # Rest of initialization
        self.split = split
        self.patch_size = patch_size
        self.scaled_size = int(patch_size * scale_factor)
        self.scale_factor = scale_factor
        self.transform = transform
        self.stride = patch_size // 4  # Initialize stride attribute
        self.min_cell_area = 50  # Minimum cell area to count
        
        num_train = int(len(self.image_paths) * train_ratio)
        if num_train == 0:
            raise ValueError("Training set is empty. Check train_ratio or number of images.")
            
        # Split dataset
        if split == 'train':
            self.image_paths = self.image_paths[:num_train]
        else:  # validation
            self.image_paths = self.image_paths[num_train:]
            
        # Get corresponding annotation paths
        self.xml_paths = []
        for img_path in self.image_paths:
            xml_path = self.annot_dir / f"{img_path.stem}.xml"
            if not xml_path.exists():
                raise FileNotFoundError(f"Annotation file not found: {xml_path}")
            self.xml_paths.append(xml_path)
            
        print(f"Initialized {split} dataset with {len(self.image_paths)} images")
        
        self.patches_per_image = patches_per_image
        self.max_cache_size = cache_size
        self.cache = {}
        
        # Preload with reduced memory usage
        self.preloaded_data = {}
        print(f"Preloading {len(self.image_paths)} images...")
        for idx, (img_path, xml_path) in enumerate(zip(self.image_paths, self.xml_paths)):
            # Load image at reduced size
            image = cv2.imread(str(img_path))
            image = cv2.resize(image, (1000, 1000))  # Resize to standard size
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = self._xml_to_mask(xml_path)
            self.preloaded_data[idx] = (image, mask)
        
        # Add visualization of first few images
        from src.utils.visualization import visualize_preprocessing
        vis_dir = Path("debug_visualization") / split
        
        print(f"\nAnalyzing preprocessing for first 3 images in {split} set...")
        for idx in range(min(3, len(self.image_paths))):
            image, mask = self.preloaded_data[idx]
            patches_img, patches_mask = self.extract_patches(image, mask)
            visualize_preprocessing(
                image, mask, patches_img, patches_mask,
                vis_dir, prefix=f"image_{idx}_"
            )
        
    def __len__(self):
        return len(self.image_paths)
    
    def _xml_to_mask(self, xml_path):
        """Convert XML annotations to binary mask"""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Get image dimensions from XML
        regions = root.findall(".//Region")
        mask = np.zeros((1000, 1000), dtype=np.uint8)  # Assuming fixed size
        
        # Draw contours for each annotation
        for region in regions:
            vertices = region.findall(".//Vertex")
            points = np.array([[float(v.get('X')), float(v.get('Y'))] 
                             for v in vertices])
            points = points.astype(np.int32)
            cv2.fillPoly(mask, [points], 1)
            
        return mask

    def extract_patches(self, image, mask):
        """Improved patch extraction with cell-aware sampling"""
        h, w = image.shape[:2]
        patches_img = []
        patches_mask = []
        patch_scores = []  # Store importance score for each patch
        
        # Calculate sliding window steps
        y_steps = range(0, h - self.patch_size + 1, self.stride)
        x_steps = range(0, w - self.patch_size + 1, self.stride)
        
        from skimage import measure
        
        # Extract all possible patches with their cell counts
        for y in y_steps:
            for x in x_steps:
                patch_img = image[y:y + self.patch_size, x:x + self.patch_size]
                patch_mask = mask[y:y + self.patch_size, x:x + self.patch_size]
                
                # Count cells in patch
                labels = measure.label(patch_mask)
                cell_count = len(np.unique(labels)) - 1  # Subtract background
                
                # Calculate patch importance score
                cell_density = cell_count / (self.patch_size * self.patch_size)
                edge_distance = min(y, h-y, x, w-x) / max(h, w)  # Prefer central patches
                score = cell_density + 0.2 * edge_distance
                
                if cell_count > 0:  # Only keep patches with cells
                    scaled_img = cv2.resize(patch_img, (self.scaled_size, self.scaled_size))
                    scaled_mask = cv2.resize(patch_mask, (self.scaled_size, self.scaled_size))
                    
                    patches_img.append(scaled_img)
                    patches_mask.append(scaled_mask)
                    patch_scores.append(score)
        
        if len(patches_img) == 0:
            print(f"Warning: No cells found in image patches")
            # Fall back to grid sampling
            return self._grid_sample_patches(image, mask)
        
        # Select top-scoring patches
        if len(patches_img) > self.patches_per_image:
            indices = np.argsort(patch_scores)[-self.patches_per_image:]
            patches_img = [patches_img[i] for i in indices]
            patches_mask = [patches_mask[i] for i in indices]
        else:
            # If we have too few patches, duplicate high-scoring ones
            while len(patches_img) < self.patches_per_image:
                # Select from top half of patches
                idx = np.random.randint(len(patches_img) // 2, len(patches_img))
                patches_img.append(patches_img[idx].copy())
                patches_mask.append(patches_mask[idx].copy())
        
        return patches_img, patches_mask
    
    def _grid_sample_patches(self, image, mask):
        """Fallback grid sampling method"""
        h, w = image.shape[:2]
        patches_img = []
        patches_mask = []
        
        rows = np.linspace(0, h - self.patch_size, 4, dtype=int)
        cols = np.linspace(0, w - self.patch_size, 4, dtype=int)
        
        for y in rows:
            for x in cols:
                patch_img = image[y+y + self.patch_size, x:x + self.patch_size]
                patch_mask = mask[y:y + self.patch_size, x:x + self.patch_size]
                
                scaled_img = cv2.resize(patch_img, (self.scaled_size, self.scaled_size))
                scaled_mask = cv2.resize(patch_mask, (self.scaled_size, self.scaled_size))
                
                patches_img.append(scaled_img)
                patches_mask.append(scaled_mask)
        
        # Random sample if we have too many patches
        if len(patches_img) > self.patches_per_image:
            indices = np.random.choice(len(patches_img), self.patches_per_image, replace=False)
            patches_img = [patches_img[i] for i in indices]
            patches_mask = [patches_mask[i] for i in indices]
        
        return patches_img, patches_mask

    def __getitem__(self, idx):
        # Manage cache size
        if len(self.cache) > self.max_cache_size:
            self.cache.clear()
        
        # Use preloaded data
        image, mask = self.preloaded_data[idx]
        
        # Check cache first
        if idx in self.cache:
            return self.cache[idx]
        
        # Extract patches with guaranteed fixed size
        patches_img, patches_mask = self.extract_patches(image, mask)
        
        if len(patches_img) != self.patches_per_image:
            raise ValueError(f"Expected {self.patches_per_image} patches, got {len(patches_img)}")
        
        if self.transform:
            transformed_patches_img = []
            transformed_patches_mask = []
            for img, msk in zip(patches_img, patches_mask):
                img = img.astype(np.uint8)
                msk = msk.astype(np.uint8)
                transformed = self.transform(image=img, mask=msk)
                transformed_patches_img.append(transformed['image'])
                transformed_patches_mask.append(transformed['mask'])
            
            # Stack into tensors
            transformed_patches_img = torch.stack(transformed_patches_img)
            transformed_patches_mask = torch.stack(transformed_patches_mask).float()
            
            if len(transformed_patches_mask.shape) == 3:
                transformed_patches_mask = transformed_patches_mask.unsqueeze(1)
            
            result = (transformed_patches_img, transformed_patches_mask)
        else:
            patches_img = torch.stack([torch.from_numpy(p).permute(2, 0, 1) for p in patches_img])
            patches_mask = torch.stack([torch.from_numpy(p) for p in patches_mask]).unsqueeze(1)
            result = (patches_img.float() / 255.0, patches_mask.float())
        
        # Cache result
        self.cache[idx] = result
        return result