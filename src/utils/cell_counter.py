import numpy as np
import cv2
from skimage import measure, morphology
from typing import Union, Tuple, List

class CellCounter:
    def __init__(self, min_cell_size: int = 50, min_cell_intensity: float = 0.5, scale_factor: float = 1.0):
        self.min_cell_size = min_cell_size
        self.min_cell_intensity = min_cell_intensity
        self.scale_factor = scale_factor
        # Adjust min_cell_size based on scale
        self.scaled_min_size = int(self.min_cell_size * (self.scale_factor ** 2))
    
    def preprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """Preprocess mask for better cell detection"""
        # Ensure binary mask
        if mask.dtype == np.float32 or mask.dtype == np.float64:
            binary = (mask > self.min_cell_intensity).astype(np.uint8)
        else:
            binary = (mask > 127).astype(np.uint8)
        
        # Apply morphological operations to clean mask
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return binary

    def get_cell_centroids(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        """Get centroids of detected cells"""
        binary = self.preprocess_mask(mask)
        
        # Label connected components
        labels = measure.label(binary)
        props = measure.regionprops(labels)
        
        # Filter by size and get centroids
        centroids = []
        for prop in props:
            if prop.area >= self.min_cell_size:
                centroids.append(tuple(map(int, prop.centroid)))
        
        return centroids

    def count_cells(self, mask: np.ndarray) -> Tuple[int, List[Tuple[int, int]]]:
        """Count cells with scale-aware size filtering"""
        try:
            binary = self.preprocess_mask(mask)
            labels = measure.label(binary)
            props = measure.regionprops(labels)
            
            # Filter regions by scaled size
            valid_cells = [prop for prop in props if prop.area >= self.scaled_min_size]
            
            # Scale up the count for patches
            if self.scale_factor < 1.0:
                count_multiplier = 1 / (self.scale_factor ** 2)
            else:
                count_multiplier = 1.0
                
            cell_count = int(len(valid_cells) * count_multiplier)
            centroids = [tuple(map(int, prop.centroid)) for prop in valid_cells]
            
            return cell_count, centroids
        except Exception as e:
            print(f"Error in cell counting: {e}")
            return 0, []

    def visualize_cells(self, image: np.ndarray, mask: np.ndarray, 
                       save_path: str = None) -> np.ndarray:
        """
        Visualize detected cells on original image
        
        Args:
            image: Original RGB image
            mask: Binary or probability mask
            save_path: Optional path to save visualization
            
        Returns:
            Visualization image with cell markers
        """
        count, centroids = self.count_cells(mask)
        
        # Create visualization
        vis_img = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Draw cell markers
        for centroid in centroids:
            cv2.drawMarker(vis_img, (centroid[1], centroid[0]), 
                          color=(0, 255, 0), markerType=cv2.MARKER_CROSS,
                          markerSize=10, thickness=2)
        
        # Add cell count text
        cv2.putText(vis_img, f"Cells: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
        
        return vis_img

def count_cells(mask: np.ndarray, min_size: int = 50) -> int:
    """
    Legacy function for backward compatibility
    """
    counter = CellCounter(min_cell_size=min_size)
    count, _ = counter.count_cells(mask)
    return count