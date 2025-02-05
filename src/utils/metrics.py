import torch
import numpy as np
from typing import Union, Tuple

def calculate_dice_coefficient(
    pred: Union[torch.Tensor, np.ndarray], 
    target: Union[torch.Tensor, np.ndarray], 
    smooth: float = 1e-7
) -> float:
    """
    Calculate Dice coefficient between predicted and target masks
    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask
        smooth: Smoothing factor to avoid division by zero
    Returns:
        dice_coeff: Dice coefficient value
    """
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
        
    pred = pred.float()
    target = target.float()
    
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred shape {pred.shape} != target shape {target.shape}")
        
    pred = pred.flatten()
    target = target.flatten()
    
    intersection = (pred * target).sum()
    dice_coeff = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return float(dice_coeff)

def calculate_iou(
    pred: Union[torch.Tensor, np.ndarray], 
    target: Union[torch.Tensor, np.ndarray], 
    smooth: float = 1e-7
) -> float:
    """
    Calculate IoU (Intersection over Union) between predicted and target masks
    """
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
        
    pred = pred.float()
    target = target.float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return float(iou)

def calculate_pixel_accuracy(
    pred: Union[torch.Tensor, np.ndarray], 
    target: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Calculate pixel-wise accuracy between predicted and target masks
    """
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
        
    pred = pred.float()
    target = target.float()
    
    correct = (pred == target).sum()
    total = torch.numel(pred)
    
    return float(correct) / total

def evaluate_segmentation(
    pred: Union[torch.Tensor, np.ndarray], 
    target: Union[torch.Tensor, np.ndarray]
) -> Tuple[float, float, float]:
    """
    Evaluate segmentation using multiple metrics
    Returns:
        Tuple of (dice_coefficient, iou, pixel_accuracy)
    """
    dice = calculate_dice_coefficient(pred, target)
    iou = calculate_iou(pred, target)
    accuracy = calculate_pixel_accuracy(pred, target)
    
    return dice, iou, accuracy
