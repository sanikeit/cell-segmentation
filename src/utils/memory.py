import os
import torch
import psutil
from typing import Tuple, Dict

def get_memory_info() -> Dict[str, int]:
    """Get system and GPU memory information"""
    system_memory = psutil.virtual_memory()
    
    if torch.backends.mps.is_available():
        # For M1/M2 Macs, limit to 70% of system memory
        max_memory = int(system_memory.total * 0.7)
        available_memory = int(system_memory.available * 0.7)
    else:
        max_memory = system_memory.total
        available_memory = system_memory.available
    
    return {
        'total': max_memory,
        'available': available_memory,
        'percent_used': system_memory.percent
    }

def calculate_safe_batch_size(image_size: Tuple[int, int], min_batch: int = 1, max_batch: int = 16) -> int:
    """
    Calculate safe batch size based on available memory.
    
    Args:
        image_size: Tuple of (height, width)
        min_batch: Minimum batch size to return
        max_batch: Maximum batch size to consider
    
    Returns:
        Safe batch size to use
    """
    mem_info = get_memory_info()
    available_gb = mem_info['available'] / (1024 ** 3)
    
    # Calculate memory needed per sample (in GB)
    h, w = image_size
    bytes_per_sample = h * w * 3 * 4  # RGB image, float32
    bytes_per_sample *= 4  # Account for gradients, optimizer states
    sample_size_gb = bytes_per_sample / (1024 ** 3)
    
    # Calculate safe batch size with 20% buffer
    safe_batch = int((available_gb * 0.8) / sample_size_gb)
    
    # Clamp between min and max batch size
    return max(min_batch, min(safe_batch, max_batch))

def setup_memory_efficient_training() -> None:
    """Configure environment for memory-efficient training"""
    # Set MPS memory efficiency options for M1/M2 Macs
    if torch.backends.mps.is_available():
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.7'
        os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = '0.5'
    
    # General memory efficiency settings
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True

def get_suggested_training_params() -> Dict[str, int]:
    """Get suggested training parameters based on available memory"""
    mem_info = get_memory_info()
    available_gb = mem_info['available'] / (1024 ** 3)
    
    if available_gb < 8:  # Low memory
        return {
            'batch_size': 4,
            'num_workers': 1,
            'prefetch_factor': 1,
            'patch_size': 128
        }
    elif available_gb < 16:  # Medium memory
        return {
            'batch_size': 8,
            'num_workers': 2,
            'prefetch_factor': 2,
            'patch_size': 256
        }
    else:  # High memory
        return {
            'batch_size': 16,
            'num_workers': 4,
            'prefetch_factor': 2,
            'patch_size': 512
        }

def print_memory_stats() -> None:
    """Print current memory usage statistics"""
    mem_info = get_memory_info()
    print("\nMemory Statistics:")
    print(f"Total Memory: {mem_info['total'] / (1024**3):.1f} GB")
    print(f"Available Memory: {mem_info['available'] / (1024**3):.1f} GB")
    print(f"Memory Usage: {mem_info['percent_used']}%")
    
    if torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon) backend")

def get_available_memory():
    """Get available system memory in bytes"""
    return psutil.virtual_memory().available

def calculate_safe_batch_size(patch_dims, target_memory_usage=0.8):
    """Calculate safe batch size based on available memory"""
    available_memory = get_available_memory()
    # Assume 4 bytes per float32 value
    patch_memory = 4 * patch_dims[0] * patch_dims[1] * 3  # 3 channels
    safe_batch_size = int((available_memory * target_memory_usage) / patch_memory)
    return max(1, min(safe_batch_size, 32))  # Cap at 32

def get_suggested_training_params():
    """Get suggested training parameters based on system resources"""
    return {
        'batch_size': calculate_safe_batch_size((128, 128)),
        'num_workers': min(4, os.cpu_count() or 1),
        'pin_memory': torch.cuda.is_available(),
    }
