import torch

# Clear MPS cache
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
