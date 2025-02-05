import os
from pathlib import Path
from src.utils.memory import calculate_safe_batch_size, get_suggested_training_params  # Changed from relative to absolute import

class Config:
    def __init__(self):
        # Base directories
        self.PROJECT_ROOT = Path(__file__).parent.parent
        self.MONUSEG_DIR = Path('/Users/sanikeit/Documents/cell-segmentation/MoNuSegDataset')
        
        # Data directories
        self.TISSUE_IMAGES_DIR = self.MONUSEG_DIR / 'Tissue Images'
        self.ANNOTATIONS_DIR = self.MONUSEG_DIR / 'Annotations'
        
        # Output directories
        self.OUTPUT_DIR = self.PROJECT_ROOT / 'outputs'
        self.LOG_DIR = self.OUTPUT_DIR / 'logs'
        self.CHECKPOINT_DIR = self.OUTPUT_DIR / 'checkpoints'
        self.VIS_DIR = self.OUTPUT_DIR / 'visualizations'
        
        # Get suggested parameters based on available memory
        suggested_params = get_suggested_training_params()
        
        # Optimized training parameters for speed
        self.PATCH_SIZE = 128
        self.EPOCHS = 200
        self.LEARNING_RATE = 1e-3
        self.SCALE_FACTOR = 0.5
        self.TRAIN_SPLIT = 0.85
        
        # Aggressive performance optimization
        self.BATCH_SIZE = 16
        self.NUM_WORKERS = 0  # No workers for M1 Mac
        self.PIN_MEMORY = False
        self.PREFETCH_FACTOR = 1
        self.PERSISTENT_WORKERS = False
        
        # Reduced data loading overhead
        self.PATCHES_PER_IMAGE = 16  # Reduced for better quality patches
        self.CACHE_SIZE = 100  # Limit cache size
        
        # Reduced logging and checkpointing
        self.LOG_INTERVAL = 20
        self.CHECKPOINT_FREQ = 25
        self.VIS_FREQ = 25
        
        # Rest of parameters
        self.SAVE_TOP_K = 5
        
        # Model parameters
        self.UNET_INPUT_CHANNELS = 3
        self.UNET_OUTPUT_CHANNELS = 1
        
        # Performance settings
        self.USE_MIXED_PRECISION = True
        
        # Optimization settings
        self.WEIGHT_DECAY = 0.01
        self.GRADIENT_CLIP = 1.0
        self.GRADIENT_CLIP_VAL = 1.0
        self.WARMUP_EPOCHS = 5
        
        # Data settings
        self.AUGMENTATION_STRATEGY = 'aggressive'
        self.PATIENCE = 15
        self.MIN_DELTA = 1e-4
        
        # Create and verify directories
        self._create_directories()
        self._verify_data_dirs()
    
    def _create_directories(self):
        """Create necessary output directories"""
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.VIS_DIR, exist_ok=True)
    
    def _verify_data_dirs(self):
        """Verify that data directories exist"""
        if not self.TISSUE_IMAGES_DIR.exists():
            raise FileNotFoundError(
                f"Tissue images directory not found at {self.TISSUE_IMAGES_DIR}."
            )
        if not self.ANNOTATIONS_DIR.exists():
            raise FileNotFoundError(
                f"Annotations directory not found at {self.ANNOTATIONS_DIR}."
            )
    
    def update_from_args(self, args):
        """Update config from command line arguments"""
        if args.epochs:
            self.EPOCHS = args.epochs
        if args.batch_size:
            # Ensure batch size doesn't exceed memory limits
            patch_dims = (self.PATCH_SIZE, self.PATCH_SIZE)
            max_safe_batch = calculate_safe_batch_size(patch_dims)
            self.BATCH_SIZE = min(args.batch_size, max_safe_batch)
            if self.BATCH_SIZE != args.batch_size:
                print(f"Warning: Requested batch size {args.batch_size} exceeded memory limits. Using {self.BATCH_SIZE} instead.")
        if args.aug_strategy:
            self.AUGMENTATION_STRATEGY = args.aug_strategy
        
        # Update dependent parameters
        self.GRADIENT_ACCUMULATION_STEPS = max(1, 16 // self.BATCH_SIZE)