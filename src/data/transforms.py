import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

class CellSegmentationTransforms:
    """Factory class for cell segmentation data augmentations"""
    
    @staticmethod
    def get_transforms(split="train", strategy="baseline"):
        """
        Get transforms for different splits and strategies
        
        Args:
            split (str): 'train' or 'val'
            strategy (str): 'baseline', 'aggressive', or 'minimal'
        """
        if split == "train":
            if strategy == "baseline":
                return A.Compose([
                    A.RandomRotate90(p=0.5),
                    A.HorizontalFlip(p=0.5),  # Changed from Flip to HorizontalFlip
                    A.VerticalFlip(p=0.5),    # Added VerticalFlip separately
                    A.ShiftScaleRotate(
                        shift_limit=0.0625, 
                        scale_limit=0.1, 
                        rotate_limit=45, 
                        p=0.5
                    ),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ])
            
            elif strategy == "aggressive":
                return A.Compose([
                    A.OneOf([
                        A.RandomRotate90(p=1.0),
                        A.Rotate(limit=45, p=1.0),
                    ], p=0.8),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.OneOf([
                        A.ElasticTransform(
                            alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03,
                            border_mode=cv2.BORDER_CONSTANT, p=0.7
                        ),
                        A.GridDistortion(distort_limit=0.3, border_mode=cv2.BORDER_CONSTANT, p=0.7),
                        A.OpticalDistortion(distort_limit=0.3, shift_limit=0.3, p=0.7),
                    ], p=0.7),
                    A.OneOf([
                        A.GaussNoise(var_limit=(10.0, 50.0), p=0.7),
                        A.ISONoise(p=0.7),
                        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
                        A.RandomGamma(gamma_limit=(80, 120), p=0.7),
                    ], p=0.7),
                    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ])
            
            else:  # minimal
                return A.Compose([
                    A.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ])
                
        else:  # validation transforms
            return A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

    @staticmethod
    def get_post_transforms():
        """Get post-processing transforms for prediction"""
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    @staticmethod
    def get_tta_transforms():
        """Test-time augmentation transforms"""
        return [
            A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]),
            A.Compose([
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]),
            A.Compose([
                A.VerticalFlip(p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]),
            A.Compose([
                A.Rotate(limit=(90, 90), p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]),
        ]