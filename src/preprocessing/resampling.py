"""
Resampling Module
重采样模块

Resamples CT volumes to unified spacing.
"""

import numpy as np
from typing import Tuple, Union, Optional
from scipy import ndimage
import torch


class Resampler:
    """Volume resampler for standardizing spacing.
    
    Resamples CT volumes to target spacing (default: 1x1x1 mm³).
    """
    
    def __init__(
        self,
        target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        order: int = 3,  # Cubic interpolation
        mode: str = 'constant',
        cval: float = -1024.0,  # Air HU value for padding
    ):
        """Initialize resampler.
        
        Args:
            target_spacing: Target voxel spacing in mm (z, y, x)
            order: Interpolation order (0=nearest, 1=linear, 3=cubic)
            mode: Padding mode for boundary
            cval: Constant value for 'constant' mode
        """
        self.target_spacing = target_spacing
        self.order = order
        self.mode = mode
        self.cval = cval
        
    def __call__(
        self,
        image: np.ndarray,
        current_spacing: Tuple[float, float, float],
        mask: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Resample image to target spacing.
        
        Args:
            image: Input volume (D, H, W) or (C, D, H, W)
            current_spacing: Current voxel spacing in mm
            mask: Optional segmentation mask to resample
            
        Returns:
            Resampled image (and mask if provided)
        """
        # Calculate resize factor
        resize_factor = np.array(current_spacing) / np.array(self.target_spacing)
        
        # Calculate new shape
        new_shape = np.round(np.array(image.shape[-3:]) * resize_factor).astype(int)
        
        # Handle multi-channel
        if image.ndim == 4:
            resampled = np.zeros((image.shape[0],) + tuple(new_shape), dtype=image.dtype)
            for c in range(image.shape[0]):
                resampled[c] = ndimage.zoom(
                    image[c], resize_factor, order=self.order,
                    mode=self.mode, cval=self.cval
                )
        else:
            resampled = ndimage.zoom(
                image, resize_factor, order=self.order,
                mode=self.mode, cval=self.cval
            )
            
        if mask is not None:
            # Use nearest neighbor for mask
            mask_resampled = ndimage.zoom(
                mask.astype(np.float32), resize_factor, order=0,
                mode='constant', cval=0
            ).astype(mask.dtype)
            return resampled, mask_resampled
            
        return resampled
    
    def get_new_shape(
        self,
        current_shape: Tuple[int, ...],
        current_spacing: Tuple[float, float, float],
    ) -> Tuple[int, ...]:
        """Calculate new shape after resampling."""
        resize_factor = np.array(current_spacing) / np.array(self.target_spacing)
        return tuple(np.round(np.array(current_shape[-3:]) * resize_factor).astype(int))


def resample_volume(
    image: np.ndarray,
    current_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    order: int = 3,
) -> np.ndarray:
    """Convenience function for resampling.
    
    Args:
        image: Input volume
        current_spacing: Current spacing in mm
        target_spacing: Target spacing in mm
        order: Interpolation order
        
    Returns:
        Resampled volume
    """
    resampler = Resampler(target_spacing=target_spacing, order=order)
    return resampler(image, current_spacing)


class CropOrPad:
    """Crop or pad volume to target size."""
    
    def __init__(
        self,
        target_size: Tuple[int, int, int],
        pad_value: float = 0.0,
        center_crop: bool = True,
    ):
        """Initialize crop/pad.
        
        Args:
            target_size: Target volume size (D, H, W)
            pad_value: Value for padding
            center_crop: Whether to crop from center
        """
        self.target_size = target_size
        self.pad_value = pad_value
        self.center_crop = center_crop
        
    def __call__(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Crop or pad image to target size.
        
        Args:
            image: Input volume
            mask: Optional mask
            
        Returns:
            Processed image (and mask if provided)
        """
        current_shape = image.shape[-3:]
        target_shape = self.target_size
        
        # Calculate padding/cropping for each dimension
        result = image.copy()
        
        for i in range(3):
            diff = target_shape[i] - current_shape[i]
            
            if diff > 0:
                # Pad
                pad_before = diff // 2
                pad_after = diff - pad_before
                pad_width = [(0, 0)] * (result.ndim - 3) + [(0, 0)] * i + [(pad_before, pad_after)] + [(0, 0)] * (2 - i)
                result = np.pad(result, pad_width, mode='constant', constant_values=self.pad_value)
            elif diff < 0:
                # Crop
                if self.center_crop:
                    start = (-diff) // 2
                else:
                    start = 0
                end = start + target_shape[i]
                
                slices = [slice(None)] * (result.ndim - 3 + i) + [slice(start, end)] + [slice(None)] * (2 - i)
                result = result[tuple(slices)]
                
        if mask is not None:
            mask_result = self(mask)[0] if isinstance(self(mask), tuple) else self(mask)
            return result, mask_result
            
        return result


if __name__ == "__main__":
    # Test resampling
    test_vol = np.random.randn(100, 256, 256).astype(np.float32)
    current_spacing = (2.0, 0.8, 0.8)  # Typical CT spacing
    
    resampler = Resampler(target_spacing=(1.0, 1.0, 1.0))
    resampled = resampler(test_vol, current_spacing)
    
    print(f"Original shape: {test_vol.shape}")
    print(f"Resampled shape: {resampled.shape}")
    
    # Test crop/pad
    cropper = CropOrPad(target_size=(128, 128, 128))
    cropped = cropper(resampled)
    print(f"After crop/pad: {cropped.shape}")
