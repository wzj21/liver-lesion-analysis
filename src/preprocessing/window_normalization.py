"""
Window Normalization Module
窗宽窗位标准化模块

Standardizes CT Hounsfield Units to liver window for consistent input.
"""

import numpy as np
from typing import Tuple, Optional, Union
import torch


class WindowNormalization:
    """Window normalization for CT images.
    
    Applies standard liver window (WW=160, WL=60) to normalize HU values.
    """
    
    # Standard liver window parameters
    DEFAULT_LIVER_WINDOW = {
        'window_width': 160,
        'window_level': 60,
        'hu_min': -20,   # WL - WW/2
        'hu_max': 140,   # WL + WW/2
    }
    
    # Alternative windows for multi-window fusion
    SOFT_TISSUE_WINDOW = {'window_width': 350, 'window_level': 50}
    PORTAL_VENOUS_WINDOW = {'window_width': 150, 'window_level': 80}
    
    def __init__(
        self,
        window_width: int = 160,
        window_level: int = 60,
        output_range: Tuple[float, float] = (0.0, 1.0),
    ):
        """Initialize window normalization.
        
        Args:
            window_width: Window width in HU
            window_level: Window level (center) in HU
            output_range: Output value range after normalization
        """
        self.window_width = window_width
        self.window_level = window_level
        self.output_range = output_range
        
        # Calculate HU range
        self.hu_min = window_level - window_width / 2
        self.hu_max = window_level + window_width / 2
        
    def __call__(
        self, 
        image: Union[np.ndarray, torch.Tensor],
        return_tensor: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Apply window normalization.
        
        Args:
            image: Input CT image in HU (numpy array or torch tensor)
            return_tensor: Whether to return torch tensor
            
        Returns:
            Normalized image in output_range
        """
        is_tensor = isinstance(image, torch.Tensor)
        
        if is_tensor:
            device = image.device
            dtype = image.dtype
            image_np = image.cpu().numpy()
        else:
            image_np = image.copy()
            
        # Clip to window range
        image_np = np.clip(image_np, self.hu_min, self.hu_max)
        
        # Normalize to output range
        out_min, out_max = self.output_range
        image_np = (image_np - self.hu_min) / (self.hu_max - self.hu_min)
        image_np = image_np * (out_max - out_min) + out_min
        
        if return_tensor or is_tensor:
            result = torch.from_numpy(image_np)
            if is_tensor:
                result = result.to(device=device, dtype=dtype)
            return result
            
        return image_np
    
    def inverse(
        self,
        image: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """Inverse window normalization (back to HU).
        
        Args:
            image: Normalized image
            
        Returns:
            Image in HU values
        """
        out_min, out_max = self.output_range
        
        # Inverse normalization
        image = (image - out_min) / (out_max - out_min)
        image = image * (self.hu_max - self.hu_min) + self.hu_min
        
        return image


def apply_liver_window(
    image: Union[np.ndarray, torch.Tensor],
    window_width: int = 160,
    window_level: int = 60,
) -> Union[np.ndarray, torch.Tensor]:
    """Convenience function to apply liver window.
    
    Args:
        image: CT image in HU
        window_width: Window width (default: 160)
        window_level: Window level (default: 60)
        
    Returns:
        Normalized image in [0, 1]
    """
    normalizer = WindowNormalization(window_width, window_level)
    return normalizer(image)


class MultiWindowNormalization:
    """Multi-window normalization for CT images.
    
    Combines multiple windows into multi-channel input.
    """
    
    PRESET_WINDOWS = {
        'liver': {'window_width': 160, 'window_level': 60},
        'soft_tissue': {'window_width': 350, 'window_level': 50},
        'portal_venous': {'window_width': 150, 'window_level': 80},
        'bone': {'window_width': 2000, 'window_level': 300},
        'lung': {'window_width': 1500, 'window_level': -600},
    }
    
    def __init__(
        self,
        windows: Optional[list] = None,
        fusion_method: str = 'channel_concat',
    ):
        """Initialize multi-window normalization.
        
        Args:
            windows: List of window names or dicts with 'window_width' and 'window_level'
            fusion_method: How to combine windows ('channel_concat' or 'mean')
        """
        if windows is None:
            windows = ['liver', 'soft_tissue', 'portal_venous']
            
        self.normalizers = []
        for w in windows:
            if isinstance(w, str):
                params = self.PRESET_WINDOWS[w]
            else:
                params = w
            self.normalizers.append(WindowNormalization(**params))
            
        self.fusion_method = fusion_method
        
    def __call__(
        self,
        image: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """Apply multi-window normalization.
        
        Args:
            image: Input CT image in HU
            
        Returns:
            Multi-channel normalized image
        """
        normalized = [norm(image) for norm in self.normalizers]
        
        if isinstance(image, torch.Tensor):
            if self.fusion_method == 'channel_concat':
                return torch.stack(normalized, dim=0)
            else:
                return torch.stack(normalized, dim=0).mean(dim=0)
        else:
            if self.fusion_method == 'channel_concat':
                return np.stack(normalized, axis=0)
            else:
                return np.mean(normalized, axis=0)


if __name__ == "__main__":
    # Test window normalization
    import numpy as np
    
    # Create test data (simulated CT in HU)
    test_ct = np.random.uniform(-100, 200, (64, 64, 64)).astype(np.float32)
    
    # Apply liver window
    normalizer = WindowNormalization()
    normalized = normalizer(test_ct)
    
    print(f"Input range: [{test_ct.min():.1f}, {test_ct.max():.1f}]")
    print(f"Output range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    
    # Test multi-window
    multi_norm = MultiWindowNormalization()
    multi_result = multi_norm(test_ct)
    print(f"Multi-window shape: {multi_result.shape}")
