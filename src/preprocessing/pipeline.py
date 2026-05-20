"""
Preprocessing Pipeline
预处理流水线

Complete preprocessing pipeline for CT scans.
"""

import numpy as np
from typing import Dict, Optional, Tuple, Union
from pathlib import Path
import logging

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False

from .window_normalization import WindowNormalization
from .resampling import Resampler, CropOrPad

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """Complete preprocessing pipeline for CT scans.
    
    Pipeline steps:
    1. Load DICOM/NIfTI
    2. Window normalization (liver window: WW=160, WL=60)
    3. Resample to unified spacing (1x1x1 mm³)
    4. Crop/pad to target size
    5. Quality check
    """
    
    def __init__(
        self,
        window_width: int = 160,
        window_level: int = 60,
        target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        target_size: Optional[Tuple[int, int, int]] = None,
        normalize_output: bool = True,
    ):
        """Initialize preprocessing pipeline.
        
        Args:
            window_width: CT window width
            window_level: CT window level
            target_spacing: Target voxel spacing in mm
            target_size: Target volume size (optional)
            normalize_output: Whether to normalize output to [0, 1]
        """
        self.window_norm = WindowNormalization(
            window_width=window_width,
            window_level=window_level,
        )
        self.resampler = Resampler(target_spacing=target_spacing)
        self.target_size = target_size
        if target_size is not None:
            self.cropper = CropOrPad(target_size=target_size)
        else:
            self.cropper = None
        self.normalize_output = normalize_output
        
    def load_image(
        self,
        path: Union[str, Path],
    ) -> Tuple[np.ndarray, Dict]:
        """Load CT image from file.
        
        Args:
            path: Path to NIfTI or DICOM file/folder
            
        Returns:
            Tuple of (image array in HU, metadata dict)
        """
        path = Path(path)
        
        if path.suffix in ['.nii', '.gz'] or path.name.endswith('.nii.gz'):
            return self._load_nifti(path)
        elif path.is_dir() or path.suffix == '.dcm':
            return self._load_dicom(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
            
    def _load_nifti(self, path: Path) -> Tuple[np.ndarray, Dict]:
        """Load NIfTI file."""
        if not HAS_NIBABEL:
            raise ImportError("nibabel is required for NIfTI loading")
            
        nii = nib.load(str(path))
        image = nii.get_fdata().astype(np.float32)
        
        # Get spacing from header
        spacing = tuple(nii.header.get_zooms()[:3])
        
        metadata = {
            'spacing': spacing,
            'shape': image.shape,
            'affine': nii.affine,
            'path': str(path),
        }
        
        return image, metadata
        
    def _load_dicom(self, path: Path) -> Tuple[np.ndarray, Dict]:
        """Load DICOM series."""
        if not HAS_SITK:
            raise ImportError("SimpleITK is required for DICOM loading")
            
        if path.is_file():
            # Single DICOM file
            reader = sitk.ImageFileReader()
            reader.SetFileName(str(path))
        else:
            # DICOM series
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(str(path))
            reader.SetFileNames(dicom_names)
            
        image_sitk = reader.Execute()
        image = sitk.GetArrayFromImage(image_sitk).astype(np.float32)
        
        # Get spacing (SimpleITK returns (x, y, z), we want (z, y, x))
        spacing_xyz = image_sitk.GetSpacing()
        spacing = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
        
        metadata = {
            'spacing': spacing,
            'shape': image.shape,
            'origin': image_sitk.GetOrigin(),
            'direction': image_sitk.GetDirection(),
            'path': str(path),
        }
        
        return image, metadata
        
    def __call__(
        self,
        image: Union[np.ndarray, str, Path],
        spacing: Optional[Tuple[float, float, float]] = None,
        mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Union[np.ndarray, Dict]]:
        """Run preprocessing pipeline.
        
        Args:
            image: CT image array (in HU) or path to file
            spacing: Current voxel spacing (required if image is array)
            mask: Optional segmentation mask
            
        Returns:
            Dictionary with:
            - 'image': Preprocessed image
            - 'mask': Preprocessed mask (if provided)
            - 'metadata': Processing metadata
        """
        # Load if path provided
        if isinstance(image, (str, Path)):
            image, file_metadata = self.load_image(image)
            spacing = file_metadata['spacing']
        else:
            file_metadata = {}
            
        if spacing is None:
            raise ValueError("spacing must be provided for array input")
            
        original_shape = image.shape
        
        # Step 1: Window normalization
        image = self.window_norm(image)
        
        # Step 2: Resample
        if mask is not None:
            image, mask = self.resampler(image, spacing, mask)
        else:
            image = self.resampler(image, spacing)
            
        resampled_shape = image.shape
        
        # Step 3: Crop/pad to target size
        if self.cropper is not None:
            if mask is not None:
                image, mask = self.cropper(image, mask)
            else:
                image = self.cropper(image)
                
        # Quality check
        quality_info = self._quality_check(image)
        
        result = {
            'image': image,
            'metadata': {
                'original_shape': original_shape,
                'resampled_shape': resampled_shape,
                'final_shape': image.shape,
                'spacing': spacing,
                'target_spacing': self.resampler.target_spacing,
                'quality': quality_info,
                **file_metadata,
            }
        }
        
        if mask is not None:
            result['mask'] = mask
            
        return result
        
    def _quality_check(self, image: np.ndarray) -> Dict:
        """Perform quality checks on preprocessed image."""
        return {
            'min_value': float(image.min()),
            'max_value': float(image.max()),
            'mean_value': float(image.mean()),
            'std_value': float(image.std()),
            'nan_count': int(np.isnan(image).sum()),
            'inf_count': int(np.isinf(image).sum()),
            'valid': bool(not np.isnan(image).any() and not np.isinf(image).any()),
        }
        
    def save(
        self,
        image: np.ndarray,
        path: Union[str, Path],
        spacing: Optional[Tuple[float, float, float]] = None,
    ):
        """Save preprocessed image.
        
        Args:
            image: Image to save
            path: Output path
            spacing: Voxel spacing for header
        """
        path = Path(path)
        
        if spacing is None:
            spacing = self.resampler.target_spacing
            
        if path.suffix in ['.nii', '.gz'] or path.name.endswith('.nii.gz'):
            if not HAS_NIBABEL:
                raise ImportError("nibabel is required for NIfTI saving")
            affine = np.diag([*spacing, 1.0])
            nii = nib.Nifti1Image(image, affine)
            nib.save(nii, str(path))
        else:
            raise ValueError(f"Unsupported output format: {path.suffix}")


if __name__ == "__main__":
    # Test pipeline
    pipeline = PreprocessingPipeline(
        window_width=160,
        window_level=60,
        target_spacing=(1.0, 1.0, 1.0),
        target_size=(128, 128, 128),
    )
    
    # Create dummy CT data
    test_image = np.random.uniform(-100, 200, (100, 256, 256)).astype(np.float32)
    test_spacing = (2.0, 0.8, 0.8)
    
    result = pipeline(test_image, spacing=test_spacing)
    
    print("Preprocessing results:")
    print(f"  Output shape: {result['image'].shape}")
    print(f"  Value range: [{result['image'].min():.3f}, {result['image'].max():.3f}]")
    print(f"  Quality check: {result['metadata']['quality']}")
