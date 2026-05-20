"""
半监督学习数据增强模块
Data Augmentation for Semi-Supervised Learning

包含:
- 弱增强 (Weak Augmentation)
- 强增强 (Strong Augmentation)
- Copy-Paste增强
- 3D医学图像专用增强
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from scipy import ndimage
from scipy.ndimage import gaussian_filter, map_coordinates
import random


class WeakAugmentation3D:
    """3D弱增强 - 用于Teacher模型生成伪标签"""
    
    def __init__(
        self,
        flip_prob: float = 0.5,
        rotation_range: Tuple[float, float] = (-5, 5),
        scale_range: Tuple[float, float] = (0.95, 1.05)
    ):
        self.flip_prob = flip_prob
        self.rotation_range = rotation_range
        self.scale_range = scale_range
    
    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        is_tensor = isinstance(image, torch.Tensor)
        if is_tensor:
            device = image.device
            image = image.cpu().numpy()
            if mask is not None:
                mask = mask.cpu().numpy()
        
        # 随机翻转
        if random.random() < self.flip_prob:
            axis = random.choice([1, 2, 3] if image.ndim == 4 else [0, 1, 2])
            image = np.flip(image, axis=axis).copy()
            if mask is not None:
                mask = np.flip(mask, axis=axis).copy()
        
        # 随机轻微旋转
        angle = random.uniform(*self.rotation_range)
        if abs(angle) > 0.5:
            if image.ndim == 4:
                for i in range(image.shape[0]):
                    image[i] = ndimage.rotate(image[i], angle, axes=(1, 2), reshape=False, order=1)
            else:
                image = ndimage.rotate(image, angle, axes=(1, 2), reshape=False, order=1)
            if mask is not None:
                mask = ndimage.rotate(mask, angle, axes=(0, 1) if mask.ndim == 3 else (1, 2), reshape=False, order=0)
        
        result = {'image': image}
        if mask is not None:
            result['mask'] = mask
        
        if is_tensor:
            result = {k: torch.from_numpy(v).to(device) for k, v in result.items()}
        
        return result


class StrongAugmentation3D:
    """3D强增强 - 用于Student模型训练"""
    
    def __init__(
        self,
        flip_prob: float = 0.5,
        rotation_range: Tuple[float, float] = (-15, 15),
        scale_range: Tuple[float, float] = (0.8, 1.2),
        elastic_prob: float = 0.3,
        elastic_alpha: float = 100,
        elastic_sigma: float = 10,
        noise_prob: float = 0.3,
        noise_std: float = 0.02,
        contrast_range: Tuple[float, float] = (0.7, 1.3),
        gamma_range: Tuple[float, float] = (0.7, 1.3),
        cutout_prob: float = 0.3,
        cutout_ratio: float = 0.2
    ):
        self.flip_prob = flip_prob
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.elastic_prob = elastic_prob
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.noise_prob = noise_prob
        self.noise_std = noise_std
        self.contrast_range = contrast_range
        self.gamma_range = gamma_range
        self.cutout_prob = cutout_prob
        self.cutout_ratio = cutout_ratio
    
    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        is_tensor = isinstance(image, torch.Tensor)
        if is_tensor:
            device = image.device
            image = image.cpu().numpy()
            if mask is not None:
                mask = mask.cpu().numpy()
        
        # 几何变换
        for axis in ([1, 2, 3] if image.ndim == 4 else [0, 1, 2]):
            if random.random() < self.flip_prob:
                image = np.flip(image, axis=axis).copy()
                if mask is not None:
                    mask_axis = axis if mask.ndim == image.ndim else axis - 1
                    mask = np.flip(mask, axis=mask_axis).copy()
        
        # 随机旋转
        angle = random.uniform(*self.rotation_range)
        if abs(angle) > 1:
            if image.ndim == 4:
                for i in range(image.shape[0]):
                    image[i] = ndimage.rotate(image[i], angle, axes=(1, 2), reshape=False, order=1)
            else:
                image = ndimage.rotate(image, angle, axes=(1, 2), reshape=False, order=1)
            if mask is not None:
                mask = ndimage.rotate(mask, angle, axes=(0, 1) if mask.ndim == 3 else (1, 2), reshape=False, order=0)
        
        # 弹性变形
        if random.random() < self.elastic_prob:
            image, mask = self._elastic_deformation(image, mask)
        
        # 强度变换
        if random.random() < self.noise_prob:
            noise = np.random.normal(0, self.noise_std, image.shape)
            image = image + noise
        
        # 对比度调整
        contrast = random.uniform(*self.contrast_range)
        mean = image.mean()
        image = (image - mean) * contrast + mean
        
        # Gamma校正
        gamma = random.uniform(*self.gamma_range)
        image_min, image_max = image.min(), image.max()
        image_normalized = (image - image_min) / (image_max - image_min + 1e-8)
        image = np.power(image_normalized, gamma) * (image_max - image_min) + image_min
        
        # Cutout
        if random.random() < self.cutout_prob:
            image = self._cutout(image)
        
        image = np.clip(image, 0, 1)
        
        result = {'image': image}
        if mask is not None:
            result['mask'] = mask
        
        if is_tensor:
            result = {k: torch.from_numpy(v.copy()).to(device) for k, v in result.items()}
        
        return result
    
    def _elastic_deformation(self, image: np.ndarray, mask: Optional[np.ndarray] = None):
        shape = image.shape[-3:]
        dz = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.elastic_sigma) * self.elastic_alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.elastic_sigma) * self.elastic_alpha
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.elastic_sigma) * self.elastic_alpha
        
        z, y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
        indices = [np.reshape(z + dz, (-1, 1)), np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))]
        
        if image.ndim == 4:
            deformed_image = np.zeros_like(image)
            for i in range(image.shape[0]):
                deformed_image[i] = map_coordinates(image[i], indices, order=1).reshape(shape)
        else:
            deformed_image = map_coordinates(image, indices, order=1).reshape(shape)
        
        deformed_mask = None
        if mask is not None:
            deformed_mask = map_coordinates(mask, indices, order=0).reshape(shape)
        
        return deformed_image, deformed_mask
    
    def _cutout(self, image: np.ndarray) -> np.ndarray:
        shape = image.shape[-3:]
        cut_d = int(shape[0] * self.cutout_ratio)
        cut_h = int(shape[1] * self.cutout_ratio)
        cut_w = int(shape[2] * self.cutout_ratio)
        
        z = random.randint(0, shape[0] - cut_d)
        y = random.randint(0, shape[1] - cut_h)
        x = random.randint(0, shape[2] - cut_w)
        
        if image.ndim == 4:
            image[:, z:z+cut_d, y:y+cut_h, x:x+cut_w] = 0
        else:
            image[z:z+cut_d, y:y+cut_h, x:x+cut_w] = 0
        
        return image


class CopyPasteAugmentation:
    """Copy-Paste增强 - 从其他样本复制病灶并粘贴"""
    
    def __init__(
        self,
        lesion_bank: List[Dict] = None,
        paste_prob: float = 0.5,
        max_paste: int = 3,
        intensity_matching: bool = True,
        boundary_blending: bool = True,
        blend_radius: int = 3
    ):
        self.lesion_bank = lesion_bank or []
        self.paste_prob = paste_prob
        self.max_paste = max_paste
        self.intensity_matching = intensity_matching
        self.boundary_blending = boundary_blending
        self.blend_radius = blend_radius
    
    def add_to_bank(self, image: np.ndarray, mask: np.ndarray, lesion_mask: np.ndarray):
        """将病灶添加到病灶库"""
        coords = np.where(lesion_mask > 0)
        if len(coords[0]) == 0:
            return
        
        z_min, z_max = coords[0].min(), coords[0].max()
        y_min, y_max = coords[1].min(), coords[1].max()
        x_min, x_max = coords[2].min(), coords[2].max()
        
        margin = 5
        z_min, z_max = max(0, z_min - margin), min(image.shape[-3], z_max + margin)
        y_min, y_max = max(0, y_min - margin), min(image.shape[-2], y_max + margin)
        x_min, x_max = max(0, x_min - margin), min(image.shape[-1], x_max + margin)
        
        lesion_image = image[..., z_min:z_max, y_min:y_max, x_min:x_max].copy()
        lesion_crop = lesion_mask[z_min:z_max, y_min:y_max, x_min:x_max].copy()
        
        self.lesion_bank.append({
            'image': lesion_image,
            'mask': lesion_crop,
            'bbox': (z_max - z_min, y_max - y_min, x_max - x_min)
        })
    
    def __call__(self, image: np.ndarray, liver_mask: np.ndarray, lesion_mask: Optional[np.ndarray] = None) -> Dict:
        if len(self.lesion_bank) == 0 or random.random() > self.paste_prob:
            result = {'image': image, 'liver_mask': liver_mask}
            if lesion_mask is not None:
                result['lesion_mask'] = lesion_mask
            return result
        
        image = image.copy()
        lesion_mask = lesion_mask.copy() if lesion_mask is not None else np.zeros(liver_mask.shape, dtype=np.uint8)
        
        num_paste = random.randint(1, self.max_paste)
        
        for _ in range(num_paste):
            lesion_data = random.choice(self.lesion_bank)
            lesion_img = lesion_data['image']
            lesion_msk = lesion_data['mask']
            bbox = lesion_data['bbox']
            
            paste_pos = self._find_paste_position(liver_mask, lesion_mask, bbox)
            
            if paste_pos is None:
                continue
            
            z, y, x = paste_pos
            self._paste_lesion(image, lesion_mask, lesion_img, lesion_msk, z, y, x)
        
        return {'image': image, 'liver_mask': liver_mask, 'lesion_mask': lesion_mask}
    
    def _find_paste_position(self, liver_mask, lesion_mask, bbox, max_attempts=50):
        d, h, w = bbox
        
        for _ in range(max_attempts):
            z = random.randint(0, max(1, liver_mask.shape[0] - d))
            y = random.randint(0, max(1, liver_mask.shape[1] - h))
            x = random.randint(0, max(1, liver_mask.shape[2] - w))
            
            region = liver_mask[z:z+d, y:y+h, x:x+w]
            if region.mean() < 0.8:
                continue
            
            existing = lesion_mask[z:z+d, y:y+h, x:x+w]
            if existing.max() > 0:
                continue
            
            return (z, y, x)
        
        return None
    
    def _paste_lesion(self, image, lesion_mask, lesion_img, lesion_msk, z, y, x):
        d, h, w = lesion_msk.shape
        
        if self.intensity_matching:
            if image.ndim == 4:
                target_region = image[:, z:z+d, y:y+h, x:x+w]
            else:
                target_region = image[z:z+d, y:y+h, x:x+w]
            
            target_mean = target_region.mean()
            target_std = target_region.std() + 1e-8
            source_mean = lesion_img[lesion_msk > 0].mean() if (lesion_msk > 0).any() else lesion_img.mean()
            source_std = lesion_img[lesion_msk > 0].std() + 1e-8 if (lesion_msk > 0).any() else lesion_img.std() + 1e-8
            
            lesion_img = (lesion_img - source_mean) / source_std * target_std + target_mean
        
        # 边界混合
        if self.boundary_blending:
            blend_mask = ndimage.distance_transform_edt(lesion_msk > 0)
            blend_mask = np.clip(blend_mask / self.blend_radius, 0, 1)
        else:
            blend_mask = (lesion_msk > 0).astype(float)
        
        # 粘贴
        if image.ndim == 4:
            for c in range(image.shape[0]):
                image[c, z:z+d, y:y+h, x:x+w] = (
                    image[c, z:z+d, y:y+h, x:x+w] * (1 - blend_mask) +
                    lesion_img[c] * blend_mask
                )
        else:
            image[z:z+d, y:y+h, x:x+w] = (
                image[z:z+d, y:y+h, x:x+w] * (1 - blend_mask) +
                lesion_img * blend_mask
            )
        
        lesion_mask[z:z+d, y:y+h, x:x+w] = np.maximum(lesion_mask[z:z+d, y:y+h, x:x+w], lesion_msk)


class MixUp3D:
    """3D MixUp增强"""
    
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
    
    def __call__(self, image1, mask1, image2, mask2):
        lam = np.random.beta(self.alpha, self.alpha)
        mixed_image = lam * image1 + (1 - lam) * image2
        mixed_mask = mask1 if lam > 0.5 else mask2
        return mixed_image, mixed_mask, lam
