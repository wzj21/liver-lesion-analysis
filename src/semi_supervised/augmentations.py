"""
半监督学习数据增强模块
Data Augmentation for Semi-Supervised Learning

包含弱增强和强增强策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import random
from scipy import ndimage
from scipy.ndimage import gaussian_filter, map_coordinates


class WeakAugmentation3D:
    """
    弱增强 (Weak Augmentation)
    
    用于Teacher模型生成伪标签
    只包含轻微的几何变换
    """
    
    def __init__(
        self,
        random_flip: bool = True,
        random_rotation: Tuple[float, float] = (-5, 5),
        random_scale: Tuple[float, float] = (0.95, 1.05),
        p: float = 0.5
    ):
        """
        Args:
            random_flip: 是否随机翻转
            random_rotation: 旋转角度范围（度）
            random_scale: 缩放比例范围
            p: 每种增强的概率
        """
        self.random_flip = random_flip
        self.random_rotation = random_rotation
        self.random_scale = random_scale
        self.p = p
    
    def __call__(
        self,
        image: Union[np.ndarray, torch.Tensor],
        mask: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        应用弱增强
        
        Args:
            image: 输入图像 [C, D, H, W] 或 [D, H, W]
            mask: 可选的掩膜 [D, H, W]
            
        Returns:
            增强后的图像（和掩膜）
        """
        # 转换为numpy
        if torch.is_tensor(image):
            image = image.numpy()
        if mask is not None and torch.is_tensor(mask):
            mask = mask.numpy()
        
        # 确保是[C, D, H, W]格式
        if image.ndim == 3:
            image = image[np.newaxis, ...]
            squeeze_channel = True
        else:
            squeeze_channel = False
        
        # 随机翻转
        if self.random_flip and random.random() < self.p:
            axis = random.choice([1, 2, 3])  # D, H, W
            image = np.flip(image, axis=axis).copy()
            if mask is not None:
                mask = np.flip(mask, axis=axis-1).copy()
        
        # 随机旋转（仅在H-W平面）
        if self.random_rotation and random.random() < self.p:
            angle = random.uniform(*self.random_rotation)
            for c in range(image.shape[0]):
                for d in range(image.shape[1]):
                    image[c, d] = ndimage.rotate(
                        image[c, d], angle, reshape=False, order=1, mode='nearest'
                    )
            if mask is not None:
                for d in range(mask.shape[0]):
                    mask[d] = ndimage.rotate(
                        mask[d], angle, reshape=False, order=0, mode='nearest'
                    )
        
        if squeeze_channel:
            image = image[0]
        
        if mask is not None:
            return image, mask
        return image


class StrongAugmentation3D:
    """
    强增强 (Strong Augmentation)
    
    用于Student模型训练
    包含更多样化和更强烈的增强
    """
    
    def __init__(
        self,
        # 几何增强
        random_flip: bool = True,
        random_rotation: Tuple[float, float] = (-15, 15),
        random_scale: Tuple[float, float] = (0.8, 1.2),
        elastic_deformation: bool = True,
        elastic_alpha: float = 100,
        elastic_sigma: float = 10,
        # 强度增强
        random_contrast: Tuple[float, float] = (0.7, 1.3),
        random_brightness: Tuple[float, float] = (-0.2, 0.2),
        random_gamma: Tuple[float, float] = (0.7, 1.3),
        gaussian_noise: float = 0.02,
        gaussian_blur: bool = True,
        blur_sigma: Tuple[float, float] = (0.5, 1.5),
        # Cutout
        cutout: bool = True,
        cutout_ratio: float = 0.2,
        cutout_fill: str = 'zero',  # zero, mean, random
        # 概率
        p: float = 0.5
    ):
        """
        Args:
            各种增强参数...
        """
        self.random_flip = random_flip
        self.random_rotation = random_rotation
        self.random_scale = random_scale
        self.elastic_deformation = elastic_deformation
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        
        self.random_contrast = random_contrast
        self.random_brightness = random_brightness
        self.random_gamma = random_gamma
        self.gaussian_noise = gaussian_noise
        self.gaussian_blur = gaussian_blur
        self.blur_sigma = blur_sigma
        
        self.cutout = cutout
        self.cutout_ratio = cutout_ratio
        self.cutout_fill = cutout_fill
        
        self.p = p
    
    def __call__(
        self,
        image: Union[np.ndarray, torch.Tensor],
        mask: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        应用强增强
        
        Args:
            image: 输入图像
            mask: 可选的掩膜
            
        Returns:
            增强后的图像（和掩膜）
        """
        # 转换为numpy
        if torch.is_tensor(image):
            image = image.numpy()
        if mask is not None and torch.is_tensor(mask):
            mask = mask.numpy()
        
        # 确保是[C, D, H, W]格式
        if image.ndim == 3:
            image = image[np.newaxis, ...]
            squeeze_channel = True
        else:
            squeeze_channel = False
        
        # ========== 几何增强 ==========
        
        # 随机翻转
        if self.random_flip and random.random() < self.p:
            axis = random.choice([1, 2, 3])
            image = np.flip(image, axis=axis).copy()
            if mask is not None:
                mask = np.flip(mask, axis=axis-1).copy()
        
        # 随机旋转
        if self.random_rotation and random.random() < self.p:
            angle = random.uniform(*self.random_rotation)
            for c in range(image.shape[0]):
                for d in range(image.shape[1]):
                    image[c, d] = ndimage.rotate(
                        image[c, d], angle, reshape=False, order=1, mode='nearest'
                    )
            if mask is not None:
                for d in range(mask.shape[0]):
                    mask[d] = ndimage.rotate(
                        mask[d], angle, reshape=False, order=0, mode='nearest'
                    )
        
        # 随机缩放
        if self.random_scale and random.random() < self.p:
            scale = random.uniform(*self.random_scale)
            image = ndimage.zoom(image, [1, scale, scale, scale], order=1)
            if mask is not None:
                mask = ndimage.zoom(mask, [scale, scale, scale], order=0)
            
            # 裁剪或填充回原始大小
            # 这里简化处理，实际应用中需要更复杂的逻辑
        
        # 弹性变形
        if self.elastic_deformation and random.random() < self.p:
            image, mask = self._apply_elastic_deformation(image, mask)
        
        # ========== 强度增强 ==========
        
        # 随机对比度
        if self.random_contrast and random.random() < self.p:
            factor = random.uniform(*self.random_contrast)
            mean = image.mean()
            image = (image - mean) * factor + mean
        
        # 随机亮度
        if self.random_brightness and random.random() < self.p:
            delta = random.uniform(*self.random_brightness)
            image = image + delta
        
        # 随机Gamma
        if self.random_gamma and random.random() < self.p:
            gamma = random.uniform(*self.random_gamma)
            # 确保非负
            image_min = image.min()
            image = (image - image_min + 1e-8) ** gamma + image_min
        
        # 高斯噪声
        if self.gaussian_noise > 0 and random.random() < self.p:
            noise = np.random.normal(0, self.gaussian_noise, image.shape)
            image = image + noise
        
        # 高斯模糊
        if self.gaussian_blur and random.random() < self.p:
            sigma = random.uniform(*self.blur_sigma)
            for c in range(image.shape[0]):
                image[c] = gaussian_filter(image[c], sigma=sigma)
        
        # Cutout
        if self.cutout and random.random() < self.p:
            image = self._apply_cutout(image)
        
        # 裁剪到[0, 1]范围
        image = np.clip(image, 0, 1)
        
        if squeeze_channel:
            image = image[0]
        
        if mask is not None:
            return image, mask
        return image
    
    def _apply_elastic_deformation(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """应用弹性变形"""
        shape = image.shape[1:]  # D, H, W
        
        # 生成随机位移场
        dx = gaussian_filter(
            (np.random.rand(*shape) * 2 - 1),
            self.elastic_sigma
        ) * self.elastic_alpha
        dy = gaussian_filter(
            (np.random.rand(*shape) * 2 - 1),
            self.elastic_sigma
        ) * self.elastic_alpha
        dz = gaussian_filter(
            (np.random.rand(*shape) * 2 - 1),
            self.elastic_sigma
        ) * self.elastic_alpha
        
        # 创建网格
        z, y, x = np.meshgrid(
            np.arange(shape[0]),
            np.arange(shape[1]),
            np.arange(shape[2]),
            indexing='ij'
        )
        
        indices = [
            np.reshape(z + dz, (-1, 1)),
            np.reshape(y + dy, (-1, 1)),
            np.reshape(x + dx, (-1, 1))
        ]
        
        # 应用变形
        for c in range(image.shape[0]):
            image[c] = map_coordinates(
                image[c], indices, order=1, mode='reflect'
            ).reshape(shape)
        
        if mask is not None:
            mask = map_coordinates(
                mask, indices, order=0, mode='reflect'
            ).reshape(shape)
        
        return image, mask
    
    def _apply_cutout(self, image: np.ndarray) -> np.ndarray:
        """应用Cutout"""
        shape = image.shape[1:]  # D, H, W
        
        # 计算cutout大小
        cut_d = int(shape[0] * self.cutout_ratio)
        cut_h = int(shape[1] * self.cutout_ratio)
        cut_w = int(shape[2] * self.cutout_ratio)
        
        # 随机位置
        d = random.randint(0, shape[0] - cut_d)
        h = random.randint(0, shape[1] - cut_h)
        w = random.randint(0, shape[2] - cut_w)
        
        # 填充值
        if self.cutout_fill == 'zero':
            fill_value = 0
        elif self.cutout_fill == 'mean':
            fill_value = image.mean()
        else:  # random
            fill_value = np.random.uniform(0, 1)
        
        # 应用cutout
        image = image.copy()
        image[:, d:d+cut_d, h:h+cut_h, w:w+cut_w] = fill_value
        
        return image


class RandAugment3D:
    """
    RandAugment for 3D medical images
    
    随机选择N种增强并应用
    """
    
    def __init__(
        self,
        n: int = 2,  # 应用的增强数量
        m: int = 10,  # 增强强度 (0-30)
    ):
        """
        Args:
            n: 应用的增强数量
            m: 增强强度
        """
        self.n = n
        self.m = m
        
        # 定义可用的增强操作
        self.augmentations = [
            self._rotate,
            self._shear,
            self._translate,
            self._contrast,
            self._brightness,
            self._sharpness,
            self._blur,
            self._noise,
        ]
    
    def __call__(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """应用RandAugment"""
        # 随机选择n个增强
        ops = random.sample(self.augmentations, self.n)
        
        for op in ops:
            image, mask = op(image, mask)
        
        if mask is not None:
            return image, mask
        return image
    
    def _rotate(self, image, mask):
        """旋转"""
        magnitude = (self.m / 30) * 30  # 最大30度
        angle = random.uniform(-magnitude, magnitude)
        
        if image.ndim == 4:
            for c in range(image.shape[0]):
                for d in range(image.shape[1]):
                    image[c, d] = ndimage.rotate(
                        image[c, d], angle, reshape=False, order=1
                    )
        else:
            for d in range(image.shape[0]):
                image[d] = ndimage.rotate(
                    image[d], angle, reshape=False, order=1
                )
        
        if mask is not None:
            for d in range(mask.shape[0]):
                mask[d] = ndimage.rotate(
                    mask[d], angle, reshape=False, order=0
                )
        
        return image, mask
    
    def _shear(self, image, mask):
        """剪切变换"""
        magnitude = (self.m / 30) * 0.3
        shear = random.uniform(-magnitude, magnitude)
        
        # 简化实现
        return image, mask
    
    def _translate(self, image, mask):
        """平移"""
        magnitude = int((self.m / 30) * 10)
        tx = random.randint(-magnitude, magnitude)
        ty = random.randint(-magnitude, magnitude)
        
        image = np.roll(image, tx, axis=-1)
        image = np.roll(image, ty, axis=-2)
        
        if mask is not None:
            mask = np.roll(mask, tx, axis=-1)
            mask = np.roll(mask, ty, axis=-2)
        
        return image, mask
    
    def _contrast(self, image, mask):
        """对比度"""
        magnitude = (self.m / 30) * 0.5 + 0.5  # 0.5-1.0
        factor = random.uniform(magnitude, 2 - magnitude)
        
        mean = image.mean()
        image = (image - mean) * factor + mean
        
        return image, mask
    
    def _brightness(self, image, mask):
        """亮度"""
        magnitude = (self.m / 30) * 0.3
        delta = random.uniform(-magnitude, magnitude)
        
        image = image + delta
        
        return image, mask
    
    def _sharpness(self, image, mask):
        """锐化"""
        # 简化实现
        return image, mask
    
    def _blur(self, image, mask):
        """模糊"""
        magnitude = (self.m / 30) * 2
        sigma = random.uniform(0.1, magnitude)
        
        if image.ndim == 4:
            for c in range(image.shape[0]):
                image[c] = gaussian_filter(image[c], sigma=sigma)
        else:
            image = gaussian_filter(image, sigma=sigma)
        
        return image, mask
    
    def _noise(self, image, mask):
        """噪声"""
        magnitude = (self.m / 30) * 0.05
        noise = np.random.normal(0, magnitude, image.shape)
        image = image + noise
        
        return image, mask


class AugmentationWrapper:
    """
    增强包装器
    
    统一接口处理弱增强和强增强
    """
    
    def __init__(
        self,
        weak_aug: Optional[WeakAugmentation3D] = None,
        strong_aug: Optional[StrongAugmentation3D] = None
    ):
        """
        Args:
            weak_aug: 弱增强器
            strong_aug: 强增强器
        """
        self.weak_aug = weak_aug or WeakAugmentation3D()
        self.strong_aug = strong_aug or StrongAugmentation3D()
    
    def apply_weak(
        self,
        image: Union[np.ndarray, torch.Tensor],
        mask: Optional[Union[np.ndarray, torch.Tensor]] = None
    ):
        """应用弱增强"""
        return self.weak_aug(image, mask)
    
    def apply_strong(
        self,
        image: Union[np.ndarray, torch.Tensor],
        mask: Optional[Union[np.ndarray, torch.Tensor]] = None
    ):
        """应用强增强"""
        return self.strong_aug(image, mask)
    
    def apply_both(
        self,
        image: Union[np.ndarray, torch.Tensor],
        mask: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Dict[str, Any]:
        """
        同时应用弱增强和强增强
        
        Returns:
            包含弱增强和强增强结果的字典
        """
        weak_result = self.apply_weak(image, mask)
        strong_result = self.apply_strong(image, mask)
        
        if mask is not None:
            return {
                'weak_image': weak_result[0],
                'weak_mask': weak_result[1],
                'strong_image': strong_result[0],
                'strong_mask': strong_result[1]
            }
        else:
            return {
                'weak_image': weak_result,
                'strong_image': strong_result
            }
