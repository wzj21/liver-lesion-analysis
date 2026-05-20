"""
数据增强模块
Data Augmentation Module

医学图像专用的数据增强
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List
import random


class Compose:
    """组合多个变换"""
    
    def __init__(self, transforms: List):
        self.transforms = transforms
    
    def __call__(self, data: Dict) -> Dict:
        for t in self.transforms:
            data = t(data)
        return data


class RandomFlip3D:
    """3D随机翻转"""
    
    def __init__(self, prob: float = 0.5, axes: List[int] = [0, 1, 2]):
        self.prob = prob
        self.axes = axes
    
    def __call__(self, data: Dict) -> Dict:
        image = data['image']
        
        for axis in self.axes:
            if random.random() < self.prob:
                # 翻转图像
                image = torch.flip(image, dims=[axis + 1])  # +1因为有channel维度
                
                # 翻转mask
                for key in ['liver_mask', 'lesion_mask']:
                    if key in data and data[key] is not None:
                        data[key] = torch.flip(data[key], dims=[axis])
        
        data['image'] = image
        return data


class RandomRotate3D:
    """3D随机旋转（90度倍数）"""
    
    def __init__(self, prob: float = 0.5, axes: Tuple[int, int] = (1, 2)):
        self.prob = prob
        self.axes = axes
    
    def __call__(self, data: Dict) -> Dict:
        if random.random() > self.prob:
            return data
        
        k = random.choice([1, 2, 3])  # 旋转90, 180, 270度
        
        image = data['image']
        # 调整axes以考虑channel维度
        axes = (self.axes[0] + 1, self.axes[1] + 1)
        image = torch.rot90(image, k, axes)
        data['image'] = image
        
        for key in ['liver_mask', 'lesion_mask']:
            if key in data and data[key] is not None:
                data[key] = torch.rot90(data[key], k, self.axes)
        
        return data


class RandomCrop3D:
    """3D随机裁剪"""
    
    def __init__(self, output_size: Tuple[int, int, int]):
        self.output_size = output_size
    
    def __call__(self, data: Dict) -> Dict:
        image = data['image']
        
        # 获取输入尺寸 [C, D, H, W]
        _, d, h, w = image.shape
        od, oh, ow = self.output_size
        
        # 随机起始位置
        if d > od:
            z = random.randint(0, d - od)
        else:
            z = 0
        
        if h > oh:
            y = random.randint(0, h - oh)
        else:
            y = 0
        
        if w > ow:
            x = random.randint(0, w - ow)
        else:
            x = 0
        
        # 裁剪
        image = image[:, z:z+od, y:y+oh, x:x+ow]
        data['image'] = image
        
        for key in ['liver_mask', 'lesion_mask']:
            if key in data and data[key] is not None:
                data[key] = data[key][z:z+od, y:y+oh, x:x+ow]
        
        return data


class CenterCrop3D:
    """3D中心裁剪"""
    
    def __init__(self, output_size: Tuple[int, int, int]):
        self.output_size = output_size
    
    def __call__(self, data: Dict) -> Dict:
        image = data['image']
        
        _, d, h, w = image.shape
        od, oh, ow = self.output_size
        
        z = max(0, (d - od) // 2)
        y = max(0, (h - oh) // 2)
        x = max(0, (w - ow) // 2)
        
        image = image[:, z:z+od, y:y+oh, x:x+ow]
        data['image'] = image
        
        for key in ['liver_mask', 'lesion_mask']:
            if key in data and data[key] is not None:
                data[key] = data[key][z:z+od, y:y+oh, x:x+ow]
        
        return data


class RandomIntensityShift:
    """随机强度偏移"""
    
    def __init__(self, shift_range: float = 0.1):
        self.shift_range = shift_range
    
    def __call__(self, data: Dict) -> Dict:
        shift = random.uniform(-self.shift_range, self.shift_range)
        data['image'] = data['image'] + shift
        return data


class RandomIntensityScale:
    """随机强度缩放"""
    
    def __init__(self, scale_range: Tuple[float, float] = (0.9, 1.1)):
        self.scale_range = scale_range
    
    def __call__(self, data: Dict) -> Dict:
        scale = random.uniform(*self.scale_range)
        data['image'] = data['image'] * scale
        return data


class RandomGaussianNoise:
    """随机高斯噪声"""
    
    def __init__(self, std_range: Tuple[float, float] = (0.0, 0.1), prob: float = 0.5):
        self.std_range = std_range
        self.prob = prob
    
    def __call__(self, data: Dict) -> Dict:
        if random.random() > self.prob:
            return data
        
        std = random.uniform(*self.std_range)
        noise = torch.randn_like(data['image']) * std
        data['image'] = data['image'] + noise
        return data


class RandomGaussianBlur3D:
    """随机高斯模糊"""
    
    def __init__(self, sigma_range: Tuple[float, float] = (0.5, 1.5), prob: float = 0.3):
        self.sigma_range = sigma_range
        self.prob = prob
    
    def __call__(self, data: Dict) -> Dict:
        if random.random() > self.prob:
            return data
        
        sigma = random.uniform(*self.sigma_range)
        
        # 简化实现：使用平均池化近似
        kernel_size = int(sigma * 4) // 2 * 2 + 1
        if kernel_size > 1:
            image = data['image'].unsqueeze(0)  # [1, C, D, H, W]
            padding = kernel_size // 2
            image = F.avg_pool3d(image, kernel_size=kernel_size, stride=1, padding=padding)
            data['image'] = image.squeeze(0)
        
        return data


class RandomElasticDeformation:
    """随机弹性变形"""
    
    def __init__(
        self, 
        alpha: float = 100,
        sigma: float = 10,
        prob: float = 0.3
    ):
        self.alpha = alpha
        self.sigma = sigma
        self.prob = prob
    
    def __call__(self, data: Dict) -> Dict:
        if random.random() > self.prob:
            return data
        
        image = data['image']
        shape = image.shape[1:]  # D, H, W
        
        # 生成随机位移场
        dx = torch.randn(1, *shape) * self.sigma
        dy = torch.randn(1, *shape) * self.sigma
        dz = torch.randn(1, *shape) * self.sigma
        
        # 高斯平滑
        # 简化：使用小位移
        dx = dx * self.alpha / 1000
        dy = dy * self.alpha / 1000
        dz = dz * self.alpha / 1000
        
        # 创建采样网格
        d, h, w = shape
        grid_z, grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, d),
            torch.linspace(-1, 1, h),
            torch.linspace(-1, 1, w),
            indexing='ij'
        )
        
        # 添加位移
        grid_x = grid_x + dx.squeeze()
        grid_y = grid_y + dy.squeeze()
        grid_z = grid_z + dz.squeeze()
        
        # 构建采样网格 [1, D, H, W, 3]
        grid = torch.stack([grid_x, grid_y, grid_z], dim=-1).unsqueeze(0)
        
        # 应用变形
        image = image.unsqueeze(0)  # [1, C, D, H, W]
        image = F.grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=True)
        data['image'] = image.squeeze(0)
        
        # 变形mask
        for key in ['liver_mask', 'lesion_mask']:
            if key in data and data[key] is not None:
                mask = data[key].unsqueeze(0).unsqueeze(0).float()
                mask = F.grid_sample(mask, grid, mode='nearest', padding_mode='border', align_corners=True)
                data[key] = mask.squeeze().long()
        
        return data


class Normalize:
    """标准化"""
    
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std
    
    def __call__(self, data: Dict) -> Dict:
        data['image'] = (data['image'] - self.mean) / (self.std + 1e-8)
        return data


class ClipIntensity:
    """裁剪强度范围"""
    
    def __init__(self, min_val: float = 0.0, max_val: float = 1.0):
        self.min_val = min_val
        self.max_val = max_val
    
    def __call__(self, data: Dict) -> Dict:
        data['image'] = torch.clamp(data['image'], self.min_val, self.max_val)
        return data


class Resize3D:
    """3D尺寸调整"""
    
    def __init__(self, output_size: Tuple[int, int, int]):
        self.output_size = output_size
    
    def __call__(self, data: Dict) -> Dict:
        image = data['image']
        
        # [C, D, H, W] -> [1, C, D, H, W]
        image = image.unsqueeze(0)
        image = F.interpolate(image, size=self.output_size, mode='trilinear', align_corners=False)
        data['image'] = image.squeeze(0)
        
        for key in ['liver_mask', 'lesion_mask']:
            if key in data and data[key] is not None:
                mask = data[key].unsqueeze(0).unsqueeze(0).float()
                mask = F.interpolate(mask, size=self.output_size, mode='nearest')
                data[key] = mask.squeeze().long()
        
        return data


def get_train_transforms(config: Dict = None) -> Compose:
    """获取训练时的数据变换"""
    config = config or {}
    
    transforms = [
        RandomFlip3D(prob=0.5, axes=[0, 1, 2]),
        RandomRotate3D(prob=0.3),
        RandomIntensityShift(shift_range=0.1),
        RandomIntensityScale(scale_range=(0.9, 1.1)),
        RandomGaussianNoise(std_range=(0.0, 0.05), prob=0.3),
        ClipIntensity(min_val=0.0, max_val=1.0),
    ]
    
    if config.get('use_elastic', False):
        transforms.insert(2, RandomElasticDeformation(prob=0.2))
    
    if 'crop_size' in config:
        transforms.insert(0, RandomCrop3D(config['crop_size']))
    
    if 'target_size' in config:
        transforms.append(Resize3D(config['target_size']))
    
    return Compose(transforms)


def get_val_transforms(config: Dict = None) -> Compose:
    """获取验证时的数据变换"""
    config = config or {}
    
    transforms = [
        ClipIntensity(min_val=0.0, max_val=1.0),
    ]
    
    if 'crop_size' in config:
        transforms.insert(0, CenterCrop3D(config['crop_size']))
    
    if 'target_size' in config:
        transforms.append(Resize3D(config['target_size']))
    
    return Compose(transforms)


# 半监督学习的弱/强增强
def get_weak_transforms() -> Compose:
    """弱增强（用于Teacher/伪标签生成）"""
    return Compose([
        RandomFlip3D(prob=0.5, axes=[0, 1, 2]),
        ClipIntensity(min_val=0.0, max_val=1.0),
    ])


def get_strong_transforms() -> Compose:
    """强增强（用于Student/一致性正则）"""
    return Compose([
        RandomFlip3D(prob=0.5, axes=[0, 1, 2]),
        RandomRotate3D(prob=0.5),
        RandomIntensityShift(shift_range=0.2),
        RandomIntensityScale(scale_range=(0.8, 1.2)),
        RandomGaussianNoise(std_range=(0.0, 0.1), prob=0.5),
        RandomGaussianBlur3D(prob=0.3),
        ClipIntensity(min_val=0.0, max_val=1.0),
    ])
