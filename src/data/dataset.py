"""
数据集模块
Dataset Module

支持多种数据格式和任务
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
from typing import Dict, List, Optional, Tuple, Any, Callable
import random


class LiverLesionDataset(Dataset):
    """
    肝脏病灶数据集
    
    支持:
    - NIfTI (.nii, .nii.gz)
    - NumPy (.npy, .npz)
    - 预处理后的数据
    """
    
    def __init__(
        self,
        data_dir: str,
        annotation_file: str,
        mode: str = 'train',  # train, val, test
        task: str = 'all',    # all, segmentation, detection, classification
        transform: Optional[Callable] = None,
        cache_data: bool = False
    ):
        """
        Args:
            data_dir: 数据目录
            annotation_file: 标注JSON文件路径
            mode: 模式
            task: 任务类型
            transform: 数据变换
            cache_data: 是否缓存数据到内存
        """
        self.data_dir = data_dir
        self.mode = mode
        self.task = task
        self.transform = transform
        self.cache_data = cache_data
        
        # 加载标注
        with open(annotation_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        
        self.patient_ids = list(self.annotations.keys())
        
        # 数据缓存
        self.cache = {} if cache_data else None
        
        print(f"加载数据集: {len(self.patient_ids)} 个样本 ({mode})")
    
    def __len__(self) -> int:
        return len(self.patient_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        patient_id = self.patient_ids[idx]
        
        # 检查缓存
        if self.cache is not None and patient_id in self.cache:
            data = self.cache[patient_id].copy()
        else:
            data = self._load_data(patient_id)
            if self.cache is not None:
                self.cache[patient_id] = data.copy()
        
        # 应用变换
        if self.transform is not None:
            data = self.transform(data)
        
        return data
    
    def _load_data(self, patient_id: str) -> Dict[str, Any]:
        """加载单个样本数据"""
        info = self.annotations[patient_id]
        
        # 尝试加载预处理后的数据
        npz_path = os.path.join(self.data_dir, f"{patient_id}.npz")
        npy_path = os.path.join(self.data_dir, f"{patient_id}.npy")
        
        if os.path.exists(npz_path):
            loaded = np.load(npz_path)
            image = loaded['image']
            liver_mask = loaded.get('liver_mask', None)
            lesion_mask = loaded.get('lesion_mask', None)
        elif os.path.exists(npy_path):
            image = np.load(npy_path)
            liver_mask = None
            lesion_mask = None
        else:
            # 加载原始NIfTI
            image = self._load_nifti(info.get('image_path'))
            liver_mask = self._load_nifti(info.get('liver_mask_path'))
            lesion_mask = self._load_nifti(info.get('lesion_mask_path'))
        
        # 确保image是float32
        if image is not None:
            image = image.astype(np.float32)
            # 添加通道维度 [C, D, H, W]
            if image.ndim == 3:
                image = image[np.newaxis, ...]
        
        # 构建返回数据
        data = {
            'patient_id': patient_id,
            'image': torch.from_numpy(image) if image is not None else None,
        }
        
        # 添加标注
        if liver_mask is not None:
            data['liver_mask'] = torch.from_numpy(liver_mask.astype(np.int64))
        
        if lesion_mask is not None:
            data['lesion_mask'] = torch.from_numpy(lesion_mask.astype(np.int64))
        
        # 添加分类标签
        if 'diagnosis_code' in info and info['diagnosis_code'] >= 0:
            data['label'] = torch.tensor(info['diagnosis_code'], dtype=torch.long)
        
        if 'is_active' in info and info['is_active'] is not None:
            data['is_active'] = torch.tensor(1 if info['is_active'] else 0, dtype=torch.long)
        
        # 添加检测框
        if 'lesions' in info:
            boxes = []
            labels = []
            for lesion in info['lesions']:
                if 'bbox' in lesion:
                    boxes.append(lesion['bbox'])
                    labels.append(lesion.get('type_code', 1))
            
            if boxes:
                data['boxes'] = torch.tensor(boxes, dtype=torch.float32)
                data['box_labels'] = torch.tensor(labels, dtype=torch.long)
        
        return data
    
    def _load_nifti(self, path: Optional[str]) -> Optional[np.ndarray]:
        """加载NIfTI文件"""
        if path is None or not os.path.exists(path):
            return None
        
        try:
            import nibabel as nib
            return nib.load(path).get_fdata()
        except Exception as e:
            print(f"警告: 无法加载 {path}: {e}")
            return None
    
    def get_class_distribution(self) -> Dict[int, int]:
        """获取类别分布"""
        distribution = {}
        for patient_id in self.patient_ids:
            info = self.annotations[patient_id]
            code = info.get('diagnosis_code', -1)
            if code >= 0:
                distribution[code] = distribution.get(code, 0) + 1
        return distribution
    
    def get_sample_weights(self) -> torch.Tensor:
        """获取样本权重（用于不平衡数据）"""
        distribution = self.get_class_distribution()
        total = sum(distribution.values())
        
        weights = []
        for patient_id in self.patient_ids:
            info = self.annotations[patient_id]
            code = info.get('diagnosis_code', -1)
            if code >= 0 and code in distribution:
                weight = total / (len(distribution) * distribution[code])
            else:
                weight = 1.0
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32)


class SemiSupervisedDataset(Dataset):
    """
    半监督学习数据集
    
    同时处理有标注和无标注数据
    """
    
    def __init__(
        self,
        labeled_data_dir: str,
        labeled_annotation_file: str,
        unlabeled_data_dir: str,
        unlabeled_list_file: str,
        transform_weak: Optional[Callable] = None,
        transform_strong: Optional[Callable] = None,
        labeled_ratio: float = 0.5
    ):
        """
        Args:
            labeled_data_dir: 有标注数据目录
            labeled_annotation_file: 有标注数据标注文件
            unlabeled_data_dir: 无标注数据目录
            unlabeled_list_file: 无标注数据列表文件
            transform_weak: 弱增强
            transform_strong: 强增强
            labeled_ratio: 每个batch中有标注数据的比例
        """
        # 有标注数据
        self.labeled_dataset = LiverLesionDataset(
            labeled_data_dir, labeled_annotation_file, 
            mode='train', transform=None
        )
        
        # 无标注数据
        with open(unlabeled_list_file, 'r') as f:
            self.unlabeled_ids = [line.strip() for line in f]
        self.unlabeled_data_dir = unlabeled_data_dir
        
        self.transform_weak = transform_weak
        self.transform_strong = transform_strong
        self.labeled_ratio = labeled_ratio
        
        print(f"半监督数据集: {len(self.labeled_dataset)} 有标注, {len(self.unlabeled_ids)} 无标注")
    
    def __len__(self) -> int:
        return len(self.labeled_dataset) + len(self.unlabeled_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # 决定返回有标注还是无标注数据
        if idx < len(self.labeled_dataset):
            # 有标注数据
            data = self.labeled_dataset[idx]
            data['is_labeled'] = True
            
            if self.transform_weak:
                data = self.transform_weak(data)
        else:
            # 无标注数据
            unlabeled_idx = idx - len(self.labeled_dataset)
            patient_id = self.unlabeled_ids[unlabeled_idx]
            
            # 加载图像
            npz_path = os.path.join(self.unlabeled_data_dir, f"{patient_id}.npz")
            loaded = np.load(npz_path)
            image = torch.from_numpy(loaded['image'].astype(np.float32))
            if image.ndim == 3:
                image = image.unsqueeze(0)
            
            data = {
                'patient_id': patient_id,
                'image': image,
                'is_labeled': False
            }
            
            # 生成弱增强和强增强版本
            if self.transform_weak:
                data_weak = self.transform_weak({'image': image.clone()})
                data['image_weak'] = data_weak['image']
            
            if self.transform_strong:
                data_strong = self.transform_strong({'image': image.clone()})
                data['image_strong'] = data_strong['image']
        
        return data


class MultiPhaseDataset(Dataset):
    """
    多时相数据集
    
    用于Stage3的时序分类
    """
    
    def __init__(
        self,
        data_dir: str,
        annotation_file: str,
        phases: List[str] = ['arterial', 'portal', 'delayed'],
        num_slices: int = 8,
        transform: Optional[Callable] = None
    ):
        """
        Args:
            data_dir: 数据目录
            annotation_file: 标注文件
            phases: 时相列表
            num_slices: 每个时相选择的切片数
            transform: 数据变换
        """
        with open(annotation_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        
        self.data_dir = data_dir
        self.phases = phases
        self.num_slices = num_slices
        self.transform = transform
        self.patient_ids = list(self.annotations.keys())
    
    def __len__(self) -> int:
        return len(self.patient_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        patient_id = self.patient_ids[idx]
        info = self.annotations[patient_id]
        
        # 加载各时相数据
        phase_images = []
        for phase in self.phases:
            phase_path = os.path.join(self.data_dir, f"{patient_id}_{phase}.npz")
            if os.path.exists(phase_path):
                loaded = np.load(phase_path)
                image = loaded['image']
            else:
                # 如果没有特定时相，使用默认
                default_path = os.path.join(self.data_dir, f"{patient_id}.npz")
                loaded = np.load(default_path)
                image = loaded['image']
            
            # 选择关键切片
            slices = self._select_slices(image, self.num_slices)
            phase_images.append(slices)
        
        # [num_phases, num_slices, H, W]
        multi_phase = np.stack(phase_images, axis=0)
        
        data = {
            'patient_id': patient_id,
            'image': torch.from_numpy(multi_phase.astype(np.float32)),
            'label': torch.tensor(info.get('diagnosis_code', 0), dtype=torch.long)
        }
        
        if 'is_active' in info and info['is_active'] is not None:
            data['is_active'] = torch.tensor(1 if info['is_active'] else 0, dtype=torch.long)
        
        if self.transform:
            data = self.transform(data)
        
        return data
    
    def _select_slices(self, volume: np.ndarray, num_slices: int) -> np.ndarray:
        """选择关键切片"""
        D = volume.shape[0]
        
        if D <= num_slices:
            # 不够，需要插值
            indices = np.linspace(0, D-1, num_slices, dtype=int)
        else:
            # 均匀采样
            indices = np.linspace(0, D-1, num_slices, dtype=int)
        
        return volume[indices]


def create_data_loaders(
    data_config: Dict[str, Any],
    batch_size: int = 4,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建数据加载器
    
    Args:
        data_config: 数据配置
        batch_size: 批大小
        num_workers: 工作进程数
        
    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_dataset = LiverLesionDataset(
        data_dir=data_config['train_dir'],
        annotation_file=data_config['train_annotation'],
        mode='train',
        transform=data_config.get('train_transform')
    )
    
    val_dataset = LiverLesionDataset(
        data_dir=data_config['val_dir'],
        annotation_file=data_config['val_annotation'],
        mode='val',
        transform=data_config.get('val_transform')
    )
    
    test_dataset = LiverLesionDataset(
        data_dir=data_config['test_dir'],
        annotation_file=data_config['test_annotation'],
        mode='test',
        transform=data_config.get('test_transform')
    )
    
    # 处理类别不平衡
    if data_config.get('use_weighted_sampling', False):
        weights = train_dataset.get_sample_weights()
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, 
            sampler=sampler, num_workers=num_workers,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, 
            shuffle=True, num_workers=num_workers,
            pin_memory=True
        )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    自定义collate函数
    
    处理不同大小的图像和可选的标注
    """
    collated = {}
    
    keys = batch[0].keys()
    
    for key in keys:
        values = [item[key] for item in batch if key in item and item[key] is not None]
        
        if len(values) == 0:
            continue
        
        if isinstance(values[0], torch.Tensor):
            # 尝试堆叠，如果大小不同则保持列表
            try:
                collated[key] = torch.stack(values, dim=0)
            except:
                collated[key] = values
        elif isinstance(values[0], (int, float)):
            collated[key] = torch.tensor(values)
        else:
            collated[key] = values
    
    return collated
