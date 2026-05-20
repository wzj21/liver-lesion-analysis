#!/usr/bin/env python3
"""
自监督预训练脚本
Self-Supervised Pretraining Script

使用无标签数据进行自监督预训练
支持的方法:
- 对比学习 (Contrastive Learning)
- 掩码自编码器 (Masked Autoencoder)
- 旋转预测
- 图像恢复
"""

import os
import sys
import argparse
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class UnlabeledDataset(Dataset):
    """无标签数据集"""
    
    def __init__(self, data_dir: str, file_list: str = None, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        if file_list and os.path.exists(file_list):
            with open(file_list, 'r') as f:
                self.file_ids = [line.strip().replace('.nii.gz', '').replace('.nii', '') 
                                for line in f if line.strip()]
        else:
            # 直接扫描目录
            self.file_ids = [f.replace('.npz', '') for f in os.listdir(data_dir) if f.endswith('.npz')]
        
        print(f"加载无标签数据集: {len(self.file_ids)} 个样本")
    
    def __len__(self):
        return len(self.file_ids)
    
    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        data = np.load(os.path.join(self.data_dir, f"{file_id}.npz"))
        image = data['image'].astype(np.float32)
        
        # 确保有channel维度
        if image.ndim == 3:
            image = image[np.newaxis, ...]
        
        image = torch.from_numpy(image)
        
        if self.transform:
            image = self.transform(image)
        
        return {'image': image, 'id': file_id}


class ContrastiveAugmentation:
    """对比学习的数据增强"""
    
    def __init__(self, crop_size=(64, 128, 128)):
        self.crop_size = crop_size
    
    def __call__(self, image):
        """生成两个不同的增强视图"""
        view1 = self._augment(image.clone())
        view2 = self._augment(image.clone())
        return view1, view2
    
    def _augment(self, image):
        # 随机裁剪
        image = self._random_crop(image, self.crop_size)
        
        # 随机翻转
        for dim in [1, 2, 3]:
            if random.random() > 0.5:
                image = torch.flip(image, dims=[dim])
        
        # 随机强度变换
        if random.random() > 0.5:
            image = image * random.uniform(0.8, 1.2)
        if random.random() > 0.5:
            image = image + random.uniform(-0.1, 0.1)
        
        # 随机噪声
        if random.random() > 0.5:
            noise = torch.randn_like(image) * 0.05
            image = image + noise
        
        return torch.clamp(image, 0, 1)
    
    def _random_crop(self, image, crop_size):
        _, D, H, W = image.shape
        cd, ch, cw = crop_size
        
        # 确保不超出边界
        cd = min(cd, D)
        ch = min(ch, H)
        cw = min(cw, W)
        
        d = random.randint(0, max(0, D - cd))
        h = random.randint(0, max(0, H - ch))
        w = random.randint(0, max(0, W - cw))
        
        return image[:, d:d+cd, h:h+ch, w:w+cw]


class ProjectionHead(nn.Module):
    """投影头"""
    
    def __init__(self, in_dim, hidden_dim=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class ContrastiveLoss(nn.Module):
    """NT-Xent对比损失"""
    
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z1, z2):
        batch_size = z1.shape[0]
        
        # L2归一化
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # 拼接
        z = torch.cat([z1, z2], dim=0)
        
        # 相似度矩阵
        sim = torch.mm(z, z.t()) / self.temperature
        
        # 创建标签
        labels = torch.arange(batch_size, device=z.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)
        
        # 掩码对角线
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, float('-inf'))
        
        # 交叉熵损失
        loss = F.cross_entropy(sim, labels)
        
        return loss


class MaskedAutoencoder(nn.Module):
    """掩码自编码器"""
    
    def __init__(self, encoder, decoder_dim=256, mask_ratio=0.75):
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        
        # 获取encoder输出维度
        encoder_dim = 768  # 根据backbone调整
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(encoder_dim, decoder_dim, 4, stride=2, padding=1),
            nn.BatchNorm3d(decoder_dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(decoder_dim, decoder_dim // 2, 4, stride=2, padding=1),
            nn.BatchNorm3d(decoder_dim // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(decoder_dim // 2, decoder_dim // 4, 4, stride=2, padding=1),
            nn.BatchNorm3d(decoder_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(decoder_dim // 4, 1, 3, padding=1)
        )
    
    def forward(self, x):
        B, C, D, H, W = x.shape
        
        # 创建掩码
        mask = self._create_mask(x)
        
        # 掩码后的输入
        masked_x = x * (1 - mask)
        
        # 编码
        features = self.encoder(masked_x)
        if isinstance(features, dict):
            features = features.get('features', list(features.values())[-1])
        if isinstance(features, list):
            features = features[-1]
        
        # 解码
        reconstructed = self.decoder(features)
        
        # 调整大小
        if reconstructed.shape[2:] != x.shape[2:]:
            reconstructed = F.interpolate(reconstructed, size=x.shape[2:], mode='trilinear', align_corners=False)
        
        return reconstructed, mask
    
    def _create_mask(self, x):
        B, C, D, H, W = x.shape
        
        # 随机块掩码
        mask = torch.zeros_like(x)
        
        for b in range(B):
            num_masks = int(D * H * W * self.mask_ratio / (16 * 16 * 8))
            for _ in range(max(1, num_masks)):
                md, mh, mw = random.randint(4, 8), random.randint(8, 16), random.randint(8, 16)
                d = random.randint(0, max(0, D - md))
                h = random.randint(0, max(0, H - mh))
                w = random.randint(0, max(0, W - mw))
                mask[b, :, d:d+md, h:h+mh, w:w+mw] = 1
        
        return mask


class SelfSupervisedPretrainer:
    """自监督预训练器"""
    
    def __init__(
        self,
        encoder,
        method='contrastive',  # contrastive, mae, rotation
        device='cuda',
        learning_rate=1e-4,
        weight_decay=0.01,
        crop_size=(64, 128, 128)
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.method = method
        self.crop_size = crop_size
        
        self.encoder = encoder.to(self.device)
        
        if method == 'contrastive':
            # 对比学习
            self.augmentation = ContrastiveAugmentation(crop_size)
            encoder_dim = 768  # 根据backbone调整
            self.projection_head = ProjectionHead(encoder_dim, 256, 128).to(self.device)
            self.criterion = ContrastiveLoss(temperature=0.5)
            
            params = list(self.encoder.parameters()) + list(self.projection_head.parameters())
        
        elif method == 'mae':
            # 掩码自编码器
            self.mae = MaskedAutoencoder(self.encoder).to(self.device)
            self.criterion = nn.MSELoss()
            params = self.mae.parameters()
        
        else:
            params = self.encoder.parameters()
        
        self.optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
        self.scaler = GradScaler()
    
    def train_epoch(self, dataloader, epoch):
        self.encoder.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        for batch in pbar:
            images = batch['image'].to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast():
                if self.method == 'contrastive':
                    loss = self._contrastive_step(images)
                elif self.method == 'mae':
                    loss = self._mae_step(images)
                else:
                    loss = self._rotation_step(images)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(dataloader)
    
    def _contrastive_step(self, images):
        batch_size = images.shape[0]
        views1, views2 = [], []
        
        for i in range(batch_size):
            v1, v2 = self.augmentation(images[i])
            views1.append(v1)
            views2.append(v2)
        
        views1 = torch.stack(views1).to(self.device)
        views2 = torch.stack(views2).to(self.device)
        
        # 编码
        f1 = self.encoder(views1)
        f2 = self.encoder(views2)
        
        if isinstance(f1, dict):
            f1 = f1.get('global_features', f1.get('features'))
        if isinstance(f2, dict):
            f2 = f2.get('global_features', f2.get('features'))
        
        if isinstance(f1, list):
            f1 = f1[-1]
        if isinstance(f2, list):
            f2 = f2[-1]
        
        # 全局平均池化
        if f1.dim() > 2:
            f1 = f1.mean(dim=tuple(range(2, f1.dim())))
        if f2.dim() > 2:
            f2 = f2.mean(dim=tuple(range(2, f2.dim())))
        
        # 投影
        z1 = self.projection_head(f1)
        z2 = self.projection_head(f2)
        
        # 对比损失
        loss = self.criterion(z1, z2)
        
        return loss
    
    def _mae_step(self, images):
        # 随机裁剪
        images = self._random_crop_batch(images, self.crop_size)
        
        reconstructed, mask = self.mae(images)
        
        # 只计算掩码区域的损失
        loss = ((reconstructed - images) ** 2 * mask).sum() / (mask.sum() + 1e-6)
        
        return loss
    
    def _rotation_step(self, images):
        # 随机裁剪
        images = self._random_crop_batch(images, self.crop_size)
        
        batch_size = images.shape[0]
        
        # 随机旋转
        rotated_images = []
        labels = []
        
        for i in range(batch_size):
            k = random.randint(0, 3)  # 0, 90, 180, 270度
            rotated = torch.rot90(images[i], k, dims=[2, 3])
            rotated_images.append(rotated)
            labels.append(k)
        
        rotated_images = torch.stack(rotated_images)
        labels = torch.tensor(labels, device=self.device)
        
        # 预测旋转
        features = self.encoder(rotated_images)
        if isinstance(features, dict):
            features = features.get('global_features', features.get('features'))
        if isinstance(features, list):
            features = features[-1]
        
        if features.dim() > 2:
            features = features.mean(dim=tuple(range(2, features.dim())))
        
        # 分类头
        if not hasattr(self, 'rotation_head'):
            self.rotation_head = nn.Linear(features.shape[1], 4).to(self.device)
        
        logits = self.rotation_head(features)
        loss = F.cross_entropy(logits, labels)
        
        return loss
    
    def _random_crop_batch(self, images, crop_size):
        B, C, D, H, W = images.shape
        cd, ch, cw = crop_size
        
        cd = min(cd, D)
        ch = min(ch, H)
        cw = min(cw, W)
        
        cropped = []
        for i in range(B):
            d = random.randint(0, max(0, D - cd))
            h = random.randint(0, max(0, H - ch))
            w = random.randint(0, max(0, W - cw))
            cropped.append(images[i:i+1, :, d:d+cd, h:h+ch, w:w+cw])
        
        return torch.cat(cropped, dim=0)
    
    def save_checkpoint(self, path, epoch):
        torch.save({
            'epoch': epoch,
            'encoder_state_dict': self.encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"✓ 保存检查点: {path}")
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']


def main():
    parser = argparse.ArgumentParser(description="自监督预训练")
    parser.add_argument("--data_dir", type=str, default="data/processed/unlabeled/pretrain",
                        help="预处理后的无标签数据目录")
    parser.add_argument("--output_dir", type=str, default="checkpoints/pretrained",
                        help="输出目录")
    parser.add_argument("--method", type=str, default="contrastive",
                        choices=["contrastive", "mae", "rotation"],
                        help="预训练方法")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--crop_size", type=int, nargs=3, default=[64, 128, 128],
                        help="裁剪大小")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载进程数")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的检查点")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建数据集
    dataset = UnlabeledDataset(args.data_dir)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 创建编码器（使用ConvNeXt）
    try:
        from backbones import ConvNeXt3d
        encoder = ConvNeXt3d(in_channels=1, num_classes=0)  # 不需要分类头
    except:
        print("使用简化的编码器...")
        encoder = nn.Sequential(
            nn.Conv3d(1, 64, 7, stride=2, padding=3),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(128, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten()
        )
    
    # 创建预训练器
    pretrainer = SelfSupervisedPretrainer(
        encoder,
        method=args.method,
        learning_rate=args.lr,
        crop_size=tuple(args.crop_size)
    )
    
    # 恢复训练
    start_epoch = 0
    if args.resume:
        start_epoch = pretrainer.load_checkpoint(args.resume) + 1
        print(f"从epoch {start_epoch}恢复训练")
    
    # 训练
    print(f"\n开始自监督预训练 (方法: {args.method})")
    print(f"数据: {len(dataset)} 个样本")
    print(f"批大小: {args.batch_size}")
    print(f"总轮数: {args.epochs}")
    print("=" * 50)
    
    best_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        loss = pretrainer.train_epoch(dataloader, epoch + 1)
        
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss:.4f}")
        
        # 保存检查点
        if (epoch + 1) % 10 == 0:
            pretrainer.save_checkpoint(
                os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pth"),
                epoch
            )
        
        if loss < best_loss:
            best_loss = loss
            pretrainer.save_checkpoint(
                os.path.join(args.output_dir, "best_pretrained.pth"),
                epoch
            )
    
    # 保存最终模型
    pretrainer.save_checkpoint(
        os.path.join(args.output_dir, "final_pretrained.pth"),
        args.epochs - 1
    )
    
    print("\n" + "=" * 50)
    print("✓ 预训练完成!")
    print(f"最佳损失: {best_loss:.4f}")
    print(f"模型保存在: {args.output_dir}")


if __name__ == "__main__":
    main()
