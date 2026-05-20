"""
知识蒸馏训练器
Knowledge Distillation Trainer

统一的蒸馏训练接口
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import logging
import os
from tqdm import tqdm
import copy

from .losses import (
    CombinedDistillationLoss,
    FeatureProjector,
    SegmentationDistillationLoss,
    DetectionDistillationLoss
)
from .student_models import LightweightLiverLesionModel, create_student_model


class DistillationTrainer:
    """
    知识蒸馏训练器
    
    支持:
    - 多层特征蒸馏
    - 输出蒸馏
    - 关系蒸馏
    - 注意力蒸馏
    - 不确定性蒸馏
    - 分阶段蒸馏
    """
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        config: Dict = None,
        device: torch.device = torch.device('cuda'),
        logger: Optional[logging.Logger] = None
    ):
        """
        Args:
            teacher_model: 教师模型
            student_model: 学生模型
            train_dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器
            config: 配置字典
            device: 设备
            logger: 日志记录器
        """
        self.teacher = teacher_model.to(device)
        self.student = student_model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config or {}
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        
        # 冻结教师模型
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # 蒸馏损失
        self.distill_loss = CombinedDistillationLoss(config.get('distillation', {}))
        
        # 任务损失
        self.task_losses = self._create_task_losses()
        
        # 特征投影器（如果需要）
        self.projectors = self._create_projectors()
        
        # 优化器
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # 训练配置
        self.total_epochs = config.get('training', {}).get('total_epochs', 100)
        self.warmup_epochs = config.get('training', {}).get('warmup_epochs', 5)
        self.current_epoch = 0
        
        # 损失权重调度
        self.loss_weight_config = config.get('training', {}).get('loss_weight_schedule', {})
        
        # 混合精度
        self.use_amp = config.get('global', {}).get('mixed_precision', True)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # 最佳模型追踪
        self.best_metric = 0.0
        self.best_epoch = 0
        
        # 检查点目录
        self.checkpoint_dir = config.get('deployment', {}).get('output_dir', 'checkpoints/distillation')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 训练历史
        self.train_history = []
        self.val_history = []
    
    def _create_task_losses(self) -> Dict[str, nn.Module]:
        """创建任务损失函数"""
        return {
            'segmentation': SegmentationDistillationLoss(
                temperature=self.config.get('distillation', {}).get('output_distillation', {}).get('temperature', 4.0)
            ),
            'detection': DetectionDistillationLoss(),
            'classification': nn.CrossEntropyLoss()
        }
    
    def _create_projectors(self) -> Optional[nn.ModuleDict]:
        """创建特征投影器"""
        if not self.config.get('distillation', {}).get('feature_distillation', {}).get('projector', True):
            return None
        
        # 获取教师和学生的通道配置
        teacher_dims = [96, 192, 384, 768]  # 完整版
        student_dims = [48, 96, 192, 384]   # 轻量版
        
        projectors = nn.ModuleDict()
        for i, (s_dim, t_dim) in enumerate(zip(student_dims, teacher_dims)):
            if s_dim != t_dim:
                projectors[f'stage{i}'] = FeatureProjector(s_dim, t_dim).to(self.device)
        
        return projectors
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """创建优化器"""
        opt_config = self.config.get('training', {}).get('optimizer', {})
        
        # 收集所有需要优化的参数
        params = list(self.student.parameters())
        if self.projectors is not None:
            params += list(self.projectors.parameters())
        
        return torch.optim.AdamW(
            params,
            lr=opt_config.get('lr', 1e-4),
            weight_decay=opt_config.get('weight_decay', 0.01)
        )
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        sched_config = self.config.get('training', {}).get('scheduler', {})
        
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.total_epochs,
            eta_min=sched_config.get('min_lr', 1e-6)
        )
    
    def _get_loss_weights(self, epoch: int) -> Tuple[float, float]:
        """获取当前epoch的损失权重"""
        distill_config = self.loss_weight_config.get('distillation_weight', {})
        task_config = self.loss_weight_config.get('task_weight', {})
        
        # 蒸馏损失权重（逐渐降低）
        distill_start = distill_config.get('start', 0.8)
        distill_end = distill_config.get('end', 0.3)
        
        # 任务损失权重（逐渐提高）
        task_start = task_config.get('start', 0.2)
        task_end = task_config.get('end', 0.7)
        
        # 线性插值
        progress = min(1.0, epoch / self.total_epochs)
        
        distill_weight = distill_start + (distill_end - distill_start) * progress
        task_weight = task_start + (task_end - task_start) * progress
        
        return distill_weight, task_weight
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        训练一个epoch
        """
        self.student.train()
        self.current_epoch = epoch
        
        epoch_losses = {}
        distill_weight, task_weight = self._get_loss_weights(epoch)
        
        pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # 移动到设备
            images = batch['image'].to(self.device)
            targets = {k: v.to(self.device) if torch.is_tensor(v) else v 
                      for k, v in batch.items() if k != 'image'}
            
            # 教师前向传播（不计算梯度）
            with torch.no_grad():
                teacher_outputs = self.teacher(images)
            
            # 学生前向传播
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    student_outputs = self.student(images)
                    losses = self._compute_losses(
                        student_outputs, teacher_outputs, targets,
                        distill_weight, task_weight
                    )
            else:
                student_outputs = self.student(images)
                losses = self._compute_losses(
                    student_outputs, teacher_outputs, targets,
                    distill_weight, task_weight
                )
            
            # 反向传播
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(losses['total_loss']).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses['total_loss'].backward()
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                self.optimizer.step()
            
            # 记录损失
            for k, v in losses.items():
                if k not in epoch_losses:
                    epoch_losses[k] = []
                epoch_losses[k].append(v.item() if torch.is_tensor(v) else v)
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{losses['total_loss'].item():.4f}",
                'd_w': f"{distill_weight:.2f}",
                't_w': f"{task_weight:.2f}"
            })
        
        # 更新学习率
        self.scheduler.step()
        
        # 计算平均损失
        return {k: np.mean(v) for k, v in epoch_losses.items()}
    
    def _compute_losses(
        self,
        student_outputs: Dict,
        teacher_outputs: Dict,
        targets: Dict,
        distill_weight: float,
        task_weight: float
    ) -> Dict[str, torch.Tensor]:
        """
        计算总损失
        """
        losses = {}
        
        # 1. 蒸馏损失
        # 投影特征（如果需要）
        if self.projectors is not None and 'features' in student_outputs:
            projected_features = []
            for i, feat in enumerate(student_outputs['features']):
                proj_name = f'stage{i}'
                if proj_name in self.projectors:
                    projected_features.append(self.projectors[proj_name](feat))
                else:
                    projected_features.append(feat)
            student_outputs_proj = {**student_outputs, 'features': projected_features}
        else:
            student_outputs_proj = student_outputs
        
        distill_losses = self.distill_loss(student_outputs_proj, teacher_outputs, targets)
        
        for k, v in distill_losses.items():
            losses[f'distill_{k}'] = distill_weight * v
        
        # 2. 任务损失
        task_loss = torch.tensor(0.0, device=self.device)
        
        # 分割损失
        if 'segmentation' in student_outputs and 'mask' in targets:
            seg_loss = self.task_losses['segmentation'](
                student_outputs['segmentation']['logits'],
                teacher_outputs['segmentation']['logits'],
                targets.get('mask')
            )
            task_loss += seg_loss
            losses['task_seg_loss'] = seg_loss
        
        # 检测损失（如果有目标）
        if 'detection' in student_outputs and 'boxes' in targets:
            det_loss = self.task_losses['detection'](
                student_outputs['detection'],
                teacher_outputs['detection']
            )
            task_loss += det_loss
            losses['task_det_loss'] = det_loss
        
        # 分类损失
        if 'classification' in student_outputs and 'label' in targets:
            cls_logits = student_outputs['classification']['logits']
            cls_loss = self.task_losses['classification'](cls_logits, targets['label'])
            task_loss += cls_loss
            losses['task_cls_loss'] = cls_loss
        
        losses['total_task_loss'] = task_weight * task_loss
        
        # 3. 总损失
        losses['total_loss'] = sum(v for k, v in losses.items() 
                                   if k.startswith('distill_') or k == 'total_task_loss')
        
        return losses
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """验证模型"""
        if self.val_dataloader is None:
            return {}
        
        self.student.eval()
        
        all_preds = []
        all_targets = []
        
        for batch in tqdm(self.val_dataloader, desc="Validation"):
            images = batch['image'].to(self.device)
            
            outputs = self.student(images)
            
            # 收集预测
            if 'segmentation' in outputs:
                pred = outputs['segmentation']['logits'].argmax(dim=1)
                all_preds.append(pred.cpu())
                if 'mask' in batch:
                    all_targets.append(batch['mask'])
        
        # 计算指标
        if all_preds and all_targets:
            all_preds = torch.cat(all_preds)
            all_targets = torch.cat(all_targets)
            
            # Dice
            intersection = ((all_preds == 1) & (all_targets == 1)).float().sum()
            union = (all_preds == 1).float().sum() + (all_targets == 1).float().sum()
            dice = (2 * intersection / (union + 1e-8)).item()
            
            return {'val_dice': dice}
        
        return {}
    
    def train(self) -> Dict[str, List[float]]:
        """完整训练流程"""
        self.logger.info(f"Starting distillation training for {self.total_epochs} epochs")
        self.logger.info(f"Teacher params: {sum(p.numel() for p in self.teacher.parameters()):,}")
        self.logger.info(f"Student params: {sum(p.numel() for p in self.student.parameters()):,}")
        
        for epoch in range(self.total_epochs):
            # 训练
            train_losses = self.train_epoch(epoch)
            self.train_history.append(train_losses)
            
            # 验证
            val_metrics = self.validate()
            self.val_history.append(val_metrics)
            
            # 日志
            log_str = f"Epoch {epoch}: "
            log_str += f"total_loss: {train_losses.get('total_loss', 0):.4f}"
            if val_metrics:
                log_str += f" | val_dice: {val_metrics.get('val_dice', 0):.4f}"
            self.logger.info(log_str)
            
            # 保存最佳模型
            current_metric = val_metrics.get('val_dice', 0)
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.best_epoch = epoch
                self.save_checkpoint(epoch, is_best=True)
            
            # 定期保存
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch)
        
        self.logger.info(f"Training completed. Best dice: {self.best_metric:.4f} at epoch {self.best_epoch}")
        
        return {
            'train_history': self.train_history,
            'val_history': self.val_history
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'student_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config
        }
        
        if self.projectors is not None:
            checkpoint['projectors_state_dict'] = self.projectors.state_dict()
        
        # 保存最新
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'latest.pth'))
        
        # 保存最佳
        if is_best:
            torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'best_student.pth'))
        
        # 周期保存
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, f'epoch_{epoch}.pth'))
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.student.load_state_dict(checkpoint['student_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint.get('best_metric', 0)
        
        if self.projectors is not None and 'projectors_state_dict' in checkpoint:
            self.projectors.load_state_dict(checkpoint['projectors_state_dict'])
        
        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")


class DynamicInference:
    """
    动态推理
    
    根据样本难度选择使用Teacher或Student
    """
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        difficulty_threshold: float = 0.3,
        method: str = "uncertainty",  # uncertainty, confidence, entropy
        device: torch.device = torch.device('cuda')
    ):
        self.teacher = teacher_model.to(device)
        self.student = student_model.to(device)
        self.difficulty_threshold = difficulty_threshold
        self.method = method
        self.device = device
        
        self.teacher.eval()
        self.student.eval()
        
        # 统计
        self.stats = {'student_count': 0, 'teacher_count': 0}
    
    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        动态推理
        
        Args:
            x: 输入图像
            
        Returns:
            推理结果
        """
        # 先用Student推理
        student_outputs = self.student(x)
        
        # 评估难度
        difficulty = self._estimate_difficulty(student_outputs)
        
        # 决定是否使用Teacher
        if difficulty > self.difficulty_threshold:
            # 困难样本使用Teacher
            teacher_outputs = self.teacher(x)
            self.stats['teacher_count'] += 1
            return {'outputs': teacher_outputs, 'model': 'teacher', 'difficulty': difficulty}
        else:
            # 简单样本使用Student
            self.stats['student_count'] += 1
            return {'outputs': student_outputs, 'model': 'student', 'difficulty': difficulty}
    
    def _estimate_difficulty(self, outputs: Dict) -> float:
        """估计样本难度"""
        if self.method == "uncertainty":
            # 使用Evidential不确定性
            if 'segmentation' in outputs and 'uncertainty' in outputs['segmentation']:
                return outputs['segmentation']['uncertainty'].mean().item()
            elif 'classification' in outputs and 'uncertainty' in outputs['classification']:
                return outputs['classification']['uncertainty'].mean().item()
        
        elif self.method == "confidence":
            # 使用置信度（1-confidence作为难度）
            if 'segmentation' in outputs:
                probs = torch.softmax(outputs['segmentation']['logits'], dim=1)
                confidence = probs.max(dim=1)[0].mean()
                return 1 - confidence.item()
            elif 'classification' in outputs:
                probs = torch.softmax(outputs['classification']['logits'], dim=1)
                confidence = probs.max(dim=1)[0].mean()
                return 1 - confidence.item()
        
        elif self.method == "entropy":
            # 使用熵
            if 'classification' in outputs:
                probs = torch.softmax(outputs['classification']['logits'], dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
                max_entropy = np.log(probs.shape[1])
                return (entropy / max_entropy).item()
        
        return 0.5  # 默认值
    
    def get_speedup(self) -> float:
        """计算实际加速比"""
        total = self.stats['student_count'] + self.stats['teacher_count']
        if total == 0:
            return 1.0
        
        student_ratio = self.stats['student_count'] / total
        
        # 假设Student是Teacher的2倍速度
        student_speed = 2.0
        teacher_speed = 1.0
        
        avg_speed = student_ratio * student_speed + (1 - student_ratio) * teacher_speed
        
        return avg_speed
    
    def reset_stats(self):
        """重置统计"""
        self.stats = {'student_count': 0, 'teacher_count': 0}
