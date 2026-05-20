"""
Teacher-Student半监督学习框架
Teacher-Student Semi-Supervised Learning Framework

用于Stage2检测分割任务的半监督学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .base import (
    EMAModel,
    PseudoLabelGenerator,
    SegmentationPseudoLabelGenerator,
    DetectionPseudoLabelGenerator,
    ConsistencyLoss,
    RampUpScheduler,
    AdaptiveThreshold,
    PseudoLabelQualityMonitor
)


class TeacherStudentFramework(nn.Module):
    """
    Teacher-Student半监督学习框架
    
    特点:
    - Teacher模型使用EMA更新
    - 弱增强生成伪标签，强增强训练Student
    - 支持检测和分割任务
    - 集成不确定性估计
    """
    
    def __init__(
        self,
        student_model: nn.Module,
        config: Dict,
        device: torch.device = torch.device('cuda')
    ):
        """
        Args:
            student_model: 学生模型
            config: 配置字典
            device: 设备
        """
        super().__init__()
        
        self.student = student_model
        self.config = config
        self.device = device
        
        # 创建Teacher模型（EMA）
        self.teacher = EMAModel(
            student_model,
            decay=config.get('ema_decay', 0.999),
            warmup_epochs=config.get('ema_warmup_epochs', 5),
            total_epochs=config.get('total_epochs', 200),
            decay_schedule=config.get('ema_decay_schedule', 'constant')
        )
        
        # 伪标签生成器
        self.seg_pseudo_generator = SegmentationPseudoLabelGenerator(
            confidence_threshold=config.get('seg_confidence_threshold', 0.8),
            uncertainty_threshold=config.get('seg_uncertainty_threshold', 0.3)
        )
        
        self.det_pseudo_generator = DetectionPseudoLabelGenerator(
            confidence_threshold=config.get('det_confidence_threshold', 0.7),
            nms_threshold=config.get('nms_threshold', 0.5)
        )
        
        # 一致性损失
        self.consistency_loss = ConsistencyLoss(
            loss_type=config.get('consistency_loss_type', 'mse'),
            temperature=config.get('temperature', 1.0)
        )
        
        # 预热调度器
        self.ramp_up_scheduler = RampUpScheduler(
            ramp_up_epochs=config.get('ramp_up_epochs', 20),
            ramp_type=config.get('ramp_up_type', 'sigmoid'),
            final_weight=config.get('unsupervised_weight', 1.0)
        )
        
        # 自适应阈值
        self.adaptive_threshold = AdaptiveThreshold(
            initial_threshold=config.get('initial_threshold', 0.7),
            target_ratio=config.get('target_ratio', 0.5)
        )
        
        # 质量监控
        self.quality_monitor = PseudoLabelQualityMonitor()
        
        # 当前epoch
        self.current_epoch = 0
    
    def set_epoch(self, epoch: int):
        """设置当前epoch"""
        self.current_epoch = epoch
        self.teacher.set_epoch(epoch)
    
    @torch.no_grad()
    def generate_pseudo_labels(
        self,
        unlabeled_weak: torch.Tensor
    ) -> Dict[str, Any]:
        """
        使用Teacher模型生成伪标签
        
        Args:
            unlabeled_weak: 弱增强的无标签数据
            
        Returns:
            伪标签字典
        """
        self.teacher.model.eval()
        
        # Teacher前向传播
        teacher_outputs = self.teacher(unlabeled_weak)
        
        # 分割伪标签
        seg_pseudo = self.seg_pseudo_generator.generate(
            self.teacher.model,
            unlabeled_weak
        )
        
        # 检测伪标签
        det_pseudo = self.det_pseudo_generator.generate(
            self.teacher.model,
            unlabeled_weak
        )
        
        return {
            'segmentation': seg_pseudo,
            'detection': det_pseudo,
            'teacher_outputs': teacher_outputs
        }
    
    def compute_supervised_loss(
        self,
        student_outputs: Dict,
        targets: Dict
    ) -> Dict[str, torch.Tensor]:
        """
        计算有监督损失
        
        Args:
            student_outputs: 学生模型输出
            targets: 真实标签
            
        Returns:
            损失字典
        """
        losses = {}
        
        # 分割损失
        if 'seg_logits' in student_outputs and 'seg_mask' in targets:
            seg_loss = self._compute_seg_loss(
                student_outputs['seg_logits'],
                targets['seg_mask']
            )
            losses['seg_loss'] = seg_loss
        
        # 检测损失
        if 'det_outputs' in student_outputs and 'det_targets' in targets:
            det_loss = self._compute_det_loss(
                student_outputs['det_outputs'],
                targets['det_targets']
            )
            losses['det_loss'] = det_loss
        
        # 分类损失（有病/无病）
        if 'cls_logits' in student_outputs and 'has_lesion' in targets:
            cls_loss = F.binary_cross_entropy_with_logits(
                student_outputs['cls_logits'],
                targets['has_lesion'].float()
            )
            losses['cls_loss'] = cls_loss
        
        return losses
    
    def compute_unsupervised_loss(
        self,
        student_outputs: Dict,
        pseudo_labels: Dict,
        unlabeled_weak: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        计算无监督损失（伪标签损失 + 一致性损失）
        
        Args:
            student_outputs: 学生模型输出（强增强输入）
            pseudo_labels: 伪标签字典
            unlabeled_weak: 弱增强输入（用于一致性）
            
        Returns:
            损失字典
        """
        losses = {}
        
        # 获取无监督损失权重
        unsup_weight = self.ramp_up_scheduler.get_weight(self.current_epoch)
        
        # 分割伪标签损失
        seg_pseudo = pseudo_labels['segmentation']
        if seg_pseudo['valid_mask'].any():
            seg_pseudo_loss = self._compute_pseudo_seg_loss(
                student_outputs['seg_logits'],
                seg_pseudo['pseudo_labels'],
                seg_pseudo['valid_mask'],
                seg_pseudo.get('confidence', None)
            )
            losses['seg_pseudo_loss'] = unsup_weight * seg_pseudo_loss
            
            # 更新质量监控
            self.quality_monitor.update(
                seg_pseudo['confidence'],
                seg_pseudo.get('uncertainty'),
                seg_pseudo['valid_mask']
            )
        
        # 检测伪标签损失
        det_pseudo = pseudo_labels['detection']
        if any(det_pseudo['valid_mask']):
            det_pseudo_loss = self._compute_pseudo_det_loss(
                student_outputs['det_outputs'],
                det_pseudo
            )
            losses['det_pseudo_loss'] = unsup_weight * det_pseudo_loss
        
        # 一致性损失
        if 'teacher_outputs' in pseudo_labels:
            teacher_outputs = pseudo_labels['teacher_outputs']
            
            # 分割一致性
            if 'seg_logits' in student_outputs and 'seg_logits' in teacher_outputs:
                seg_consistency = self.consistency_loss(
                    student_outputs['seg_logits'],
                    teacher_outputs['seg_logits'].detach()
                )
                losses['seg_consistency'] = unsup_weight * seg_consistency
        
        return losses
    
    def _compute_seg_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """计算分割损失（Dice + CE）"""
        # Cross Entropy
        ce_loss = F.cross_entropy(logits, targets, ignore_index=-1)
        
        # Dice Loss
        probs = torch.softmax(logits, dim=1)
        dice_loss = self._dice_loss(probs, targets)
        
        return 0.5 * ce_loss + 0.5 * dice_loss
    
    def _dice_loss(
        self,
        probs: torch.Tensor,
        targets: torch.Tensor,
        smooth: float = 1e-5
    ) -> torch.Tensor:
        """计算Dice损失"""
        num_classes = probs.shape[1]
        
        # One-hot编码
        targets_one_hot = F.one_hot(targets.clamp(min=0), num_classes).permute(0, 4, 1, 2, 3).float()
        
        # 忽略无效标签
        valid_mask = (targets >= 0).unsqueeze(1).expand_as(probs)
        probs = probs * valid_mask
        targets_one_hot = targets_one_hot * valid_mask
        
        # Dice计算
        intersection = (probs * targets_one_hot).sum(dim=(2, 3, 4))
        union = probs.sum(dim=(2, 3, 4)) + targets_one_hot.sum(dim=(2, 3, 4))
        
        dice = (2 * intersection + smooth) / (union + smooth)
        
        return 1 - dice.mean()
    
    def _compute_det_loss(
        self,
        outputs: Dict,
        targets: Dict
    ) -> torch.Tensor:
        """计算检测损失"""
        # 这里简化实现，实际应该使用DINO的损失函数
        loss = torch.tensor(0.0, device=self.device)
        
        if 'cls_loss' in outputs:
            loss += outputs['cls_loss']
        if 'box_loss' in outputs:
            loss += outputs['box_loss']
        if 'giou_loss' in outputs:
            loss += outputs['giou_loss']
        
        return loss
    
    def _compute_pseudo_seg_loss(
        self,
        logits: torch.Tensor,
        pseudo_labels: torch.Tensor,
        valid_mask: torch.Tensor,
        confidence: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """计算分割伪标签损失"""
        # 只在有效位置计算损失
        if confidence is not None:
            # 置信度加权
            weights = confidence * valid_mask.float()
        else:
            weights = valid_mask.float()
        
        # Cross Entropy with weights
        ce_loss = F.cross_entropy(logits, pseudo_labels, reduction='none')
        
        # 应用权重
        weighted_loss = (ce_loss * weights).sum() / (weights.sum() + 1e-8)
        
        return weighted_loss
    
    def _compute_pseudo_det_loss(
        self,
        outputs: Dict,
        pseudo_labels: Dict
    ) -> torch.Tensor:
        """计算检测伪标签损失"""
        # 简化实现
        total_loss = torch.tensor(0.0, device=self.device)
        
        pseudo_boxes = pseudo_labels['pseudo_boxes']
        pseudo_cls = pseudo_labels['pseudo_labels']
        valid_mask = pseudo_labels['valid_mask']
        
        # 只对有效样本计算损失
        valid_count = sum(valid_mask)
        if valid_count == 0:
            return total_loss
        
        # 这里需要根据具体检测器实现
        # 伪代码：将伪标签格式化后送入检测损失函数
        
        return total_loss / valid_count
    
    def forward(
        self,
        labeled_batch: Dict,
        unlabeled_weak: torch.Tensor,
        unlabeled_strong: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播，计算所有损失
        
        Args:
            labeled_batch: 有标签数据批次
            unlabeled_weak: 弱增强的无标签数据
            unlabeled_strong: 强增强的无标签数据
            
        Returns:
            损失字典
        """
        all_losses = {}
        
        # 1. 有监督损失
        student_labeled_outputs = self.student(labeled_batch['image'])
        sup_losses = self.compute_supervised_loss(
            student_labeled_outputs,
            labeled_batch
        )
        all_losses.update({f'sup_{k}': v for k, v in sup_losses.items()})
        
        # 2. 生成伪标签（使用弱增强）
        pseudo_labels = self.generate_pseudo_labels(unlabeled_weak)
        
        # 3. 无监督损失（使用强增强）
        student_unlabeled_outputs = self.student(unlabeled_strong)
        unsup_losses = self.compute_unsupervised_loss(
            student_unlabeled_outputs,
            pseudo_labels,
            unlabeled_weak
        )
        all_losses.update({f'unsup_{k}': v for k, v in unsup_losses.items()})
        
        # 4. 计算总损失
        sup_weight = self.config.get('supervised_weight', 1.0)
        unsup_weight = self.config.get('unsupervised_weight', 1.0)
        
        total_sup_loss = sum(sup_losses.values())
        total_unsup_loss = sum(unsup_losses.values()) if unsup_losses else torch.tensor(0.0)
        
        all_losses['total_loss'] = sup_weight * total_sup_loss + unsup_weight * total_unsup_loss
        
        return all_losses
    
    def update_teacher(self):
        """更新Teacher模型"""
        self.teacher.update(self.student)
    
    def train_step(
        self,
        labeled_batch: Dict,
        unlabeled_weak: torch.Tensor,
        unlabeled_strong: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        单步训练
        
        Args:
            labeled_batch: 有标签数据
            unlabeled_weak: 弱增强无标签数据
            unlabeled_strong: 强增强无标签数据
            optimizer: 优化器
            
        Returns:
            损失值字典
        """
        self.student.train()
        
        # 前向传播
        losses = self.forward(labeled_batch, unlabeled_weak, unlabeled_strong)
        
        # 反向传播
        optimizer.zero_grad()
        losses['total_loss'].backward()
        
        # 梯度裁剪
        if self.config.get('gradient_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(
                self.student.parameters(),
                self.config['gradient_clip']
            )
        
        optimizer.step()
        
        # 更新Teacher
        self.update_teacher()
        
        # 返回损失值
        return {k: v.item() for k, v in losses.items()}
    
    def state_dict(self):
        """返回状态字典"""
        return {
            'student_state_dict': self.student.state_dict(),
            'teacher_state_dict': self.teacher.state_dict(),
            'current_epoch': self.current_epoch,
            'adaptive_threshold': self.adaptive_threshold.threshold
        }
    
    def load_state_dict(self, state_dict: Dict):
        """加载状态字典"""
        self.student.load_state_dict(state_dict['student_state_dict'])
        self.teacher.load_state_dict(state_dict['teacher_state_dict'])
        self.current_epoch = state_dict['current_epoch']
        self.adaptive_threshold.threshold = state_dict['adaptive_threshold']


class MultiTaskTeacherStudent(TeacherStudentFramework):
    """
    多任务Teacher-Student框架
    
    针对Stage2的检测+分类+分割联合任务
    """
    
    def __init__(
        self,
        student_model: nn.Module,
        config: Dict,
        device: torch.device = torch.device('cuda')
    ):
        super().__init__(student_model, config, device)
        
        # 任务特定的损失权重
        self.task_weights = config.get('task_weights', {
            'detection': 1.0,
            'classification': 1.0,
            'segmentation': 1.0
        })
        
        # 不确定性加权（可学习）
        self.use_uncertainty_weighting = config.get('use_uncertainty_weighting', True)
        if self.use_uncertainty_weighting:
            # 可学习的任务不确定性参数
            self.log_vars = nn.Parameter(torch.zeros(3))
    
    def compute_multitask_loss(
        self,
        losses: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        计算多任务加权损失
        
        Args:
            losses: 各任务损失字典
            
        Returns:
            加权总损失
        """
        if self.use_uncertainty_weighting:
            # 使用不确定性加权
            total_loss = 0
            task_names = ['detection', 'classification', 'segmentation']
            
            for i, task in enumerate(task_names):
                task_loss = sum(v for k, v in losses.items() if task[:3] in k.lower())
                if task_loss > 0:
                    # 不确定性加权公式: L_i / (2 * sigma_i^2) + log(sigma_i)
                    precision = torch.exp(-self.log_vars[i])
                    total_loss += precision * task_loss + self.log_vars[i]
            
            return total_loss
        else:
            # 使用固定权重
            total_loss = 0
            for task, weight in self.task_weights.items():
                task_loss = sum(v for k, v in losses.items() if task[:3] in k.lower())
                total_loss += weight * task_loss
            
            return total_loss
