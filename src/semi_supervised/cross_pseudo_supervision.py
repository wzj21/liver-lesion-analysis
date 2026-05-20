"""
Cross Pseudo Supervision (CPS) 半监督分割框架
Cross Pseudo Supervision for Semi-Supervised Segmentation

用于Stage1肝脏分割任务的半监督学习
论文: Semi-Supervised Semantic Segmentation Needs Strong, Varied Perturbations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import copy

from .base import (
    RampUpScheduler,
    AdaptiveThreshold,
    PseudoLabelQualityMonitor
)


class CrossPseudoSupervision(nn.Module):
    """
    Cross Pseudo Supervision (CPS) 半监督分割框架
    
    核心思想:
    - 训练两个结构相同但初始化不同的网络
    - 网络A的预测作为网络B的伪标签
    - 网络B的预测作为网络A的伪标签
    - 互相监督减少确认偏差
    """
    
    def __init__(
        self,
        network_a: nn.Module,
        network_b: nn.Module,
        num_classes: int = 2,
        config: Dict = None,
        device: torch.device = torch.device('cuda')
    ):
        super().__init__()
        
        self.network_a = network_a
        self.network_b = network_b
        self.num_classes = num_classes
        self.config = config or {}
        self.device = device
        
        # CPS配置
        self.cps_weight = config.get('cps_weight', 1.0)
        self.pseudo_threshold = config.get('pseudo_threshold', 0.8)
        self.use_uncertainty_weighting = config.get('use_uncertainty_weighting', True)
        self.sup_weight = config.get('supervised_weight', 1.0)
        
        # 预热调度
        self.ramp_up_scheduler = RampUpScheduler(
            ramp_up_epochs=config.get('ramp_up_epochs', 20),
            ramp_type=config.get('ramp_up_type', 'sigmoid'),
            final_weight=self.cps_weight
        )
        
        # 边界感知
        self.boundary_aware = config.get('boundary_aware', True)
        
        # 质量监控
        self.quality_monitor_a = PseudoLabelQualityMonitor()
        self.quality_monitor_b = PseudoLabelQualityMonitor()
        
        self.current_epoch = 0
        self.dice_weight = config.get('dice_weight', 0.5)
        self.ce_weight = config.get('ce_weight', 0.5)
    
    def set_epoch(self, epoch: int):
        self.current_epoch = epoch
    
    def _get_predictions(self, network: nn.Module, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = network(inputs)
        
        if isinstance(outputs, dict):
            logits = outputs.get('logits', outputs.get('pred'))
            uncertainty = outputs.get('uncertainty', None)
        else:
            logits = outputs
            uncertainty = None
        
        probs = torch.softmax(logits, dim=1)
        confidence, predictions = probs.max(dim=1)
        
        if uncertainty is None:
            uncertainty = 1 - confidence
        
        return {
            'logits': logits,
            'probs': probs,
            'predictions': predictions,
            'confidence': confidence,
            'uncertainty': uncertainty
        }
    
    def _generate_pseudo_labels(self, predictions: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        confidence = predictions['confidence']
        pred_labels = predictions['predictions']
        uncertainty = predictions['uncertainty']
        
        valid_mask = confidence >= self.pseudo_threshold
        
        if self.use_uncertainty_weighting:
            valid_mask = valid_mask & (uncertainty <= (1 - self.pseudo_threshold))
        
        pseudo_labels = pred_labels.clone()
        pseudo_labels[~valid_mask] = -1
        
        return {
            'pseudo_labels': pseudo_labels,
            'valid_mask': valid_mask,
            'confidence': confidence,
            'weights': confidence * valid_mask.float()
        }
    
    def _compute_seg_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, ignore_index=-1)
        probs = torch.softmax(logits, dim=1)
        dice_loss = self._dice_loss(probs, targets)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss
    
    def _dice_loss(self, probs: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
        valid_mask = targets >= 0
        targets_clamped = targets.clamp(min=0)
        targets_one_hot = F.one_hot(targets_clamped, self.num_classes).float()
        targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3)
        
        valid_mask = valid_mask.unsqueeze(1).expand_as(probs)
        probs = probs * valid_mask
        targets_one_hot = targets_one_hot * valid_mask
        
        intersection = (probs * targets_one_hot).sum(dim=(2, 3, 4))
        union = probs.sum(dim=(2, 3, 4)) + targets_one_hot.sum(dim=(2, 3, 4))
        dice = (2 * intersection + smooth) / (union + smooth)
        
        return 1 - dice.mean()
    
    def _compute_pseudo_loss(self, logits: torch.Tensor, pseudo_labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, pseudo_labels, ignore_index=-1, reduction='none')
        weighted_ce = (ce_loss * weights).sum() / (weights.sum() + 1e-8)
        return weighted_ce
    
    def compute_supervised_loss(self, logits_a: torch.Tensor, logits_b: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        losses = {}
        loss_a = self._compute_seg_loss(logits_a, targets)
        losses['sup_loss_a'] = loss_a
        loss_b = self._compute_seg_loss(logits_b, targets)
        losses['sup_loss_b'] = loss_b
        losses['total_sup_loss'] = self.sup_weight * (loss_a + loss_b)
        return losses
    
    def compute_cps_loss(self, pred_a: Dict[str, torch.Tensor], pred_b: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        losses = {}
        cps_weight = self.ramp_up_scheduler.get_weight(self.current_epoch)
        
        pseudo_a = self._generate_pseudo_labels(pred_a)
        cps_loss_b = self._compute_pseudo_loss(
            pred_b['logits'], pseudo_a['pseudo_labels'],
            pseudo_a['weights'] if self.use_uncertainty_weighting else pseudo_a['valid_mask'].float()
        )
        losses['cps_loss_b'] = cps_weight * cps_loss_b
        
        pseudo_b = self._generate_pseudo_labels(pred_b)
        cps_loss_a = self._compute_pseudo_loss(
            pred_a['logits'], pseudo_b['pseudo_labels'],
            pseudo_b['weights'] if self.use_uncertainty_weighting else pseudo_b['valid_mask'].float()
        )
        losses['cps_loss_a'] = cps_weight * cps_loss_a
        losses['total_cps_loss'] = losses['cps_loss_a'] + losses['cps_loss_b']
        
        self.quality_monitor_a.update(pseudo_a['confidence'], pred_a['uncertainty'], pseudo_a['valid_mask'])
        self.quality_monitor_b.update(pseudo_b['confidence'], pred_b['uncertainty'], pseudo_b['valid_mask'])
        
        with torch.no_grad():
            losses['pseudo_ratio_a'] = pseudo_a['valid_mask'].float().mean()
            losses['pseudo_ratio_b'] = pseudo_b['valid_mask'].float().mean()
        
        return losses
    
    def forward(self, labeled_images: torch.Tensor, labeled_targets: torch.Tensor, unlabeled_images: torch.Tensor) -> Dict[str, torch.Tensor]:
        all_losses = {}
        
        self.network_a.train()
        self.network_b.train()
        
        logits_a_labeled = self.network_a(labeled_images)
        logits_b_labeled = self.network_b(labeled_images)
        
        if isinstance(logits_a_labeled, dict):
            logits_a_labeled = logits_a_labeled.get('logits', logits_a_labeled.get('pred'))
        if isinstance(logits_b_labeled, dict):
            logits_b_labeled = logits_b_labeled.get('logits', logits_b_labeled.get('pred'))
        
        sup_losses = self.compute_supervised_loss(logits_a_labeled, logits_b_labeled, labeled_targets)
        all_losses.update(sup_losses)
        
        with torch.no_grad():
            pred_a_unlabeled = self._get_predictions(self.network_a, unlabeled_images)
            pred_b_unlabeled = self._get_predictions(self.network_b, unlabeled_images)
        
        self.network_a.train()
        self.network_b.train()
        
        logits_a_unlabeled = self.network_a(unlabeled_images)
        logits_b_unlabeled = self.network_b(unlabeled_images)
        
        if isinstance(logits_a_unlabeled, dict):
            pred_a_unlabeled['logits'] = logits_a_unlabeled.get('logits', logits_a_unlabeled.get('pred'))
        else:
            pred_a_unlabeled['logits'] = logits_a_unlabeled
            
        if isinstance(logits_b_unlabeled, dict):
            pred_b_unlabeled['logits'] = logits_b_unlabeled.get('logits', logits_b_unlabeled.get('pred'))
        else:
            pred_b_unlabeled['logits'] = logits_b_unlabeled
        
        cps_losses = self.compute_cps_loss(pred_a_unlabeled, pred_b_unlabeled)
        all_losses.update(cps_losses)
        
        all_losses['total_loss'] = all_losses['total_sup_loss'] + all_losses['total_cps_loss']
        
        return all_losses
    
    def train_step(self, labeled_images, labeled_targets, unlabeled_images, optimizer_a, optimizer_b) -> Dict[str, float]:
        losses = self.forward(labeled_images, labeled_targets, unlabeled_images)
        
        optimizer_a.zero_grad()
        optimizer_b.zero_grad()
        losses['total_loss'].backward()
        
        if self.config.get('gradient_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(self.network_a.parameters(), self.config['gradient_clip'])
            torch.nn.utils.clip_grad_norm_(self.network_b.parameters(), self.config['gradient_clip'])
        
        optimizer_a.step()
        optimizer_b.step()
        
        return {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}
    
    def get_ensemble_prediction(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.network_a.eval()
        self.network_b.eval()
        
        with torch.no_grad():
            pred_a = self._get_predictions(self.network_a, inputs)
            pred_b = self._get_predictions(self.network_b, inputs)
            
            ensemble_probs = (pred_a['probs'] + pred_b['probs']) / 2
            confidence, predictions = ensemble_probs.max(dim=1)
            uncertainty = (pred_a['uncertainty'] + pred_b['uncertainty']) / 2
            agreement = (pred_a['predictions'] == pred_b['predictions']).float()
        
        return {
            'predictions': predictions,
            'probs': ensemble_probs,
            'confidence': confidence,
            'uncertainty': uncertainty,
            'agreement': agreement
        }
    
    def state_dict(self):
        return {
            'network_a_state_dict': self.network_a.state_dict(),
            'network_b_state_dict': self.network_b.state_dict(),
            'current_epoch': self.current_epoch
        }
    
    def load_state_dict(self, state_dict: Dict):
        self.network_a.load_state_dict(state_dict['network_a_state_dict'])
        self.network_b.load_state_dict(state_dict['network_b_state_dict'])
        self.current_epoch = state_dict['current_epoch']


class MeanTeacher(nn.Module):
    """Mean Teacher半监督学习框架"""
    
    def __init__(self, student_model: nn.Module, num_classes: int = 2, config: Dict = None, device: torch.device = torch.device('cuda')):
        super().__init__()
        
        self.student = student_model
        self.num_classes = num_classes
        self.config = config or {}
        self.device = device
        
        self.teacher = copy.deepcopy(student_model)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        self.ema_decay = config.get('ema_decay', 0.999)
        self.consistency_weight = config.get('consistency_weight', 1.0)
        
        self.ramp_up_scheduler = RampUpScheduler(
            ramp_up_epochs=config.get('ramp_up_epochs', 20),
            ramp_type='sigmoid',
            final_weight=self.consistency_weight
        )
        
        self.current_epoch = 0
    
    def set_epoch(self, epoch: int):
        self.current_epoch = epoch
    
    @torch.no_grad()
    def update_teacher(self):
        student_params = dict(self.student.named_parameters())
        teacher_params = dict(self.teacher.named_parameters())
        
        for name, teacher_param in teacher_params.items():
            if name in student_params:
                student_param = student_params[name]
                teacher_param.data.mul_(self.ema_decay).add_(student_param.data, alpha=1 - self.ema_decay)
    
    def forward(self, labeled_images, labeled_targets, unlabeled_images) -> Dict[str, torch.Tensor]:
        losses = {}
        
        self.student.train()
        student_labeled = self.student(labeled_images)
        
        if isinstance(student_labeled, dict):
            logits = student_labeled.get('logits', student_labeled.get('pred'))
        else:
            logits = student_labeled
        
        sup_loss = F.cross_entropy(logits, labeled_targets, ignore_index=-1)
        losses['supervised_loss'] = sup_loss
        
        self.teacher.eval()
        with torch.no_grad():
            teacher_unlabeled = self.teacher(unlabeled_images)
            if isinstance(teacher_unlabeled, dict):
                teacher_logits = teacher_unlabeled.get('logits', teacher_unlabeled.get('pred'))
            else:
                teacher_logits = teacher_unlabeled
        
        self.student.train()
        student_unlabeled = self.student(unlabeled_images)
        if isinstance(student_unlabeled, dict):
            student_logits = student_unlabeled.get('logits', student_unlabeled.get('pred'))
        else:
            student_logits = student_unlabeled
        
        consistency_loss = F.mse_loss(
            torch.softmax(student_logits, dim=1),
            torch.softmax(teacher_logits, dim=1)
        )
        
        weight = self.ramp_up_scheduler.get_weight(self.current_epoch)
        losses['consistency_loss'] = weight * consistency_loss
        losses['total_loss'] = losses['supervised_loss'] + losses['consistency_loss']
        
        return losses
    
    def train_step(self, labeled_images, labeled_targets, unlabeled_images, optimizer) -> Dict[str, float]:
        losses = self.forward(labeled_images, labeled_targets, unlabeled_images)
        
        optimizer.zero_grad()
        losses['total_loss'].backward()
        optimizer.step()
        
        self.update_teacher()
        
        return {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}
