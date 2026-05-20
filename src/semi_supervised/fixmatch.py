"""
FixMatch半监督学习框架
FixMatch Semi-Supervised Learning for Classification

用于Stage3分类任务的半监督学习
论文: FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .base import (
    EMAModel,
    ClassificationPseudoLabelGenerator,
    RampUpScheduler,
    AdaptiveThreshold,
    PseudoLabelQualityMonitor
)


class FixMatch(nn.Module):
    """
    FixMatch半监督分类框架
    
    核心思想:
    1. 使用弱增强数据生成伪标签
    2. 只保留高置信度的伪标签
    3. 使用强增强数据和伪标签训练模型
    
    针对肝脏病灶分类任务的优化:
    - 集成Evidential不确定性估计
    - 分布对齐处理类别不平衡
    - 自适应阈值调整
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 4,
        config: Dict = None,
        device: torch.device = torch.device('cuda')
    ):
        """
        Args:
            model: 分类模型
            num_classes: 类别数（良性、恶性、囊型包虫、泡型包虫）
            config: 配置字典
            device: 设备
        """
        super().__init__()
        
        self.model = model
        self.num_classes = num_classes
        self.config = config or {}
        self.device = device
        
        # 伪标签配置
        self.confidence_threshold = config.get('confidence_threshold', 0.95)
        self.use_uncertainty_filter = config.get('use_uncertainty_filter', True)
        self.uncertainty_threshold = config.get('uncertainty_threshold', 0.2)
        
        # 伪标签生成器
        self.pseudo_generator = ClassificationPseudoLabelGenerator(
            confidence_threshold=self.confidence_threshold,
            uncertainty_threshold=self.uncertainty_threshold,
            num_classes=num_classes
        )
        
        # 损失权重
        self.supervised_weight = config.get('supervised_weight', 1.0)
        self.unsupervised_weight = config.get('unsupervised_weight', 1.0)
        
        # 预热调度
        self.ramp_up_scheduler = RampUpScheduler(
            ramp_up_epochs=config.get('ramp_up_epochs', 20),
            ramp_type=config.get('ramp_up_type', 'sigmoid'),
            final_weight=self.unsupervised_weight
        )
        
        # 分布对齐
        self.use_distribution_alignment = config.get('distribution_alignment', True)
        self.da_temperature = config.get('da_temperature', 0.5)
        
        # 类别分布估计（用于分布对齐）
        self.register_buffer(
            'class_distribution',
            torch.ones(num_classes) / num_classes
        )
        self.distribution_momentum = config.get('distribution_momentum', 0.999)
        
        # 自适应阈值
        self.use_adaptive_threshold = config.get('use_adaptive_threshold', True)
        self.adaptive_threshold = AdaptiveThreshold(
            initial_threshold=self.confidence_threshold,
            target_ratio=config.get('target_ratio', 0.5),
            min_threshold=config.get('min_threshold', 0.7),
            max_threshold=config.get('max_threshold', 0.98)
        )
        
        # 质量监控
        self.quality_monitor = PseudoLabelQualityMonitor()
        
        # 当前epoch
        self.current_epoch = 0
        
        # 类别权重（处理不平衡）
        self.class_weights = config.get('class_weights', None)
        if self.class_weights is not None:
            self.class_weights = torch.tensor(self.class_weights).to(device)
    
    def set_epoch(self, epoch: int):
        """设置当前epoch"""
        self.current_epoch = epoch
    
    @torch.no_grad()
    def generate_pseudo_labels(
        self,
        unlabeled_weak: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        生成伪标签
        
        Args:
            unlabeled_weak: 弱增强的无标签数据
            
        Returns:
            伪标签字典
        """
        self.model.eval()
        
        # 模型预测
        outputs = self.model(unlabeled_weak)
        
        # 处理输出
        if isinstance(outputs, dict):
            logits = outputs.get('logits', outputs.get('pred'))
            uncertainty = outputs.get('uncertainty', None)
        else:
            logits = outputs
            uncertainty = None
        
        # 计算概率
        probs = torch.softmax(logits, dim=1)
        
        # 分布对齐
        if self.use_distribution_alignment:
            probs = self._apply_distribution_alignment(probs)
        
        # 获取伪标签和置信度
        confidence, pseudo_labels = probs.max(dim=1)
        
        # 如果没有不确定性输出，使用熵估计
        if uncertainty is None and self.use_uncertainty_filter:
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            max_entropy = np.log(self.num_classes)
            uncertainty = entropy / max_entropy
        
        # 获取当前阈值
        if self.use_adaptive_threshold:
            threshold = self.adaptive_threshold.get_threshold()
        else:
            threshold = self.confidence_threshold
        
        # 过滤低置信度样本
        valid_mask = confidence >= threshold
        
        # 过滤高不确定性样本
        if self.use_uncertainty_filter and uncertainty is not None:
            valid_mask = valid_mask & (uncertainty <= self.uncertainty_threshold)
        
        # 更新分布估计
        if valid_mask.any():
            self._update_class_distribution(pseudo_labels[valid_mask])
        
        # 更新自适应阈值
        if self.use_adaptive_threshold:
            usage_ratio = valid_mask.float().mean().item()
            self.adaptive_threshold.update(usage_ratio)
        
        # 更新质量监控
        self.quality_monitor.update(
            confidence,
            uncertainty,
            valid_mask
        )
        
        result = {
            'pseudo_labels': pseudo_labels,
            'confidence': confidence,
            'valid_mask': valid_mask,
            'probs': probs
        }
        
        if uncertainty is not None:
            result['uncertainty'] = uncertainty
        
        return result
    
    def _apply_distribution_alignment(
        self,
        probs: torch.Tensor
    ) -> torch.Tensor:
        """
        应用分布对齐
        
        防止模型对某些类别过度自信
        
        Args:
            probs: 预测概率 [B, C]
            
        Returns:
            对齐后的概率
        """
        # 使用温度缩放
        probs_temp = probs ** (1 / self.da_temperature)
        
        # 除以类别分布
        probs_aligned = probs_temp / (self.class_distribution.unsqueeze(0) + 1e-8)
        
        # 重新归一化
        probs_aligned = probs_aligned / probs_aligned.sum(dim=1, keepdim=True)
        
        return probs_aligned
    
    @torch.no_grad()
    def _update_class_distribution(self, pseudo_labels: torch.Tensor):
        """
        更新类别分布估计
        
        Args:
            pseudo_labels: 有效的伪标签
        """
        # 计算当前批次的类别分布
        batch_distribution = torch.zeros(self.num_classes, device=self.device)
        for c in range(self.num_classes):
            batch_distribution[c] = (pseudo_labels == c).float().sum()
        
        batch_distribution = batch_distribution / (batch_distribution.sum() + 1e-8)
        
        # EMA更新
        self.class_distribution = (
            self.distribution_momentum * self.class_distribution +
            (1 - self.distribution_momentum) * batch_distribution
        )
    
    def compute_supervised_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        计算有监督损失
        
        Args:
            logits: 模型输出
            targets: 真实标签
            
        Returns:
            损失值
        """
        if self.class_weights is not None:
            loss = F.cross_entropy(logits, targets, weight=self.class_weights)
        else:
            loss = F.cross_entropy(logits, targets)
        
        return loss
    
    def compute_unsupervised_loss(
        self,
        logits: torch.Tensor,
        pseudo_labels: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        计算无监督损失（伪标签损失）
        
        Args:
            logits: 强增强数据的模型输出
            pseudo_labels: 伪标签字典
            
        Returns:
            损失值
        """
        labels = pseudo_labels['pseudo_labels']
        valid_mask = pseudo_labels['valid_mask']
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=self.device)
        
        # 只对有效样本计算损失
        loss = F.cross_entropy(logits, labels, reduction='none')
        
        # 应用掩码
        masked_loss = (loss * valid_mask.float()).sum() / (valid_mask.sum() + 1e-8)
        
        # 应用预热权重
        unsup_weight = self.ramp_up_scheduler.get_weight(self.current_epoch)
        
        return unsup_weight * masked_loss
    
    def forward(
        self,
        labeled_images: torch.Tensor,
        labeled_targets: torch.Tensor,
        unlabeled_weak: torch.Tensor,
        unlabeled_strong: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            labeled_images: 有标签图像
            labeled_targets: 标签
            unlabeled_weak: 弱增强无标签图像
            unlabeled_strong: 强增强无标签图像
            
        Returns:
            损失字典
        """
        losses = {}
        
        # 1. 有监督损失
        self.model.train()
        labeled_outputs = self.model(labeled_images)
        
        if isinstance(labeled_outputs, dict):
            labeled_logits = labeled_outputs.get('logits', labeled_outputs.get('pred'))
        else:
            labeled_logits = labeled_outputs
        
        sup_loss = self.compute_supervised_loss(labeled_logits, labeled_targets)
        losses['supervised_loss'] = self.supervised_weight * sup_loss
        
        # 2. 生成伪标签（使用弱增强）
        pseudo_labels = self.generate_pseudo_labels(unlabeled_weak)
        
        # 3. 无监督损失（使用强增强）
        self.model.train()
        unlabeled_outputs = self.model(unlabeled_strong)
        
        if isinstance(unlabeled_outputs, dict):
            unlabeled_logits = unlabeled_outputs.get('logits', unlabeled_outputs.get('pred'))
        else:
            unlabeled_logits = unlabeled_outputs
        
        unsup_loss = self.compute_unsupervised_loss(unlabeled_logits, pseudo_labels)
        losses['unsupervised_loss'] = unsup_loss
        
        # 4. 总损失
        losses['total_loss'] = losses['supervised_loss'] + losses['unsupervised_loss']
        
        # 5. 记录统计信息
        with torch.no_grad():
            losses['pseudo_label_ratio'] = pseudo_labels['valid_mask'].float().mean()
            losses['pseudo_confidence_mean'] = pseudo_labels['confidence'].mean()
            if 'uncertainty' in pseudo_labels:
                losses['pseudo_uncertainty_mean'] = pseudo_labels['uncertainty'].mean()
        
        return losses
    
    def train_step(
        self,
        labeled_images: torch.Tensor,
        labeled_targets: torch.Tensor,
        unlabeled_weak: torch.Tensor,
        unlabeled_strong: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        单步训练
        
        Args:
            labeled_images: 有标签图像
            labeled_targets: 标签
            unlabeled_weak: 弱增强无标签图像
            unlabeled_strong: 强增强无标签图像
            optimizer: 优化器
            
        Returns:
            损失字典
        """
        # 前向传播
        losses = self.forward(
            labeled_images,
            labeled_targets,
            unlabeled_weak,
            unlabeled_strong
        )
        
        # 反向传播
        optimizer.zero_grad()
        losses['total_loss'].backward()
        
        # 梯度裁剪
        if self.config.get('gradient_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['gradient_clip']
            )
        
        optimizer.step()
        
        # 转换为Python数值
        return {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}


class FlexMatch(FixMatch):
    """
    FlexMatch: 自适应阈值的FixMatch变体
    
    特点:
    - 每个类别使用不同的阈值
    - 根据学习状态自动调整阈值
    - 更好地处理类别不平衡
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 4,
        config: Dict = None,
        device: torch.device = torch.device('cuda')
    ):
        super().__init__(model, num_classes, config, device)
        
        # 每个类别的阈值
        self.register_buffer(
            'class_thresholds',
            torch.ones(num_classes) * self.confidence_threshold
        )
        
        # 每个类别的学习状态估计
        self.register_buffer(
            'class_learning_status',
            torch.zeros(num_classes)
        )
        
        # 阈值调整参数
        self.threshold_warmup = config.get('threshold_warmup', 0.5)
        self.flex_rate = config.get('flex_rate', 0.95)
    
    @torch.no_grad()
    def update_class_thresholds(self, probs: torch.Tensor, valid_mask: torch.Tensor):
        """
        更新每个类别的阈值
        
        Args:
            probs: 预测概率
            valid_mask: 有效掩码
        """
        confidence, predictions = probs.max(dim=1)
        
        for c in range(self.num_classes):
            class_mask = predictions == c
            
            if class_mask.any():
                class_confidence = confidence[class_mask]
                
                # 计算该类别的学习状态（使用平均置信度）
                avg_confidence = class_confidence.mean()
                
                # EMA更新学习状态
                self.class_learning_status[c] = (
                    self.flex_rate * self.class_learning_status[c] +
                    (1 - self.flex_rate) * avg_confidence
                )
                
                # 根据学习状态调整阈值
                # 学习状态好 -> 提高阈值
                # 学习状态差 -> 降低阈值
                self.class_thresholds[c] = (
                    self.threshold_warmup +
                    (self.confidence_threshold - self.threshold_warmup) *
                    self.class_learning_status[c]
                )
    
    @torch.no_grad()
    def generate_pseudo_labels(
        self,
        unlabeled_weak: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        使用类别自适应阈值生成伪标签
        """
        self.model.eval()
        
        outputs = self.model(unlabeled_weak)
        
        if isinstance(outputs, dict):
            logits = outputs.get('logits', outputs.get('pred'))
            uncertainty = outputs.get('uncertainty', None)
        else:
            logits = outputs
            uncertainty = None
        
        probs = torch.softmax(logits, dim=1)
        
        if self.use_distribution_alignment:
            probs = self._apply_distribution_alignment(probs)
        
        confidence, pseudo_labels = probs.max(dim=1)
        
        # 使用类别特定阈值
        thresholds = self.class_thresholds[pseudo_labels]
        valid_mask = confidence >= thresholds
        
        # 不确定性过滤
        if uncertainty is None and self.use_uncertainty_filter:
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            max_entropy = np.log(self.num_classes)
            uncertainty = entropy / max_entropy
        
        if self.use_uncertainty_filter and uncertainty is not None:
            valid_mask = valid_mask & (uncertainty <= self.uncertainty_threshold)
        
        # 更新类别阈值
        self.update_class_thresholds(probs, valid_mask)
        
        # 更新分布
        if valid_mask.any():
            self._update_class_distribution(pseudo_labels[valid_mask])
        
        self.quality_monitor.update(
            confidence,
            uncertainty,
            valid_mask
        )
        
        result = {
            'pseudo_labels': pseudo_labels,
            'confidence': confidence,
            'valid_mask': valid_mask,
            'probs': probs,
            'class_thresholds': self.class_thresholds.clone()
        }
        
        if uncertainty is not None:
            result['uncertainty'] = uncertainty
        
        return result


class EvidentialFixMatch(FixMatch):
    """
    结合Evidential Learning的FixMatch
    
    使用Evidential不确定性进行更精确的伪标签过滤
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 4,
        config: Dict = None,
        device: torch.device = torch.device('cuda')
    ):
        super().__init__(model, num_classes, config, device)
        
        # Evidential参数
        self.annealing_epochs = config.get('annealing_epochs', 10)
        self.evidence_threshold = config.get('evidence_threshold', 5.0)
    
    @torch.no_grad()
    def generate_pseudo_labels(
        self,
        unlabeled_weak: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        使用Evidential不确定性生成伪标签
        """
        self.model.eval()
        
        outputs = self.model(unlabeled_weak)
        
        # 期望模型输出Dirichlet参数
        if isinstance(outputs, dict):
            if 'alpha' in outputs:
                # Evidential输出
                alpha = outputs['alpha']
                S = alpha.sum(dim=1, keepdim=True)  # Dirichlet强度
                probs = alpha / S  # 期望概率
                
                # Evidential不确定性
                uncertainty = self.num_classes / S.squeeze()
                
                # 证据量
                evidence = S.squeeze() - self.num_classes
            else:
                logits = outputs.get('logits', outputs.get('pred'))
                probs = torch.softmax(logits, dim=1)
                
                # 用熵估计不确定性
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                uncertainty = entropy / np.log(self.num_classes)
                evidence = None
        else:
            probs = torch.softmax(outputs, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            uncertainty = entropy / np.log(self.num_classes)
            evidence = None
        
        if self.use_distribution_alignment:
            probs = self._apply_distribution_alignment(probs)
        
        confidence, pseudo_labels = probs.max(dim=1)
        
        # 获取阈值
        if self.use_adaptive_threshold:
            threshold = self.adaptive_threshold.get_threshold()
        else:
            threshold = self.confidence_threshold
        
        # 置信度过滤
        valid_mask = confidence >= threshold
        
        # Evidential不确定性过滤
        valid_mask = valid_mask & (uncertainty <= self.uncertainty_threshold)
        
        # 证据量过滤（如果有）
        if evidence is not None:
            valid_mask = valid_mask & (evidence >= self.evidence_threshold)
        
        # 更新
        if self.use_adaptive_threshold:
            usage_ratio = valid_mask.float().mean().item()
            self.adaptive_threshold.update(usage_ratio)
        
        if valid_mask.any():
            self._update_class_distribution(pseudo_labels[valid_mask])
        
        self.quality_monitor.update(
            confidence,
            uncertainty,
            valid_mask
        )
        
        result = {
            'pseudo_labels': pseudo_labels,
            'confidence': confidence,
            'uncertainty': uncertainty,
            'valid_mask': valid_mask,
            'probs': probs
        }
        
        if evidence is not None:
            result['evidence'] = evidence
        
        return result
    
    def compute_supervised_loss(
        self,
        outputs: Any,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        计算Evidential监督损失
        """
        if isinstance(outputs, dict) and 'alpha' in outputs:
            alpha = outputs['alpha']
            return self._evidential_loss(alpha, targets)
        else:
            if isinstance(outputs, dict):
                logits = outputs.get('logits', outputs.get('pred'))
            else:
                logits = outputs
            return super().compute_supervised_loss(logits, targets)
    
    def _evidential_loss(
        self,
        alpha: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        计算Evidential损失
        
        Args:
            alpha: Dirichlet参数 [B, C]
            targets: 标签 [B]
            
        Returns:
            损失值
        """
        S = alpha.sum(dim=1, keepdim=True)
        
        # One-hot编码
        targets_one_hot = F.one_hot(targets, self.num_classes).float()
        
        # Type II Maximum Likelihood Loss
        loss_mll = torch.sum(
            targets_one_hot * (torch.digamma(S) - torch.digamma(alpha)),
            dim=1
        ).mean()
        
        # KL散度正则化
        annealing_coef = min(1.0, self.current_epoch / self.annealing_epochs)
        
        alpha_tilde = targets_one_hot + (1 - targets_one_hot) * alpha
        S_tilde = alpha_tilde.sum(dim=1, keepdim=True)
        
        kl_loss = torch.sum(
            torch.lgamma(S_tilde) - torch.lgamma(torch.tensor(self.num_classes).float()) -
            torch.sum(torch.lgamma(alpha_tilde), dim=1, keepdim=True) +
            torch.sum((alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde)), dim=1, keepdim=True),
            dim=1
        ).mean()
        
        return loss_mll + annealing_coef * kl_loss
