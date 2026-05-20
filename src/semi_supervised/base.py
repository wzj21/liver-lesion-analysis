"""
半监督学习框架 - 基础模块
Semi-Supervised Learning Framework - Base Module

包含:
- EMA (Exponential Moving Average) 模型更新
- 伪标签生成基类
- 一致性损失
- 数据增强包装器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
import copy


class EMAModel:
    """
    指数移动平均模型 (Exponential Moving Average Model)
    
    用于Teacher-Student框架中的Teacher模型更新
    """
    
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        warmup_epochs: int = 5,
        total_epochs: int = 200,
        decay_schedule: str = "constant"  # constant, linear, cosine
    ):
        """
        Args:
            model: 需要进行EMA的模型
            decay: EMA衰减系数
            warmup_epochs: 预热轮数（预热期间decay从0.5线性增加到目标值）
            total_epochs: 总训练轮数
            decay_schedule: 衰减调度策略
        """
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.decay = decay
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.decay_schedule = decay_schedule
        self.current_epoch = 0
        
        # 冻结EMA模型参数
        for param in self.model.parameters():
            param.requires_grad = False
    
    def _get_current_decay(self) -> float:
        """根据当前epoch获取衰减系数"""
        if self.current_epoch < self.warmup_epochs:
            # 预热期间线性增加
            return 0.5 + (self.decay - 0.5) * (self.current_epoch / self.warmup_epochs)
        
        if self.decay_schedule == "constant":
            return self.decay
        elif self.decay_schedule == "linear":
            # 线性增加到0.9999
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            return self.decay + (0.9999 - self.decay) * progress
        elif self.decay_schedule == "cosine":
            # 余弦调度
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            return self.decay + (0.9999 - self.decay) * (1 - np.cos(np.pi * progress)) / 2
        else:
            return self.decay
    
    @torch.no_grad()
    def update(self, student_model: nn.Module):
        """
        更新EMA模型参数
        
        Args:
            student_model: 学生模型（正在训练的模型）
        """
        decay = self._get_current_decay()
        
        student_params = dict(student_model.named_parameters())
        ema_params = dict(self.model.named_parameters())
        
        for name, ema_param in ema_params.items():
            if name in student_params:
                student_param = student_params[name]
                ema_param.data.mul_(decay).add_(student_param.data, alpha=1 - decay)
        
        # 更新BN层的running_mean和running_var
        student_buffers = dict(student_model.named_buffers())
        ema_buffers = dict(self.model.named_buffers())
        
        for name, ema_buffer in ema_buffers.items():
            if name in student_buffers and 'running' in name:
                student_buffer = student_buffers[name]
                ema_buffer.data.mul_(decay).add_(student_buffer.data, alpha=1 - decay)
    
    def set_epoch(self, epoch: int):
        """设置当前epoch"""
        self.current_epoch = epoch
    
    def forward(self, *args, **kwargs):
        """前向传播"""
        return self.model(*args, **kwargs)
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def state_dict(self):
        """返回状态字典"""
        return {
            'model_state_dict': self.model.state_dict(),
            'decay': self.decay,
            'current_epoch': self.current_epoch
        }
    
    def load_state_dict(self, state_dict: Dict):
        """加载状态字典"""
        self.model.load_state_dict(state_dict['model_state_dict'])
        self.decay = state_dict['decay']
        self.current_epoch = state_dict['current_epoch']


class PseudoLabelGenerator(ABC):
    """伪标签生成器基类"""
    
    def __init__(
        self,
        confidence_threshold: float = 0.7,
        uncertainty_threshold: float = 0.3,
        use_soft_labels: bool = False
    ):
        """
        Args:
            confidence_threshold: 置信度阈值
            uncertainty_threshold: 不确定性阈值
            use_soft_labels: 是否使用软标签
        """
        self.confidence_threshold = confidence_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.use_soft_labels = use_soft_labels
    
    @abstractmethod
    def generate(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        生成伪标签
        
        Args:
            model: 用于生成伪标签的模型（通常是Teacher）
            inputs: 输入数据
            
        Returns:
            包含伪标签和相关信息的字典
        """
        pass
    
    def filter_by_confidence(
        self,
        predictions: torch.Tensor,
        confidence: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根据置信度过滤预测
        
        Args:
            predictions: 预测结果
            confidence: 置信度
            mask: 可选的额外掩码
            
        Returns:
            过滤后的预测和有效掩码
        """
        valid_mask = confidence >= self.confidence_threshold
        
        if mask is not None:
            valid_mask = valid_mask & mask
        
        return predictions, valid_mask
    
    def filter_by_uncertainty(
        self,
        predictions: torch.Tensor,
        uncertainty: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根据不确定性过滤预测
        
        Args:
            predictions: 预测结果
            uncertainty: 不确定性
            mask: 可选的额外掩码
            
        Returns:
            过滤后的预测和有效掩码
        """
        valid_mask = uncertainty <= self.uncertainty_threshold
        
        if mask is not None:
            valid_mask = valid_mask & mask
        
        return predictions, valid_mask


class SegmentationPseudoLabelGenerator(PseudoLabelGenerator):
    """分割任务的伪标签生成器"""
    
    def __init__(
        self,
        confidence_threshold: float = 0.8,
        uncertainty_threshold: float = 0.3,
        use_soft_labels: bool = False,
        ignore_index: int = -1
    ):
        super().__init__(confidence_threshold, uncertainty_threshold, use_soft_labels)
        self.ignore_index = ignore_index
    
    @torch.no_grad()
    def generate(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        return_uncertainty: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        生成分割伪标签
        
        Args:
            model: Teacher模型
            inputs: 输入图像 [B, C, D, H, W]
            return_uncertainty: 是否返回不确定性
            
        Returns:
            Dict包含:
                - pseudo_labels: 伪标签 [B, D, H, W]
                - confidence: 置信度 [B, D, H, W]
                - uncertainty: 不确定性 [B, D, H, W] (可选)
                - valid_mask: 有效掩码 [B, D, H, W]
        """
        model.eval()
        
        # 获取模型输出
        outputs = model(inputs)
        
        # 处理不同的输出格式
        if isinstance(outputs, dict):
            logits = outputs.get('logits', outputs.get('pred', None))
            uncertainty = outputs.get('uncertainty', None)
        else:
            logits = outputs
            uncertainty = None
        
        # 计算概率和伪标签
        probs = torch.softmax(logits, dim=1)
        confidence, pseudo_labels = probs.max(dim=1)
        
        # 如果模型没有输出不确定性，用1-confidence估计
        if uncertainty is None and return_uncertainty:
            uncertainty = 1 - confidence
        
        # 根据置信度和不确定性过滤
        valid_mask = confidence >= self.confidence_threshold
        if uncertainty is not None:
            valid_mask = valid_mask & (uncertainty <= self.uncertainty_threshold)
        
        # 软标签处理
        if self.use_soft_labels:
            pseudo_labels = probs
        else:
            # 将无效位置设为ignore_index
            pseudo_labels = pseudo_labels.clone()
            pseudo_labels[~valid_mask] = self.ignore_index
        
        result = {
            'pseudo_labels': pseudo_labels,
            'confidence': confidence,
            'valid_mask': valid_mask
        }
        
        if return_uncertainty and uncertainty is not None:
            result['uncertainty'] = uncertainty
        
        return result


class ClassificationPseudoLabelGenerator(PseudoLabelGenerator):
    """分类任务的伪标签生成器"""
    
    def __init__(
        self,
        confidence_threshold: float = 0.95,
        uncertainty_threshold: float = 0.2,
        use_soft_labels: bool = False,
        num_classes: int = 4
    ):
        super().__init__(confidence_threshold, uncertainty_threshold, use_soft_labels)
        self.num_classes = num_classes
    
    @torch.no_grad()
    def generate(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        return_uncertainty: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        生成分类伪标签
        
        Args:
            model: Teacher模型
            inputs: 输入数据
            return_uncertainty: 是否返回不确定性
            
        Returns:
            Dict包含:
                - pseudo_labels: 伪标签 [B]
                - confidence: 置信度 [B]
                - uncertainty: 不确定性 [B] (可选)
                - valid_mask: 有效掩码 [B]
                - soft_labels: 软标签 [B, C] (如果use_soft_labels=True)
        """
        model.eval()
        
        # 获取模型输出
        outputs = model(inputs)
        
        # 处理不同的输出格式
        if isinstance(outputs, dict):
            logits = outputs.get('logits', outputs.get('pred', None))
            uncertainty = outputs.get('uncertainty', None)
        else:
            logits = outputs
            uncertainty = None
        
        # 计算概率和伪标签
        probs = torch.softmax(logits, dim=1)
        confidence, pseudo_labels = probs.max(dim=1)
        
        # 如果模型没有输出不确定性，用熵估计
        if uncertainty is None and return_uncertainty:
            # 使用熵作为不确定性
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            max_entropy = np.log(self.num_classes)
            uncertainty = entropy / max_entropy  # 归一化到[0, 1]
        
        # 根据置信度和不确定性过滤
        valid_mask = confidence >= self.confidence_threshold
        if uncertainty is not None:
            valid_mask = valid_mask & (uncertainty <= self.uncertainty_threshold)
        
        result = {
            'pseudo_labels': pseudo_labels,
            'confidence': confidence,
            'valid_mask': valid_mask
        }
        
        if self.use_soft_labels:
            result['soft_labels'] = probs
        
        if return_uncertainty and uncertainty is not None:
            result['uncertainty'] = uncertainty
        
        return result


class DetectionPseudoLabelGenerator(PseudoLabelGenerator):
    """检测任务的伪标签生成器"""
    
    def __init__(
        self,
        confidence_threshold: float = 0.7,
        uncertainty_threshold: float = 0.3,
        nms_threshold: float = 0.5,
        min_boxes: int = 0,
        max_boxes: int = 100
    ):
        super().__init__(confidence_threshold, uncertainty_threshold)
        self.nms_threshold = nms_threshold
        self.min_boxes = min_boxes
        self.max_boxes = max_boxes
    
    @torch.no_grad()
    def generate(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        **kwargs
    ) -> Dict[str, Any]:
        """
        生成检测伪标签
        
        Args:
            model: Teacher模型
            inputs: 输入图像
            
        Returns:
            Dict包含:
                - pseudo_boxes: 伪边界框列表
                - pseudo_labels: 伪类别标签列表
                - confidence: 置信度列表
                - valid_mask: 有效掩码
        """
        model.eval()
        
        # 获取模型输出
        outputs = model(inputs)
        
        batch_pseudo_boxes = []
        batch_pseudo_labels = []
        batch_confidence = []
        batch_valid = []
        
        for i in range(len(outputs)):
            output = outputs[i]
            
            # 获取boxes, labels, scores
            boxes = output.get('boxes', output.get('pred_boxes', None))
            labels = output.get('labels', output.get('pred_labels', None))
            scores = output.get('scores', output.get('pred_scores', None))
            
            if boxes is None or len(boxes) == 0:
                batch_pseudo_boxes.append(torch.empty(0, 6))
                batch_pseudo_labels.append(torch.empty(0, dtype=torch.long))
                batch_confidence.append(torch.empty(0))
                batch_valid.append(False)
                continue
            
            # 置信度过滤
            valid_mask = scores >= self.confidence_threshold
            
            if valid_mask.sum() == 0:
                batch_pseudo_boxes.append(torch.empty(0, 6))
                batch_pseudo_labels.append(torch.empty(0, dtype=torch.long))
                batch_confidence.append(torch.empty(0))
                batch_valid.append(False)
                continue
            
            # 应用过滤
            filtered_boxes = boxes[valid_mask]
            filtered_labels = labels[valid_mask]
            filtered_scores = scores[valid_mask]
            
            # 限制数量
            if len(filtered_boxes) > self.max_boxes:
                topk_indices = filtered_scores.topk(self.max_boxes).indices
                filtered_boxes = filtered_boxes[topk_indices]
                filtered_labels = filtered_labels[topk_indices]
                filtered_scores = filtered_scores[topk_indices]
            
            batch_pseudo_boxes.append(filtered_boxes)
            batch_pseudo_labels.append(filtered_labels)
            batch_confidence.append(filtered_scores)
            batch_valid.append(len(filtered_boxes) >= self.min_boxes)
        
        return {
            'pseudo_boxes': batch_pseudo_boxes,
            'pseudo_labels': batch_pseudo_labels,
            'confidence': batch_confidence,
            'valid_mask': batch_valid
        }


class ConsistencyLoss(nn.Module):
    """一致性损失"""
    
    def __init__(
        self,
        loss_type: str = "mse",  # mse, kl, ce
        temperature: float = 1.0,
        reduction: str = "mean"
    ):
        """
        Args:
            loss_type: 损失类型
            temperature: 温度参数（用于软化概率分布）
            reduction: 归约方式
        """
        super().__init__()
        self.loss_type = loss_type
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算一致性损失
        
        Args:
            pred: 学生模型预测
            target: 教师模型预测（作为目标）
            mask: 有效位置掩码
            
        Returns:
            一致性损失
        """
        # 温度软化
        if self.temperature != 1.0:
            pred = pred / self.temperature
            target = target / self.temperature
        
        if self.loss_type == "mse":
            loss = F.mse_loss(pred, target, reduction='none')
        elif self.loss_type == "kl":
            pred_log_prob = F.log_softmax(pred, dim=1)
            target_prob = F.softmax(target, dim=1)
            loss = F.kl_div(pred_log_prob, target_prob, reduction='none').sum(dim=1)
        elif self.loss_type == "ce":
            target_prob = F.softmax(target, dim=1)
            loss = F.cross_entropy(pred, target_prob, reduction='none')
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # 应用掩码
        if mask is not None:
            if mask.dim() < loss.dim():
                mask = mask.unsqueeze(1).expand_as(loss)
            loss = loss * mask.float()
            
            if self.reduction == "mean":
                return loss.sum() / (mask.sum() + 1e-8)
            elif self.reduction == "sum":
                return loss.sum()
        else:
            if self.reduction == "mean":
                return loss.mean()
            elif self.reduction == "sum":
                return loss.sum()
        
        return loss


class RampUpScheduler:
    """预热调度器，用于逐渐增加无监督损失权重"""
    
    def __init__(
        self,
        ramp_up_epochs: int = 20,
        ramp_type: str = "sigmoid",  # linear, sigmoid, cosine
        final_weight: float = 1.0
    ):
        """
        Args:
            ramp_up_epochs: 预热轮数
            ramp_type: 预热类型
            final_weight: 最终权重
        """
        self.ramp_up_epochs = ramp_up_epochs
        self.ramp_type = ramp_type
        self.final_weight = final_weight
    
    def get_weight(self, epoch: int) -> float:
        """
        获取当前epoch的权重
        
        Args:
            epoch: 当前轮数
            
        Returns:
            权重值
        """
        if epoch >= self.ramp_up_epochs:
            return self.final_weight
        
        progress = epoch / self.ramp_up_epochs
        
        if self.ramp_type == "linear":
            return self.final_weight * progress
        elif self.ramp_type == "sigmoid":
            # Sigmoid ramp-up
            return self.final_weight * np.exp(-5 * (1 - progress) ** 2)
        elif self.ramp_type == "cosine":
            return self.final_weight * (1 - np.cos(np.pi * progress)) / 2
        else:
            return self.final_weight * progress


class AdaptiveThreshold:
    """自适应阈值调整器"""
    
    def __init__(
        self,
        initial_threshold: float = 0.7,
        target_ratio: float = 0.5,
        min_threshold: float = 0.5,
        max_threshold: float = 0.95,
        adjustment_rate: float = 0.02,
        momentum: float = 0.9
    ):
        """
        Args:
            initial_threshold: 初始阈值
            target_ratio: 目标使用比例
            min_threshold: 最小阈值
            max_threshold: 最大阈值
            adjustment_rate: 调整速率
            momentum: 动量
        """
        self.threshold = initial_threshold
        self.target_ratio = target_ratio
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.adjustment_rate = adjustment_rate
        self.momentum = momentum
        self.ema_ratio = target_ratio
    
    def update(self, current_ratio: float) -> float:
        """
        根据当前使用比例更新阈值
        
        Args:
            current_ratio: 当前伪标签使用比例
            
        Returns:
            更新后的阈值
        """
        # 更新EMA比例
        self.ema_ratio = self.momentum * self.ema_ratio + (1 - self.momentum) * current_ratio
        
        # 根据比例调整阈值
        if self.ema_ratio < self.target_ratio:
            # 使用比例过低，降低阈值
            self.threshold = max(
                self.min_threshold,
                self.threshold - self.adjustment_rate
            )
        elif self.ema_ratio > self.target_ratio:
            # 使用比例过高，提高阈值
            self.threshold = min(
                self.max_threshold,
                self.threshold + self.adjustment_rate
            )
        
        return self.threshold
    
    def get_threshold(self) -> float:
        """获取当前阈值"""
        return self.threshold


class SemiSupervisedDataset(torch.utils.data.Dataset):
    """半监督学习数据集包装器"""
    
    def __init__(
        self,
        labeled_dataset: torch.utils.data.Dataset,
        unlabeled_dataset: torch.utils.data.Dataset,
        weak_transform: Any = None,
        strong_transform: Any = None
    ):
        """
        Args:
            labeled_dataset: 有标签数据集
            unlabeled_dataset: 无标签数据集
            weak_transform: 弱增强
            strong_transform: 强增强
        """
        self.labeled_dataset = labeled_dataset
        self.unlabeled_dataset = unlabeled_dataset
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
    
    def __len__(self):
        return max(len(self.labeled_dataset), len(self.unlabeled_dataset))
    
    def __getitem__(self, idx):
        # 获取有标签数据
        labeled_idx = idx % len(self.labeled_dataset)
        labeled_sample = self.labeled_dataset[labeled_idx]
        
        # 获取无标签数据
        unlabeled_idx = idx % len(self.unlabeled_dataset)
        unlabeled_sample = self.unlabeled_dataset[unlabeled_idx]
        
        # 对无标签数据应用弱增强和强增强
        unlabeled_image = unlabeled_sample['image']
        
        weak_augmented = self.weak_transform(unlabeled_image) if self.weak_transform else unlabeled_image
        strong_augmented = self.strong_transform(unlabeled_image) if self.strong_transform else unlabeled_image
        
        return {
            'labeled': labeled_sample,
            'unlabeled_weak': weak_augmented,
            'unlabeled_strong': strong_augmented,
            'unlabeled_id': unlabeled_idx
        }


class PseudoLabelQualityMonitor:
    """伪标签质量监控器"""
    
    def __init__(self):
        self.history = {
            'confidence_mean': [],
            'confidence_std': [],
            'uncertainty_mean': [],
            'uncertainty_std': [],
            'usage_ratio': [],
            'accuracy': []  # 如果有验证集
        }
    
    def update(
        self,
        confidence: torch.Tensor,
        uncertainty: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        true_labels: Optional[torch.Tensor] = None,
        pseudo_labels: Optional[torch.Tensor] = None
    ):
        """
        更新监控统计
        
        Args:
            confidence: 置信度
            uncertainty: 不确定性
            valid_mask: 有效掩码
            true_labels: 真实标签（可选，用于计算准确率）
            pseudo_labels: 伪标签
        """
        self.history['confidence_mean'].append(confidence.mean().item())
        self.history['confidence_std'].append(confidence.std().item())
        
        if uncertainty is not None:
            self.history['uncertainty_mean'].append(uncertainty.mean().item())
            self.history['uncertainty_std'].append(uncertainty.std().item())
        
        if valid_mask is not None:
            usage_ratio = valid_mask.float().mean().item()
            self.history['usage_ratio'].append(usage_ratio)
        
        if true_labels is not None and pseudo_labels is not None:
            accuracy = (pseudo_labels == true_labels).float().mean().item()
            self.history['accuracy'].append(accuracy)
    
    def get_summary(self) -> Dict[str, float]:
        """获取统计摘要"""
        summary = {}
        for key, values in self.history.items():
            if values:
                summary[f'{key}_latest'] = values[-1]
                summary[f'{key}_mean'] = np.mean(values)
        return summary
    
    def reset(self):
        """重置历史记录"""
        for key in self.history:
            self.history[key] = []
