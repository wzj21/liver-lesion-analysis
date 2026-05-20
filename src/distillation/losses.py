"""
知识蒸馏基础模块
Knowledge Distillation Base Module

包含:
- 特征蒸馏损失
- 输出蒸馏损失
- 关系蒸馏损失
- 注意力蒸馏损失
- 不确定性蒸馏损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class FeatureDistillationLoss(nn.Module):
    """
    特征蒸馏损失
    
    对齐Teacher和Student的中间特征表示
    """
    
    def __init__(
        self,
        loss_type: str = "mse",  # mse, cosine, l1
        normalize: bool = True,
        temperature: float = 1.0
    ):
        """
        Args:
            loss_type: 损失类型
            normalize: 是否归一化特征
            temperature: 温度参数
        """
        super().__init__()
        self.loss_type = loss_type
        self.normalize = normalize
        self.temperature = temperature
    
    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算特征蒸馏损失
        
        Args:
            student_features: 学生特征 [B, C, ...]
            teacher_features: 教师特征 [B, C', ...]
            mask: 可选的掩码
            
        Returns:
            损失值
        """
        # 归一化
        if self.normalize:
            student_features = F.normalize(student_features, dim=1)
            teacher_features = F.normalize(teacher_features, dim=1)
        
        # 温度缩放
        if self.temperature != 1.0:
            student_features = student_features / self.temperature
            teacher_features = teacher_features / self.temperature
        
        if self.loss_type == "mse":
            loss = F.mse_loss(student_features, teacher_features, reduction='none')
        elif self.loss_type == "cosine":
            # 余弦相似度损失
            cos_sim = F.cosine_similarity(student_features, teacher_features, dim=1)
            loss = 1 - cos_sim
        elif self.loss_type == "l1":
            loss = F.l1_loss(student_features, teacher_features, reduction='none')
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # 应用掩码
        if mask is not None:
            loss = loss * mask.unsqueeze(1).float()
            return loss.sum() / (mask.sum() + 1e-8)
        
        return loss.mean()


class FeatureProjector(nn.Module):
    """
    特征投影器
    
    将Student特征投影到Teacher特征空间
    """
    
    def __init__(
        self,
        student_channels: int,
        teacher_channels: int,
        use_bn: bool = True,
        use_relu: bool = False
    ):
        """
        Args:
            student_channels: 学生特征通道数
            teacher_channels: 教师特征通道数
            use_bn: 是否使用BatchNorm
            use_relu: 是否使用ReLU
        """
        super().__init__()
        
        layers = [nn.Conv3d(student_channels, teacher_channels, 1, bias=not use_bn)]
        
        if use_bn:
            layers.append(nn.BatchNorm3d(teacher_channels))
        
        if use_relu:
            layers.append(nn.ReLU(inplace=True))
        
        self.projector = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)


class OutputDistillationLoss(nn.Module):
    """
    输出蒸馏损失 (Logit Distillation)
    
    使用软标签进行知识迁移
    """
    
    def __init__(
        self,
        temperature: float = 4.0,
        loss_type: str = "kl",  # kl, mse
        alpha: float = 0.5  # 软标签权重
    ):
        """
        Args:
            temperature: 软化温度
            loss_type: 损失类型
            alpha: 软标签与硬标签的权重
        """
        super().__init__()
        self.temperature = temperature
        self.loss_type = loss_type
        self.alpha = alpha
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算输出蒸馏损失
        
        Args:
            student_logits: 学生输出logits
            teacher_logits: 教师输出logits
            targets: 真实标签（用于硬标签损失）
            mask: 可选掩码
            
        Returns:
            损失值
        """
        # 软标签
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        
        if self.loss_type == "kl":
            # KL散度
            soft_loss = F.kl_div(soft_student, soft_teacher, reduction='none')
            soft_loss = soft_loss.sum(dim=1)  # 对类别维度求和
        elif self.loss_type == "mse":
            soft_loss = F.mse_loss(
                F.softmax(student_logits / self.temperature, dim=1),
                soft_teacher,
                reduction='none'
            )
            soft_loss = soft_loss.mean(dim=1)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # 温度平方缩放
        soft_loss = soft_loss * (self.temperature ** 2)
        
        # 应用掩码
        if mask is not None:
            soft_loss = soft_loss * mask.float()
            soft_loss = soft_loss.sum() / (mask.sum() + 1e-8)
        else:
            soft_loss = soft_loss.mean()
        
        # 结合硬标签损失
        if targets is not None and self.alpha < 1.0:
            if mask is not None:
                hard_loss = F.cross_entropy(student_logits, targets, reduction='none')
                hard_loss = (hard_loss * mask.float()).sum() / (mask.sum() + 1e-8)
            else:
                hard_loss = F.cross_entropy(student_logits, targets)
            
            total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        else:
            total_loss = soft_loss
        
        return total_loss


class SegmentationDistillationLoss(nn.Module):
    """
    分割任务的蒸馏损失
    
    结合像素级软标签和边界感知
    """
    
    def __init__(
        self,
        temperature: float = 4.0,
        boundary_weight: float = 2.0,
        use_dice: bool = True
    ):
        super().__init__()
        self.temperature = temperature
        self.boundary_weight = boundary_weight
        self.use_dice = use_dice
        
        self.kl_loss = OutputDistillationLoss(temperature=temperature, loss_type="kl")
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算分割蒸馏损失
        """
        # 基础KL散度损失
        B, C, D, H, W = student_logits.shape
        
        # Reshape for KL loss
        student_flat = student_logits.permute(0, 2, 3, 4, 1).reshape(-1, C)
        teacher_flat = teacher_logits.permute(0, 2, 3, 4, 1).reshape(-1, C)
        
        kl_loss = self.kl_loss(student_flat, teacher_flat)
        
        # 边界感知加权
        if self.boundary_weight > 1.0:
            with torch.no_grad():
                teacher_pred = teacher_logits.argmax(dim=1)
                boundary_mask = self._compute_boundary(teacher_pred)
            
            # 边界区域额外损失
            boundary_student = student_logits * boundary_mask.unsqueeze(1)
            boundary_teacher = teacher_logits * boundary_mask.unsqueeze(1)
            
            boundary_student_flat = boundary_student.permute(0, 2, 3, 4, 1).reshape(-1, C)
            boundary_teacher_flat = boundary_teacher.permute(0, 2, 3, 4, 1).reshape(-1, C)
            
            boundary_loss = self.kl_loss(boundary_student_flat, boundary_teacher_flat)
            kl_loss = kl_loss + (self.boundary_weight - 1) * boundary_loss
        
        # Dice蒸馏损失
        if self.use_dice:
            student_prob = F.softmax(student_logits / self.temperature, dim=1)
            teacher_prob = F.softmax(teacher_logits / self.temperature, dim=1)
            
            dice_loss = self._dice_loss(student_prob, teacher_prob)
            kl_loss = kl_loss + dice_loss
        
        return kl_loss
    
    def _compute_boundary(self, pred: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
        """计算边界掩码"""
        pred_float = pred.float().unsqueeze(1)
        
        # 膨胀
        kernel = torch.ones(1, 1, kernel_size, kernel_size, kernel_size, device=pred.device)
        dilated = F.conv3d(pred_float, kernel, padding=kernel_size//2)
        dilated = (dilated > 0).float()
        
        # 腐蚀
        eroded = F.conv3d(pred_float, kernel, padding=kernel_size//2)
        eroded = (eroded == kernel.sum()).float()
        
        # 边界 = 膨胀 - 腐蚀
        boundary = (dilated - eroded).squeeze(1)
        
        return boundary
    
    def _dice_loss(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
        """Dice损失"""
        intersection = (pred * target).sum(dim=(2, 3, 4))
        union = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
        dice = (2 * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()


class DetectionDistillationLoss(nn.Module):
    """
    检测任务的蒸馏损失
    
    包含分类和回归蒸馏
    """
    
    def __init__(
        self,
        cls_temperature: float = 2.0,
        box_loss_type: str = "smooth_l1",
        cls_weight: float = 1.0,
        box_weight: float = 1.0
    ):
        super().__init__()
        self.cls_temperature = cls_temperature
        self.box_loss_type = box_loss_type
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        
        self.cls_loss = OutputDistillationLoss(temperature=cls_temperature)
    
    def forward(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        计算检测蒸馏损失
        
        Args:
            student_outputs: 学生检测输出 {'cls_logits', 'box_pred', ...}
            teacher_outputs: 教师检测输出
            
        Returns:
            损失值
        """
        total_loss = 0.0
        
        # 分类蒸馏
        if 'cls_logits' in student_outputs and 'cls_logits' in teacher_outputs:
            cls_loss = self.cls_loss(
                student_outputs['cls_logits'],
                teacher_outputs['cls_logits']
            )
            total_loss += self.cls_weight * cls_loss
        
        # 边界框回归蒸馏
        if 'box_pred' in student_outputs and 'box_pred' in teacher_outputs:
            if self.box_loss_type == "smooth_l1":
                box_loss = F.smooth_l1_loss(
                    student_outputs['box_pred'],
                    teacher_outputs['box_pred']
                )
            elif self.box_loss_type == "l1":
                box_loss = F.l1_loss(
                    student_outputs['box_pred'],
                    teacher_outputs['box_pred']
                )
            else:
                box_loss = F.mse_loss(
                    student_outputs['box_pred'],
                    teacher_outputs['box_pred']
                )
            
            total_loss += self.box_weight * box_loss
        
        return total_loss


class RelationDistillationLoss(nn.Module):
    """
    关系蒸馏损失 (Relational Knowledge Distillation)
    
    保持样本间的关系结构
    """
    
    def __init__(
        self,
        loss_type: str = "rkd",  # rkd, sp
        distance_weight: float = 1.0,
        angle_weight: float = 2.0
    ):
        """
        Args:
            loss_type: 损失类型 (rkd: 距离+角度, sp: 相似度保持)
            distance_weight: 距离损失权重
            angle_weight: 角度损失权重
        """
        super().__init__()
        self.loss_type = loss_type
        self.distance_weight = distance_weight
        self.angle_weight = angle_weight
    
    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor
    ) -> torch.Tensor:
        """
        计算关系蒸馏损失
        
        Args:
            student_features: 学生特征 [B, D]
            teacher_features: 教师特征 [B, D']
            
        Returns:
            损失值
        """
        if self.loss_type == "rkd":
            return self._rkd_loss(student_features, teacher_features)
        elif self.loss_type == "sp":
            return self._sp_loss(student_features, teacher_features)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
    
    def _rkd_loss(
        self,
        student: torch.Tensor,
        teacher: torch.Tensor
    ) -> torch.Tensor:
        """RKD损失 (距离 + 角度)"""
        # 距离损失
        student_dist = self._pdist(student)
        teacher_dist = self._pdist(teacher)
        
        # 归一化
        student_dist = student_dist / (student_dist.mean() + 1e-8)
        teacher_dist = teacher_dist / (teacher_dist.mean() + 1e-8)
        
        distance_loss = F.smooth_l1_loss(student_dist, teacher_dist)
        
        # 角度损失
        student_angle = self._angle(student)
        teacher_angle = self._angle(teacher)
        
        angle_loss = F.smooth_l1_loss(student_angle, teacher_angle)
        
        return self.distance_weight * distance_loss + self.angle_weight * angle_loss
    
    def _sp_loss(
        self,
        student: torch.Tensor,
        teacher: torch.Tensor
    ) -> torch.Tensor:
        """相似度保持损失"""
        # 计算相似度矩阵
        student_sim = self._similarity_matrix(student)
        teacher_sim = self._similarity_matrix(teacher)
        
        return F.mse_loss(student_sim, teacher_sim)
    
    def _pdist(self, features: torch.Tensor) -> torch.Tensor:
        """计算成对距离"""
        # features: [B, D]
        diff = features.unsqueeze(0) - features.unsqueeze(1)  # [B, B, D]
        dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)  # [B, B]
        return dist
    
    def _angle(self, features: torch.Tensor) -> torch.Tensor:
        """计算三元组角度"""
        # features: [B, D]
        B = features.shape[0]
        if B < 3:
            return torch.tensor(0.0, device=features.device)
        
        # 随机采样三元组
        idx = torch.randperm(B)[:min(B, 64)]
        f = features[idx]
        
        # 计算角度
        f_diff_1 = f.unsqueeze(0) - f.unsqueeze(1)  # [N, N, D]
        f_diff_2 = f.unsqueeze(1) - f.unsqueeze(2)  # [N, N, D] (广播不同)
        
        # 简化：只计算相邻三元组的角度
        angles = F.cosine_similarity(
            f_diff_1[:, :-1].reshape(-1, f.shape[-1]),
            f_diff_1[:, 1:].reshape(-1, f.shape[-1]),
            dim=-1
        )
        
        return angles
    
    def _similarity_matrix(self, features: torch.Tensor) -> torch.Tensor:
        """计算相似度矩阵"""
        features = F.normalize(features, dim=-1)
        sim = torch.mm(features, features.t())
        return sim


class AttentionDistillationLoss(nn.Module):
    """
    注意力蒸馏损失 (Attention Transfer)
    
    对齐Teacher和Student的注意力图
    """
    
    def __init__(
        self,
        p: int = 2,  # 范数
        normalize: bool = True
    ):
        """
        Args:
            p: 计算注意力图的范数
            normalize: 是否归一化注意力图
        """
        super().__init__()
        self.p = p
        self.normalize = normalize
    
    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor
    ) -> torch.Tensor:
        """
        计算注意力蒸馏损失
        
        Args:
            student_features: 学生特征 [B, C, D, H, W]
            teacher_features: 教师特征 [B, C', D, H, W]
            
        Returns:
            损失值
        """
        # 计算注意力图
        student_attention = self._attention_map(student_features)
        teacher_attention = self._attention_map(teacher_features)
        
        # 归一化
        if self.normalize:
            student_attention = self._normalize_attention(student_attention)
            teacher_attention = self._normalize_attention(teacher_attention)
        
        # L2损失
        loss = F.mse_loss(student_attention, teacher_attention)
        
        return loss
    
    def _attention_map(self, features: torch.Tensor) -> torch.Tensor:
        """计算注意力图"""
        # 沿通道维度求p范数
        attention = torch.norm(features, p=self.p, dim=1)  # [B, D, H, W]
        return attention
    
    def _normalize_attention(self, attention: torch.Tensor) -> torch.Tensor:
        """归一化注意力图"""
        B = attention.shape[0]
        attention = attention.view(B, -1)
        attention = F.normalize(attention, dim=-1)
        return attention


class UncertaintyDistillationLoss(nn.Module):
    """
    不确定性蒸馏损失
    
    保持Teacher的不确定性估计能力
    """
    
    def __init__(
        self,
        loss_type: str = "mse",
        alpha_weight: float = 1.0,  # Dirichlet参数权重
        uncertainty_weight: float = 1.0  # 不确定性权重
    ):
        super().__init__()
        self.loss_type = loss_type
        self.alpha_weight = alpha_weight
        self.uncertainty_weight = uncertainty_weight
    
    def forward(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        计算不确定性蒸馏损失
        
        Args:
            student_outputs: 学生输出 {'alpha', 'uncertainty', ...}
            teacher_outputs: 教师输出
            
        Returns:
            损失值
        """
        total_loss = 0.0
        
        # Dirichlet参数对齐 (Evidential)
        if 'alpha' in student_outputs and 'alpha' in teacher_outputs:
            if self.loss_type == "mse":
                alpha_loss = F.mse_loss(
                    student_outputs['alpha'],
                    teacher_outputs['alpha']
                )
            else:
                # KL散度 for Dirichlet
                alpha_loss = self._dirichlet_kl(
                    student_outputs['alpha'],
                    teacher_outputs['alpha']
                )
            total_loss += self.alpha_weight * alpha_loss
        
        # 不确定性对齐
        if 'uncertainty' in student_outputs and 'uncertainty' in teacher_outputs:
            uncertainty_loss = F.mse_loss(
                student_outputs['uncertainty'],
                teacher_outputs['uncertainty']
            )
            total_loss += self.uncertainty_weight * uncertainty_loss
        
        return total_loss
    
    def _dirichlet_kl(
        self,
        alpha_p: torch.Tensor,
        alpha_q: torch.Tensor
    ) -> torch.Tensor:
        """Dirichlet分布的KL散度"""
        sum_alpha_p = alpha_p.sum(dim=-1, keepdim=True)
        sum_alpha_q = alpha_q.sum(dim=-1, keepdim=True)
        
        kl = torch.lgamma(sum_alpha_p) - torch.lgamma(sum_alpha_q)
        kl -= (torch.lgamma(alpha_p) - torch.lgamma(alpha_q)).sum(dim=-1, keepdim=True)
        kl += ((alpha_p - alpha_q) * (torch.digamma(alpha_p) - torch.digamma(sum_alpha_p))).sum(dim=-1, keepdim=True)
        
        return kl.mean()


class CombinedDistillationLoss(nn.Module):
    """
    组合蒸馏损失
    
    结合多种蒸馏策略
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: 蒸馏配置字典
        """
        super().__init__()
        self.config = config
        
        # 特征蒸馏
        if config.get('feature_distillation', {}).get('enabled', True):
            self.feature_loss = FeatureDistillationLoss(
                loss_type=config['feature_distillation'].get('loss_type', 'mse')
            )
            self.feature_weight = config['feature_distillation'].get('weight', 0.5)
        else:
            self.feature_loss = None
        
        # 输出蒸馏
        if config.get('output_distillation', {}).get('enabled', True):
            self.output_loss = OutputDistillationLoss(
                temperature=config['output_distillation'].get('temperature', 4.0),
                loss_type=config['output_distillation'].get('loss_type', 'kl')
            )
            self.output_weight = config['output_distillation'].get('weight', 0.3)
        else:
            self.output_loss = None
        
        # 关系蒸馏
        if config.get('relation_distillation', {}).get('enabled', True):
            self.relation_loss = RelationDistillationLoss(
                loss_type=config['relation_distillation'].get('loss_type', 'rkd')
            )
            self.relation_weight = config['relation_distillation'].get('weight', 0.2)
        else:
            self.relation_loss = None
        
        # 注意力蒸馏
        if config.get('attention_distillation', {}).get('enabled', True):
            self.attention_loss = AttentionDistillationLoss()
            self.attention_weight = config['attention_distillation'].get('weight', 0.1)
        else:
            self.attention_loss = None
        
        # 不确定性蒸馏
        if config.get('uncertainty_distillation', {}).get('enabled', True):
            self.uncertainty_loss = UncertaintyDistillationLoss()
            self.uncertainty_weight = config['uncertainty_distillation'].get('weight', 0.1)
        else:
            self.uncertainty_loss = None
    
    def forward(
        self,
        student_outputs: Dict[str, Any],
        teacher_outputs: Dict[str, Any],
        targets: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算组合蒸馏损失
        
        Args:
            student_outputs: 学生输出
            teacher_outputs: 教师输出
            targets: 真实标签
            
        Returns:
            损失字典
        """
        losses = {}
        
        # 特征蒸馏
        if self.feature_loss is not None and 'features' in student_outputs:
            feature_losses = []
            for s_feat, t_feat in zip(
                student_outputs['features'],
                teacher_outputs['features']
            ):
                feature_losses.append(self.feature_loss(s_feat, t_feat))
            losses['feature_loss'] = self.feature_weight * sum(feature_losses) / len(feature_losses)
        
        # 输出蒸馏
        if self.output_loss is not None and 'logits' in student_outputs:
            losses['output_loss'] = self.output_weight * self.output_loss(
                student_outputs['logits'],
                teacher_outputs['logits'],
                targets.get('labels') if targets else None
            )
        
        # 关系蒸馏
        if self.relation_loss is not None and 'global_features' in student_outputs:
            losses['relation_loss'] = self.relation_weight * self.relation_loss(
                student_outputs['global_features'],
                teacher_outputs['global_features']
            )
        
        # 注意力蒸馏
        if self.attention_loss is not None and 'features' in student_outputs:
            attention_losses = []
            for s_feat, t_feat in zip(
                student_outputs['features'],
                teacher_outputs['features']
            ):
                attention_losses.append(self.attention_loss(s_feat, t_feat))
            losses['attention_loss'] = self.attention_weight * sum(attention_losses) / len(attention_losses)
        
        # 不确定性蒸馏
        if self.uncertainty_loss is not None:
            losses['uncertainty_loss'] = self.uncertainty_weight * self.uncertainty_loss(
                student_outputs,
                teacher_outputs
            )
        
        # 总损失
        losses['total_distill_loss'] = sum(losses.values())
        
        return losses
