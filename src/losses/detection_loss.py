"""
检测损失函数
Detection Loss Functions

包含:
- 匈牙利匹配
- 分类损失
- 边界框回归损失
- 掩码损失
- GIoU损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from scipy.optimize import linear_sum_assignment


class HungarianMatcher(nn.Module):
    """
    匈牙利匹配器
    
    将预测与真实目标进行最优匹配
    """
    
    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        cost_mask: float = 1.0
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_mask = cost_mask
        
    @torch.no_grad()
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        执行匹配
        
        Args:
            outputs: 模型输出 {'pred_logits', 'pred_boxes', 'pred_masks'}
            targets: 真实目标列表
            
        Returns:
            匹配索引列表
        """
        B, num_queries = outputs['pred_logits'].shape[:2]
        
        # 展平预测
        out_prob = outputs['pred_logits'].flatten(0, 1).softmax(-1)  # [B*Q, C]
        out_bbox = outputs['pred_boxes'].flatten(0, 1)  # [B*Q, 6]
        
        # 拼接目标
        tgt_ids = torch.cat([t['labels'] for t in targets])
        tgt_bbox = torch.cat([t['boxes'] for t in targets])
        
        # 分类代价
        cost_class = -out_prob[:, tgt_ids]
        
        # L1边界框代价
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        
        # GIoU代价
        cost_giou = -generalized_box_iou_3d(out_bbox, tgt_bbox)
        
        # 总代价矩阵
        C = (self.cost_class * cost_class + 
             self.cost_bbox * cost_bbox + 
             self.cost_giou * cost_giou)
        
        C = C.view(B, num_queries, -1).cpu()
        
        # 对每个batch执行匈牙利匹配
        sizes = [len(t['labels']) for t in targets]
        indices = []
        
        for i, c in enumerate(C.split(sizes, -1)):
            c_i = c[i]
            if c_i.shape[1] > 0:
                row_ind, col_ind = linear_sum_assignment(c_i.numpy())
                indices.append((
                    torch.as_tensor(row_ind, dtype=torch.int64),
                    torch.as_tensor(col_ind, dtype=torch.int64)
                ))
            else:
                indices.append((
                    torch.tensor([], dtype=torch.int64),
                    torch.tensor([], dtype=torch.int64)
                ))
                
        return indices


def generalized_box_iou_3d(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    计算3D广义IoU
    
    Args:
        boxes1: [N, 6] (x, y, z, w, h, d)
        boxes2: [M, 6] (x, y, z, w, h, d)
        
    Returns:
        [N, M] GIoU矩阵
    """
    # 转换为角点格式
    b1 = box_cxcyczwhd_to_xyxyzz(boxes1)  # [N, 6]
    b2 = box_cxcyczwhd_to_xyxyzz(boxes2)  # [M, 6]
    
    # 计算交集
    inter_min = torch.max(b1[:, None, :3], b2[None, :, :3])  # [N, M, 3]
    inter_max = torch.min(b1[:, None, 3:], b2[None, :, 3:])  # [N, M, 3]
    inter_size = (inter_max - inter_min).clamp(min=0)  # [N, M, 3]
    inter_vol = inter_size.prod(dim=-1)  # [N, M]
    
    # 计算并集
    vol1 = (b1[:, 3:] - b1[:, :3]).prod(dim=-1)  # [N]
    vol2 = (b2[:, 3:] - b2[:, :3]).prod(dim=-1)  # [M]
    union = vol1[:, None] + vol2[None, :] - inter_vol  # [N, M]
    
    iou = inter_vol / (union + 1e-6)
    
    # 计算包围盒
    enclose_min = torch.min(b1[:, None, :3], b2[None, :, :3])
    enclose_max = torch.max(b1[:, None, 3:], b2[None, :, 3:])
    enclose_size = (enclose_max - enclose_min).clamp(min=0)
    enclose_vol = enclose_size.prod(dim=-1)
    
    giou = iou - (enclose_vol - union) / (enclose_vol + 1e-6)
    
    return giou


def box_cxcyczwhd_to_xyxyzz(boxes: torch.Tensor) -> torch.Tensor:
    """将中心宽高格式转换为角点格式"""
    cx, cy, cz, w, h, d = boxes.unbind(-1)
    return torch.stack([
        cx - w / 2, cy - h / 2, cz - d / 2,
        cx + w / 2, cy + h / 2, cz + d / 2
    ], dim=-1)


class SetCriterion(nn.Module):
    """
    DETR风格的集合预测损失
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        matcher: Optional[HungarianMatcher] = None,
        weight_dict: Optional[Dict[str, float]] = None,
        eos_coef: float = 0.1
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher or HungarianMatcher()
        self.weight_dict = weight_dict or {
            'loss_ce': 1.0,
            'loss_bbox': 5.0,
            'loss_giou': 2.0,
            'loss_mask': 1.0
        }
        self.eos_coef = eos_coef
        
        # 背景类权重
        empty_weight = torch.ones(num_classes)
        empty_weight[0] = eos_coef
        self.register_buffer('empty_weight', empty_weight)
        
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        计算损失
        """
        # 执行匹配
        indices = self.matcher(outputs, targets)
        
        # 计算各项损失
        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices))
        losses.update(self.loss_boxes(outputs, targets, indices))
        
        if 'pred_masks' in outputs:
            losses.update(self.loss_masks(outputs, targets, indices))
        
        return losses
    
    def loss_labels(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """分类损失"""
        src_logits = outputs['pred_logits']
        
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], 0,
            dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        
        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2), target_classes, self.empty_weight
        )
        
        return {'loss_ce': loss_ce}
    
    def loss_boxes(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """边界框损失"""
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        
        loss_giou = 1 - torch.diag(generalized_box_iou_3d(src_boxes, target_boxes))
        
        return {
            'loss_bbox': loss_bbox.sum() / max(len(target_boxes), 1),
            'loss_giou': loss_giou.sum() / max(len(target_boxes), 1)
        }
    
    def loss_masks(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """掩码损失"""
        idx = self._get_src_permutation_idx(indices)
        src_masks = outputs['pred_masks'][idx]
        
        target_masks = []
        for t, (_, i) in zip(targets, indices):
            if 'masks' in t and len(i) > 0:
                target_masks.append(t['masks'][i])
        
        if len(target_masks) == 0:
            return {'loss_mask': torch.tensor(0.0, device=src_masks.device)}
        
        target_masks = torch.cat(target_masks, dim=0)
        
        # 调整大小
        if src_masks.shape[-3:] != target_masks.shape[-3:]:
            target_masks = F.interpolate(
                target_masks.unsqueeze(1).float(),
                size=src_masks.shape[-3:],
                mode='trilinear',
                align_corners=False
            ).squeeze(1)
        
        # Dice + BCE损失
        src_masks = src_masks.flatten(1)
        target_masks = target_masks.flatten(1)
        
        # BCE损失
        loss_bce = F.binary_cross_entropy_with_logits(src_masks, target_masks)
        
        # Dice损失
        src_masks_sigmoid = src_masks.sigmoid()
        numerator = 2 * (src_masks_sigmoid * target_masks).sum(1)
        denominator = src_masks_sigmoid.sum(1) + target_masks.sum(1)
        loss_dice = 1 - (numerator + 1) / (denominator + 1)
        
        return {'loss_mask': loss_bce + loss_dice.mean()}
    
    def _get_src_permutation_idx(
        self,
        indices: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取源排列索引"""
        batch_idx = torch.cat([
            torch.full_like(src, i) for i, (src, _) in enumerate(indices)
        ])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx


class DetectionLoss(nn.Module):
    """
    简化的检测损失
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        cls_weight: float = 1.0,
        box_weight: float = 5.0,
        mask_weight: float = 1.0
    ):
        super().__init__()
        self.num_classes = num_classes
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.mask_weight = mask_weight
        
        self.criterion = SetCriterion(num_classes)
        
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """计算检测损失"""
        losses = self.criterion(outputs, targets)
        
        # 加权求和
        total_loss = (
            self.cls_weight * losses.get('loss_ce', 0) +
            self.box_weight * losses.get('loss_bbox', 0) +
            self.box_weight * losses.get('loss_giou', 0) +
            self.mask_weight * losses.get('loss_mask', 0)
        )
        
        losses['total_loss'] = total_loss
        
        return losses


if __name__ == "__main__":
    # 测试损失函数
    pred_logits = torch.randn(2, 10, 2)
    pred_boxes = torch.rand(2, 10, 6)
    pred_masks = torch.randn(2, 10, 32, 32, 32)
    
    outputs = {
        'pred_logits': pred_logits,
        'pred_boxes': pred_boxes,
        'pred_masks': pred_masks
    }
    
    targets = [
        {'labels': torch.tensor([1, 1]), 'boxes': torch.rand(2, 6), 'masks': torch.rand(2, 32, 32, 32)},
        {'labels': torch.tensor([1]), 'boxes': torch.rand(1, 6), 'masks': torch.rand(1, 32, 32, 32)}
    ]
    
    criterion = DetectionLoss()
    losses = criterion(outputs, targets)
    
    print("Detection Loss Test:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")
