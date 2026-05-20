"""
半监督学习训练器
Semi-Supervised Learning Trainer

统一的训练接口，支持:
- Stage1: CPS/Mean Teacher分割
- Stage2: Teacher-Student检测分割
- Stage3: FixMatch分类
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import logging
import os
from tqdm import tqdm
import yaml

from .teacher_student import TeacherStudentFramework, MultiTaskTeacherStudent
from .fixmatch import FixMatch, FlexMatch, EvidentialFixMatch
from .cross_pseudo_supervision import CrossPseudoSupervision, MeanTeacher


class SemiSupervisedTrainer:
    """
    半监督学习训练器
    
    支持多种半监督学习方法的统一训练接口
    """
    
    def __init__(
        self,
        framework: nn.Module,
        labeled_dataloader: DataLoader,
        unlabeled_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        config: Dict = None,
        device: torch.device = torch.device('cuda'),
        logger: Optional[logging.Logger] = None
    ):
        """
        Args:
            framework: 半监督学习框架
            labeled_dataloader: 有标签数据加载器
            unlabeled_dataloader: 无标签数据加载器
            val_dataloader: 验证数据加载器
            config: 配置字典
            device: 设备
            logger: 日志记录器
        """
        self.framework = framework
        self.labeled_dataloader = labeled_dataloader
        self.unlabeled_dataloader = unlabeled_dataloader
        self.val_dataloader = val_dataloader
        self.config = config or {}
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        
        # 训练配置
        self.warmup_epochs = config.get('warmup_epochs', 10)
        self.total_epochs = config.get('total_epochs', 200)
        self.current_epoch = 0
        
        # 优化器
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # 混合精度训练
        self.use_amp = config.get('mixed_precision', True)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # 最佳模型追踪
        self.best_metric = 0.0
        self.best_epoch = 0
        
        # 检查点目录
        self.checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 日志
        self.log_frequency = config.get('log_frequency', 100)
        
        # 历史记录
        self.train_history = []
        self.val_history = []
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """创建优化器"""
        opt_config = self.config.get('optimizer', {})
        opt_type = opt_config.get('type', 'AdamW')
        lr = opt_config.get('lr', 1e-4)
        weight_decay = opt_config.get('weight_decay', 0.01)
        
        # 获取模型参数
        if hasattr(self.framework, 'student'):
            params = self.framework.student.parameters()
        elif hasattr(self.framework, 'network_a'):
            # CPS有两个网络
            params = list(self.framework.network_a.parameters()) + \
                     list(self.framework.network_b.parameters())
        elif hasattr(self.framework, 'model'):
            params = self.framework.model.parameters()
        else:
            params = self.framework.parameters()
        
        if opt_type == 'AdamW':
            return torch.optim.AdamW(
                params,
                lr=lr,
                weight_decay=weight_decay,
                betas=opt_config.get('betas', (0.9, 0.999))
            )
        elif opt_type == 'SGD':
            return torch.optim.SGD(
                params,
                lr=lr,
                momentum=opt_config.get('momentum', 0.9),
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {opt_type}")
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """创建学习率调度器"""
        sched_config = self.config.get('scheduler', {})
        sched_type = sched_config.get('type', 'cosine')
        
        if sched_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.total_epochs,
                eta_min=sched_config.get('min_lr', 1e-6)
            )
        elif sched_type == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config.get('step_size', 30),
                gamma=sched_config.get('gamma', 0.1)
            )
        elif sched_type == 'none':
            return None
        else:
            return None
    
    def _get_batch(self, labeled_iter, unlabeled_iter):
        """获取一个批次的数据"""
        try:
            labeled_batch = next(labeled_iter)
        except StopIteration:
            labeled_iter = iter(self.labeled_dataloader)
            labeled_batch = next(labeled_iter)
        
        try:
            unlabeled_batch = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(self.unlabeled_dataloader)
            unlabeled_batch = next(unlabeled_iter)
        
        return labeled_batch, unlabeled_batch, labeled_iter, unlabeled_iter
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        训练一个epoch
        
        Args:
            epoch: 当前epoch
            
        Returns:
            平均损失字典
        """
        self.framework.train()
        self.current_epoch = epoch
        
        if hasattr(self.framework, 'set_epoch'):
            self.framework.set_epoch(epoch)
        
        # 初始化
        labeled_iter = iter(self.labeled_dataloader)
        unlabeled_iter = iter(self.unlabeled_dataloader)
        
        num_batches = max(len(self.labeled_dataloader), len(self.unlabeled_dataloader))
        
        epoch_losses = {}
        
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch}")
        
        for batch_idx in pbar:
            # 获取数据
            labeled_batch, unlabeled_batch, labeled_iter, unlabeled_iter = \
                self._get_batch(labeled_iter, unlabeled_iter)
            
            # 移动到设备
            labeled_batch = self._to_device(labeled_batch)
            unlabeled_batch = self._to_device(unlabeled_batch)
            
            # 训练步骤
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    losses = self._train_step(labeled_batch, unlabeled_batch)
            else:
                losses = self._train_step(labeled_batch, unlabeled_batch)
            
            # 累积损失
            for k, v in losses.items():
                if k not in epoch_losses:
                    epoch_losses[k] = []
                epoch_losses[k].append(v)
            
            # 更新进度条
            if batch_idx % self.log_frequency == 0:
                pbar.set_postfix({k: f"{v:.4f}" for k, v in losses.items() if 'loss' in k.lower()})
        
        # 计算平均损失
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        
        # 更新学习率
        if self.scheduler is not None:
            self.scheduler.step()
        
        return avg_losses
    
    def _train_step(
        self,
        labeled_batch: Dict,
        unlabeled_batch: Dict
    ) -> Dict[str, float]:
        """执行单步训练"""
        # 根据框架类型调用不同的训练方法
        if isinstance(self.framework, (FixMatch, FlexMatch, EvidentialFixMatch)):
            # FixMatch类型
            return self._fixmatch_step(labeled_batch, unlabeled_batch)
        elif isinstance(self.framework, CrossPseudoSupervision):
            # CPS类型
            return self._cps_step(labeled_batch, unlabeled_batch)
        elif isinstance(self.framework, MeanTeacher):
            # Mean Teacher类型
            return self._mean_teacher_step(labeled_batch, unlabeled_batch)
        elif isinstance(self.framework, (TeacherStudentFramework, MultiTaskTeacherStudent)):
            # Teacher-Student类型
            return self._teacher_student_step(labeled_batch, unlabeled_batch)
        else:
            raise ValueError(f"Unknown framework type: {type(self.framework)}")
    
    def _fixmatch_step(
        self,
        labeled_batch: Dict,
        unlabeled_batch: Dict
    ) -> Dict[str, float]:
        """FixMatch训练步骤"""
        labeled_images = labeled_batch['image']
        labeled_targets = labeled_batch['label']
        unlabeled_weak = unlabeled_batch.get('weak', unlabeled_batch['image'])
        unlabeled_strong = unlabeled_batch.get('strong', unlabeled_batch['image'])
        
        return self.framework.train_step(
            labeled_images,
            labeled_targets,
            unlabeled_weak,
            unlabeled_strong,
            self.optimizer
        )
    
    def _cps_step(
        self,
        labeled_batch: Dict,
        unlabeled_batch: Dict
    ) -> Dict[str, float]:
        """CPS训练步骤"""
        labeled_images = labeled_batch['image']
        labeled_targets = labeled_batch['mask']
        unlabeled_images = unlabeled_batch['image']
        
        # CPS需要两个优化器
        if not hasattr(self, 'optimizer_b'):
            self.optimizer_b = torch.optim.AdamW(
                self.framework.network_b.parameters(),
                lr=self.config.get('optimizer', {}).get('lr', 1e-4)
            )
        
        return self.framework.train_step(
            labeled_images,
            labeled_targets,
            unlabeled_images,
            self.optimizer,
            self.optimizer_b
        )
    
    def _mean_teacher_step(
        self,
        labeled_batch: Dict,
        unlabeled_batch: Dict
    ) -> Dict[str, float]:
        """Mean Teacher训练步骤"""
        labeled_images = labeled_batch['image']
        labeled_targets = labeled_batch['mask']
        unlabeled_images = unlabeled_batch['image']
        
        return self.framework.train_step(
            labeled_images,
            labeled_targets,
            unlabeled_images,
            self.optimizer
        )
    
    def _teacher_student_step(
        self,
        labeled_batch: Dict,
        unlabeled_batch: Dict
    ) -> Dict[str, float]:
        """Teacher-Student训练步骤"""
        unlabeled_weak = unlabeled_batch.get('weak', unlabeled_batch['image'])
        unlabeled_strong = unlabeled_batch.get('strong', unlabeled_batch['image'])
        
        return self.framework.train_step(
            labeled_batch,
            unlabeled_weak,
            unlabeled_strong,
            self.optimizer
        )
    
    def _to_device(self, batch: Union[Dict, torch.Tensor]) -> Union[Dict, torch.Tensor]:
        """将数据移动到设备"""
        if isinstance(batch, dict):
            return {k: self._to_device(v) for k, v in batch.items()}
        elif isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        elif isinstance(batch, list):
            return [self._to_device(item) for item in batch]
        else:
            return batch
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """验证模型"""
        if self.val_dataloader is None:
            return {}
        
        self.framework.eval()
        
        val_metrics = {}
        all_preds = []
        all_targets = []
        
        for batch in tqdm(self.val_dataloader, desc="Validation"):
            batch = self._to_device(batch)
            
            # 获取预测
            if hasattr(self.framework, 'get_ensemble_prediction'):
                outputs = self.framework.get_ensemble_prediction(batch['image'])
            elif hasattr(self.framework, 'student'):
                outputs = self.framework.student(batch['image'])
            elif hasattr(self.framework, 'model'):
                outputs = self.framework.model(batch['image'])
            else:
                outputs = self.framework(batch['image'])
            
            # 收集预测和标签
            if isinstance(outputs, dict):
                preds = outputs.get('predictions', outputs.get('pred'))
            else:
                preds = outputs.argmax(dim=1) if outputs.dim() > 1 else outputs
            
            all_preds.append(preds.cpu())
            
            if 'mask' in batch:
                all_targets.append(batch['mask'].cpu())
            elif 'label' in batch:
                all_targets.append(batch['label'].cpu())
        
        # 计算指标
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # 根据任务类型计算不同的指标
        if all_preds.dim() > 1:  # 分割任务
            val_metrics = self._compute_segmentation_metrics(all_preds, all_targets)
        else:  # 分类任务
            val_metrics = self._compute_classification_metrics(all_preds, all_targets)
        
        return val_metrics
    
    def _compute_segmentation_metrics(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """计算分割指标"""
        # 计算Dice
        intersection = ((preds == 1) & (targets == 1)).float().sum()
        union = (preds == 1).float().sum() + (targets == 1).float().sum()
        dice = (2 * intersection / (union + 1e-8)).item()
        
        # 计算IoU
        union_iou = ((preds == 1) | (targets == 1)).float().sum()
        iou = (intersection / (union_iou + 1e-8)).item()
        
        return {
            'val_dice': dice,
            'val_iou': iou
        }
    
    def _compute_classification_metrics(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """计算分类指标"""
        accuracy = (preds == targets).float().mean().item()
        
        return {
            'val_accuracy': accuracy
        }
    
    def train(self) -> Dict[str, List[float]]:
        """
        完整训练流程
        
        Returns:
            训练历史
        """
        self.logger.info(f"Starting training for {self.total_epochs} epochs")
        self.logger.info(f"Warmup epochs: {self.warmup_epochs}")
        
        for epoch in range(self.total_epochs):
            # 训练
            train_losses = self.train_epoch(epoch)
            self.train_history.append(train_losses)
            
            # 验证
            val_metrics = self.validate()
            self.val_history.append(val_metrics)
            
            # 日志
            log_str = f"Epoch {epoch}: "
            log_str += " | ".join([f"{k}: {v:.4f}" for k, v in train_losses.items() if 'loss' in k.lower()])
            if val_metrics:
                log_str += " | " + " | ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
            self.logger.info(log_str)
            
            # 保存最佳模型
            current_metric = val_metrics.get('val_dice', val_metrics.get('val_accuracy', 0))
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.best_epoch = epoch
                self.save_checkpoint(epoch, is_best=True)
            
            # 定期保存检查点
            if (epoch + 1) % self.config.get('save_frequency', 10) == 0:
                self.save_checkpoint(epoch)
        
        self.logger.info(f"Training completed. Best metric: {self.best_metric:.4f} at epoch {self.best_epoch}")
        
        return {
            'train_history': self.train_history,
            'val_history': self.val_history
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'framework_state_dict': self.framework.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # 保存最新检查点
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'latest.pth'))
        
        # 保存最佳检查点
        if is_best:
            torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'best.pth'))
        
        # 保存周期检查点
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, f'epoch_{epoch}.pth'))
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.framework.load_state_dict(checkpoint['framework_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint.get('best_metric', 0.0)
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")


def create_semi_supervised_framework(
    stage: str,
    model: nn.Module,
    config: Dict,
    device: torch.device = torch.device('cuda'),
    model_b: Optional[nn.Module] = None
) -> nn.Module:
    """
    创建半监督学习框架
    
    Args:
        stage: 阶段名称 ('stage1', 'stage2', 'stage3')
        model: 主模型
        config: 配置字典
        device: 设备
        model_b: 第二个模型（用于CPS）
        
    Returns:
        半监督学习框架
    """
    if stage == 'stage1':
        # Stage1: 肝脏分割 - 使用CPS或Mean Teacher
        method = config.get('method', 'cross_pseudo_supervision')
        
        if method == 'cross_pseudo_supervision':
            if model_b is None:
                import copy
                model_b = copy.deepcopy(model)
            return CrossPseudoSupervision(
                model, model_b,
                num_classes=2,
                config=config,
                device=device
            )
        elif method == 'mean_teacher':
            return MeanTeacher(
                model,
                num_classes=2,
                config=config,
                device=device
            )
    
    elif stage == 'stage2':
        # Stage2: 检测分割 - 使用Teacher-Student
        return MultiTaskTeacherStudent(
            model,
            config=config,
            device=device
        )
    
    elif stage == 'stage3':
        # Stage3: 分类 - 使用FixMatch
        method = config.get('method', 'fixmatch')
        
        if method == 'fixmatch':
            return FixMatch(
                model,
                num_classes=4,
                config=config,
                device=device
            )
        elif method == 'flexmatch':
            return FlexMatch(
                model,
                num_classes=4,
                config=config,
                device=device
            )
        elif method == 'evidential_fixmatch':
            return EvidentialFixMatch(
                model,
                num_classes=4,
                config=config,
                device=device
            )
    
    raise ValueError(f"Unknown stage or method: {stage}")
