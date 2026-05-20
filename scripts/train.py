"""
Unified Training Script for Liver Lesion Analysis System
统一训练脚本

Supports training each stage independently or in sequence.

Usage:
    python train.py --stage 1 --config configs/stage1_liver_seg.yaml
    python train.py --stage 2 --config configs/stage2_det_cls_deform_seg.yaml
    python train.py --stage 3 --config configs/stage3_temporal_cls.yaml
    python train.py --stage 4 --config configs/stage4_activity.yaml
"""

import argparse
import os
import sys
import logging
import yaml
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to terminate training when validation metric stops improving."""
    
    def __init__(self, patience: int = 20, mode: str = 'min', min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
            
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
            
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop


class BaseTrainer:
    """Base trainer with common training utilities."""
    
    def __init__(self, config: Dict, device: str = 'cuda'):
        self.config = config
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.scaler = GradScaler() if config.get('mixed_precision', True) else None
        
        # Training state
        self.epoch = 0
        self.best_metric = float('inf') if config.get('monitor_mode', 'min') == 'min' else 0.0
        self.global_step = 0
        
    def build_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Build optimizer from config."""
        opt_cfg = self.config.get('optimizer', {})
        name = opt_cfg.get('name', 'adamw').lower()
        lr = opt_cfg.get('lr', 1e-4)
        weight_decay = opt_cfg.get('weight_decay', 0.05)
        
        # Separate weight decay for different parameter groups
        decay_params = []
        no_decay_params = []
        for name_p, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name_p or 'norm' in name_p or 'bn' in name_p:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
                
        param_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
        
        if name == 'adamw':
            return optim.AdamW(param_groups, lr=lr, betas=opt_cfg.get('betas', (0.9, 0.999)))
        elif name == 'adam':
            return optim.Adam(param_groups, lr=lr)
        elif name == 'sgd':
            return optim.SGD(param_groups, lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {name}")
    
    def build_scheduler(self, optimizer: optim.Optimizer, total_steps: int):
        """Build learning rate scheduler."""
        sched_cfg = self.config.get('scheduler', {})
        name = sched_cfg.get('name', 'cosine')
        warmup_steps = sched_cfg.get('warmup_epochs', 10) * (total_steps // self.config.get('epochs', 100))
        
        if name == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            return CosineAnnealingLR(
                optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=sched_cfg.get('min_lr', 1e-6),
            )
        elif name == 'step':
            return optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        else:
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    def save_checkpoint(self, model, optimizer, scheduler, path, **extra):
        """Save training checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_metric': self.best_metric,
            'global_step': self.global_step,
            **extra,
        }
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, model, optimizer, scheduler, path):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint.get('epoch', 0)
        self.best_metric = checkpoint.get('best_metric', self.best_metric)
        self.global_step = checkpoint.get('global_step', 0)
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        logger.info(f"Loaded checkpoint from {path} (epoch {self.epoch})")


class Stage1Trainer(BaseTrainer):
    """Trainer for Stage1: Liver Segmentation."""
    
    def __init__(self, config, device='cuda'):
        super().__init__(config, device)
        
    def train(self, model, train_loader, val_loader, save_dir):
        """Train Stage1 model."""
        from src.losses import DiceLoss, FocalTverskyLoss, BoundaryLoss
        
        model = model.to(self.device)
        optimizer = self.build_optimizer(model)
        
        total_steps = len(train_loader) * self.config.get('epochs', 100)
        scheduler = self.build_scheduler(optimizer, total_steps)
        
        # Loss functions
        dice_loss = DiceLoss(smooth=1.0, sigmoid=True)
        tversky_loss = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=0.75)
        boundary_loss = BoundaryLoss()
        
        early_stopping = EarlyStopping(
            patience=self.config.get('early_stopping', {}).get('patience', 20),
            mode='max'  # Monitor Dice
        )
        
        epochs = self.config.get('epochs', 100)
        grad_accum = self.config.get('gradient_accumulation', 4)
        grad_clip = self.config.get('gradient_clip', 1.0)
        
        for epoch in range(self.epoch, epochs):
            self.epoch = epoch
            model.train()
            
            epoch_losses = []
            for step, batch in enumerate(train_loader):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                if self.scaler:
                    with autocast():
                        output = model(images)
                        logits = output['logits']
                        
                        loss_dice = dice_loss(logits, masks)
                        loss_tversky = tversky_loss(logits, masks)
                        
                        # Ramp up boundary loss
                        boundary_weight = min(1.0, epoch / 20)
                        loss_boundary = boundary_loss(logits, masks) * boundary_weight
                        
                        # Deep supervision loss
                        deep_loss = 0.0
                        if 'deep_outputs' in output:
                            for i, deep_out in enumerate(output['deep_outputs']):
                                w = 0.5 ** (len(output['deep_outputs']) - i)
                                deep_loss += w * dice_loss(deep_out, masks)
                        
                        loss = loss_dice + 0.5 * loss_tversky + 0.3 * loss_boundary + 0.3 * deep_loss
                        loss = loss / grad_accum
                    
                    self.scaler.scale(loss).backward()
                    
                    if (step + 1) % grad_accum == 0:
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        self.scaler.step(optimizer)
                        self.scaler.update()
                        optimizer.zero_grad()
                        scheduler.step()
                else:
                    output = model(images)
                    logits = output['logits']
                    loss_dice = dice_loss(logits, masks)
                    loss = loss_dice / grad_accum
                    loss.backward()
                    
                    if (step + 1) % grad_accum == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()
                
                epoch_losses.append(loss.item() * grad_accum)
                self.global_step += 1
                
                if step % 50 == 0:
                    logger.info(f"Epoch {epoch} Step {step}/{len(train_loader)} Loss: {loss.item() * grad_accum:.4f}")
            
            # Validation
            val_dice = self._validate_stage1(model, val_loader)
            avg_loss = np.mean(epoch_losses)
            logger.info(f"Epoch {epoch} - Avg Loss: {avg_loss:.4f}, Val Dice: {val_dice:.4f}")
            
            # Save best model
            if val_dice > self.best_metric:
                self.best_metric = val_dice
                self.save_checkpoint(
                    model, optimizer, scheduler,
                    os.path.join(save_dir, 'best_model.pth'),
                    val_dice=val_dice,
                )
            
            # Early stopping
            if early_stopping(val_dice):
                logger.info(f"Early stopping at epoch {epoch}")
                break
                
            # Periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(
                    model, optimizer, scheduler,
                    os.path.join(save_dir, f'checkpoint_epoch{epoch}.pth'),
                )
    
    @torch.no_grad()
    def _validate_stage1(self, model, val_loader):
        """Validate Stage1 model and return mean Dice score."""
        model.eval()
        dice_scores = []
        
        for batch in val_loader:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            output = model(images)
            pred = output['pred'].float()
            
            # Compute Dice per sample
            for b in range(pred.shape[0]):
                intersection = (pred[b] * masks[b]).sum()
                union = pred[b].sum() + masks[b].sum()
                dice = (2.0 * intersection + 1e-5) / (union + 1e-5)
                dice_scores.append(dice.item())
                
        return np.mean(dice_scores)


class Stage3Trainer(BaseTrainer):
    """Trainer for Stage3: Temporal Classification."""
    
    def __init__(self, config, device='cuda'):
        super().__init__(config, device)
        
    def train(self, model, train_loader, val_loader, save_dir):
        """Train Stage3 model."""
        from src.models.stage3_temporal_cls.temporal_classifier import Stage3Loss
        
        model = model.to(self.device)
        optimizer = self.build_optimizer(model)
        
        total_steps = len(train_loader) * self.config.get('epochs', 100)
        scheduler = self.build_scheduler(optimizer, total_steps)
        
        loss_fn = Stage3Loss(
            num_classes=self.config.get('num_classes', 4),
            evidential_weight=1.0,
            slice_aux_weight=0.3,
            consistency_weight=0.2,
        )
        
        early_stopping = EarlyStopping(patience=20, mode='max')
        epochs = self.config.get('epochs', 100)
        
        for epoch in range(self.epoch, epochs):
            self.epoch = epoch
            model.train()
            epoch_losses = []
            
            for step, batch in enumerate(train_loader):
                slices = batch['slices'].to(self.device)
                masks = batch['masks'].to(self.device)
                morphology = batch['morphology'].to(self.device)
                targets = batch['label'].to(self.device)
                
                if self.scaler:
                    with autocast():
                        output = model(slices, masks, morphology)
                        loss, info = loss_fn(output, targets, epoch=epoch)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad()
                else:
                    output = model(slices, masks, morphology)
                    loss, info = loss_fn(output, targets, epoch=epoch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                
                scheduler.step()
                epoch_losses.append(loss.item())
                self.global_step += 1
            
            val_acc, val_loss = self._validate_stage3(model, val_loader, loss_fn, epoch)
            logger.info(f"Epoch {epoch} - Loss: {np.mean(epoch_losses):.4f}, Val Acc: {val_acc:.4f}")
            
            if val_acc > self.best_metric:
                self.best_metric = val_acc
                self.save_checkpoint(model, optimizer, scheduler,
                                     os.path.join(save_dir, 'best_model.pth'), val_acc=val_acc)
            
            if early_stopping(val_acc):
                logger.info(f"Early stopping at epoch {epoch}")
                break
    
    @torch.no_grad()
    def _validate_stage3(self, model, val_loader, loss_fn, epoch):
        model.eval()
        correct, total = 0, 0
        losses = []
        
        for batch in val_loader:
            slices = batch['slices'].to(self.device)
            masks = batch['masks'].to(self.device)
            morphology = batch['morphology'].to(self.device)
            targets = batch['label'].to(self.device)
            
            output = model(slices, masks, morphology)
            loss, _ = loss_fn(output, targets, epoch=epoch)
            losses.append(loss.item())
            
            correct += (output['pred'] == targets).sum().item()
            total += targets.shape[0]
            
        return correct / max(total, 1), np.mean(losses)


def main():
    parser = argparse.ArgumentParser(description='Train Liver Lesion Analysis System')
    parser.add_argument('--stage', type=int, required=True, choices=[1, 2, 3, 4],
                        help='Training stage (1-4)')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Save directory')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    save_dir = os.path.join(args.save_dir, f'stage{args.stage}')
    os.makedirs(save_dir, exist_ok=True)
    
    logger.info(f"Training Stage {args.stage} with config: {args.config}")
    logger.info(f"Device: {args.device}, Save dir: {save_dir}")
    
    if args.stage == 1:
        from src.models.stage1_liver_seg import CascadeLiverSegmentation
        model = CascadeLiverSegmentation(
            coarse_backbone=config.get('coarse_backbone', 'tiny'),
            fine_backbone=config.get('fine_backbone', 'small'),
        )
        trainer = Stage1Trainer(config.get('training', {}), args.device)
        # User needs to create their DataLoader here
        logger.info("Stage1 trainer ready. Provide train_loader and val_loader to trainer.train()")
        
    elif args.stage == 3:
        from src.models.stage3_temporal_cls import TemporalLesionClassifier
        model = TemporalLesionClassifier(
            slice_encoder_variant=config.get('slice_encoder_variant', 'tiny'),
            temporal_hidden_dim=config.get('temporal_hidden_dim', 512),
            num_classes=config.get('num_classes', 4),
        )
        trainer = Stage3Trainer(config.get('training', {}), args.device)
        logger.info("Stage3 trainer ready. Provide train_loader and val_loader to trainer.train()")
    
    else:
        logger.info(f"Stage {args.stage} trainer: use the BaseTrainer class with appropriate model and loss.")


if __name__ == '__main__':
    main()
