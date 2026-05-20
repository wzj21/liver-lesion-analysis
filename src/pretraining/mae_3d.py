"""
3D Masked AutoEncoder (MAE) for Self-Supervised Pretraining
3D掩码自编码器

Based on "Masked Autoencoders Are Scalable Vision Learners" (He et al., CVPR 2022)
Extended to 3D for volumetric medical image pretraining.

The encoder only processes visible (unmasked) patches, and a lightweight
decoder reconstructs the original volume from the latent representation
plus mask tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PatchEmbed3D(nn.Module):
    """3D Patch Embedding: splits volume into non-overlapping patches."""
    
    def __init__(
        self,
        volume_size: Tuple[int, int, int] = (64, 64, 64),
        patch_size: Tuple[int, int, int] = (8, 8, 8),
        in_channels: int = 1,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.volume_size = volume_size
        self.patch_size = patch_size
        self.grid_size = (
            volume_size[0] // patch_size[0],
            volume_size[1] // patch_size[1],
            volume_size[2] // patch_size[2],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        
        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, D, H, W)
        Returns:
            (B, num_patches, embed_dim)
        """
        x = self.proj(x)                         # (B, E, gD, gH, gW)
        x = x.flatten(2).transpose(1, 2)         # (B, num_patches, E)
        x = self.norm(x)
        return x


class TransformerBlock(nn.Module):
    """Standard Transformer block with pre-norm."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=attn_dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class MaskedAutoEncoder3D(nn.Module):
    """3D Masked AutoEncoder for self-supervised pretraining.
    
    Pipeline:
    1. Patchify 3D volume
    2. Random masking (default 75%)
    3. Encode only visible patches
    4. Append mask tokens + positional embedding
    5. Decode to reconstruct masked patches
    6. Loss = MSE on masked patches only
    """
    
    def __init__(
        self,
        volume_size: Tuple[int, int, int] = (64, 64, 64),
        patch_size: Tuple[int, int, int] = (8, 8, 8),
        in_channels: int = 1,
        # Encoder
        encoder_embed_dim: int = 768,
        encoder_depth: int = 12,
        encoder_num_heads: int = 12,
        # Decoder
        decoder_embed_dim: int = 384,
        decoder_depth: int = 4,
        decoder_num_heads: int = 6,
        # Training
        mask_ratio: float = 0.75,
        norm_pix_loss: bool = True,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.patch_size = patch_size
        self.in_channels = in_channels
        
        # ---------- Encoder ----------
        self.patch_embed = PatchEmbed3D(
            volume_size, patch_size, in_channels, encoder_embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))
        self.encoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, encoder_embed_dim)
        )
        
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(
                encoder_embed_dim, encoder_num_heads, mlp_ratio, dropout,
            )
            for _ in range(encoder_depth)
        ])
        self.encoder_norm = nn.LayerNorm(encoder_embed_dim)
        
        # ---------- Decoder ----------
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim)
        )
        
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(
                decoder_embed_dim, decoder_num_heads, mlp_ratio, dropout,
            )
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        
        # Prediction head: predict patch pixels
        patch_pixels = patch_size[0] * patch_size[1] * patch_size[2] * in_channels
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_pixels)
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.encoder_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
                
    def patchify(self, volume: torch.Tensor) -> torch.Tensor:
        """Convert volume to sequence of flattened patches.
        
        Args:
            volume: (B, C, D, H, W)
        Returns:
            patches: (B, num_patches, patch_pixels)
        """
        B, C, D, H, W = volume.shape
        pD, pH, pW = self.patch_size
        
        x = volume.reshape(B, C, D // pD, pD, H // pH, pH, W // pW, pW)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)          # (B, gD, gH, gW, C, pD, pH, pW)
        x = x.reshape(B, -1, C * pD * pH * pW)            # (B, N, patch_pixels)
        return x
    
    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        """Reconstruct volume from patches.
        
        Args:
            patches: (B, num_patches, patch_pixels)
        Returns:
            volume: (B, C, D, H, W)
        """
        B = patches.shape[0]
        C = self.in_channels
        pD, pH, pW = self.patch_size
        gD, gH, gW = self.patch_embed.grid_size
        
        x = patches.reshape(B, gD, gH, gW, C, pD, pH, pW)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7)    # (B, C, gD, pD, gH, pH, gW, pW)
        x = x.reshape(B, C, gD * pD, gH * pH, gW * pW)
        return x
    
    def random_masking(
        self, x: torch.Tensor, mask_ratio: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Random masking: keep subset of patches, remove rest.
        
        Args:
            x: (B, N, D) patch embeddings
            mask_ratio: fraction of patches to mask
            
        Returns:
            x_masked: (B, N_visible, D)
            mask: (B, N) binary, 1 = masked
            ids_restore: (B, N) indices to restore original order
        """
        B, N, D = x.shape
        num_keep = int(N * (1 - mask_ratio))
        
        # Random permutation per sample
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep first num_keep
        ids_keep = ids_shuffle[:, :num_keep]
        x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
        
        # Binary mask: 0 = keep, 1 = mask
        mask = torch.ones(B, N, device=x.device)
        mask[:, :num_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward_encoder(
        self, x: torch.Tensor, mask_ratio: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode only visible patches.
        
        Args:
            x: (B, C, D, H, W)
            mask_ratio: masking ratio
            
        Returns:
            latent: (B, N_visible + 1, encoder_embed_dim)
            mask: (B, N)
            ids_restore: (B, N)
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, N, E)
        
        # Add positional embedding (skip cls token position)
        x = x + self.encoder_pos_embed[:, 1:, :]
        
        # Masking
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # Prepend CLS token
        cls_token = self.cls_token + self.encoder_pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Transformer blocks
        for blk in self.encoder_blocks:
            x = blk(x)
        x = self.encoder_norm(x)
        
        return x, mask, ids_restore
    
    def forward_decoder(
        self,
        x: torch.Tensor,
        ids_restore: torch.Tensor,
    ) -> torch.Tensor:
        """Decode: fill mask tokens, reconstruct patches.
        
        Args:
            x: (B, N_visible + 1, encoder_embed_dim)
            ids_restore: (B, N)
            
        Returns:
            pred: (B, N, patch_pixels)
        """
        # Project to decoder dimension
        x = self.decoder_embed(x)  # (B, N_visible + 1, decoder_dim)
        
        # Append mask tokens
        N = ids_restore.shape[1]
        num_visible = x.shape[1] - 1  # minus cls token
        
        mask_tokens = self.mask_token.expand(x.shape[0], N - num_visible, -1)
        
        # Remove cls, unshuffle, then add cls back
        x_no_cls = x[:, 1:, :]
        x_full = torch.cat([x_no_cls, mask_tokens], dim=1)
        x_full = torch.gather(
            x_full, 1,
            ids_restore.unsqueeze(-1).expand(-1, -1, x_full.shape[2]),
        )
        x = torch.cat([x[:, :1, :], x_full], dim=1)  # prepend cls
        
        # Add decoder positional embedding
        x = x + self.decoder_pos_embed
        
        # Decoder transformer
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        # Predict patch pixels (skip cls)
        pred = self.decoder_pred(x[:, 1:, :])
        return pred
    
    def forward(
        self,
        volume: torch.Tensor,
        mask_ratio: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            volume: (B, C, D, H, W)
            mask_ratio: masking ratio (default: self.mask_ratio)
            
        Returns:
            loss: reconstruction loss
            pred: predicted patches (B, N, patch_pixels)
            mask: binary mask (B, N)
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
            
        latent, mask, ids_restore = self.forward_encoder(volume, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.compute_loss(volume, pred, mask)
        
        return {
            'loss': loss,
            'pred': pred,
            'mask': mask,
            'latent': latent,
        }
    
    def compute_loss(
        self,
        volume: torch.Tensor,
        pred: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute reconstruction loss on masked patches only.
        
        Args:
            volume: original volume (B, C, D, H, W)
            pred: predicted patches (B, N, patch_pixels)
            mask: binary mask (B, N), 1 = masked
            
        Returns:
            loss: scalar
        """
        target = self.patchify(volume)  # (B, N, patch_pixels)
        
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()
            
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)     # per-patch MSE: (B, N)
        
        # Average only over masked patches
        loss = (loss * mask).sum() / mask.sum()
        return loss
    
    def get_encoder(self) -> nn.Module:
        """Extract pretrained encoder for downstream tasks.
        
        Returns a module containing patch_embed + encoder blocks.
        The CLS token feature or global-averaged patch features can
        be used for downstream classification/segmentation.
        """
        return _PretrainedEncoder(
            self.patch_embed, self.encoder_pos_embed,
            self.cls_token, self.encoder_blocks, self.encoder_norm,
        )


class _PretrainedEncoder(nn.Module):
    """Wrapper to extract encoder from MAE for downstream use."""
    
    def __init__(self, patch_embed, pos_embed, cls_token, blocks, norm):
        super().__init__()
        self.patch_embed = patch_embed
        self.pos_embed = pos_embed
        self.cls_token = cls_token
        self.blocks = blocks
        self.norm = norm
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        cls = (self.cls_token + self.pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x


class MAEPreTrainer:
    """Training helper for MAE pretraining.
    
    Usage:
        mae = MaskedAutoEncoder3D(...)
        trainer = MAEPreTrainer(mae, lr=1.5e-4, weight_decay=0.05)
        
        for epoch in range(epochs):
            for batch in dataloader:
                loss = trainer.train_step(batch)
    """
    
    def __init__(
        self,
        model: MaskedAutoEncoder3D,
        lr: float = 1.5e-4,
        weight_decay: float = 0.05,
        warmup_epochs: int = 40,
        total_epochs: int = 400,
        device: str = 'cuda',
    ):
        self.model = model.to(device)
        self.device = device
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr,
            weight_decay=weight_decay, betas=(0.9, 0.95),
        )
        self.scaler = torch.cuda.amp.GradScaler()
        
    def get_lr(self, epoch: int, base_lr: float = 1.5e-4) -> float:
        """Cosine schedule with linear warmup."""
        if epoch < self.warmup_epochs:
            return base_lr * epoch / max(self.warmup_epochs, 1)
        progress = (epoch - self.warmup_epochs) / max(
            self.total_epochs - self.warmup_epochs, 1
        )
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    
    def set_lr(self, epoch: int):
        lr = self.get_lr(epoch)
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr
    
    def train_step(self, volume: torch.Tensor) -> float:
        """Single training step.
        
        Args:
            volume: (B, C, D, H, W) CT volume batch
            
        Returns:
            loss value (float)
        """
        self.model.train()
        volume = volume.to(self.device)
        
        with torch.cuda.amp.autocast():
            output = self.model(volume)
            loss = output['loss']
            
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()
    
    def train_epoch(self, dataloader, epoch: int) -> float:
        """Train for one epoch."""
        lr = self.set_lr(epoch)
        total_loss = 0.0
        count = 0
        
        for batch in dataloader:
            if isinstance(batch, dict):
                volume = batch['image']
            else:
                volume = batch
            loss = self.train_step(volume)
            total_loss += loss
            count += 1
            
        avg_loss = total_loss / max(count, 1)
        logger.info(f"MAE Epoch {epoch}: loss={avg_loss:.4f}, lr={lr:.6f}")
        return avg_loss
    
    def save_checkpoint(self, path: str, epoch: int):
        """Save checkpoint."""
        torch.save({
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict(),
        }, path)
        
    def load_checkpoint(self, path: str) -> int:
        """Load checkpoint, return epoch."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        if 'scaler' in ckpt:
            self.scaler.load_state_dict(ckpt['scaler'])
        return ckpt.get('epoch', 0)
    
    def extract_encoder_weights(self, save_path: str):
        """Extract and save only encoder weights for downstream tasks."""
        encoder = self.model.get_encoder()
        torch.save({'model': encoder.state_dict()}, save_path)
        logger.info(f"Encoder weights saved to {save_path}")


if __name__ == "__main__":
    # Quick test
    print("Testing MaskedAutoEncoder3D...")
    mae = MaskedAutoEncoder3D(
        volume_size=(64, 64, 64),
        patch_size=(8, 8, 8),
        in_channels=1,
        encoder_embed_dim=384,
        encoder_depth=6,
        encoder_num_heads=6,
        decoder_embed_dim=192,
        decoder_depth=2,
        decoder_num_heads=3,
        mask_ratio=0.75,
    )
    
    x = torch.randn(2, 1, 64, 64, 64)
    out = mae(x)
    
    print(f"  Loss: {out['loss'].item():.4f}")
    print(f"  Pred shape: {out['pred'].shape}")
    print(f"  Mask shape: {out['mask'].shape}")
    print(f"  Mask ratio: {out['mask'].float().mean():.2f}")
    print(f"  Params: {sum(p.numel() for p in mae.parameters()) / 1e6:.2f}M")
    
    # Test encoder extraction
    encoder = mae.get_encoder()
    enc_out = encoder(x)
    print(f"  Encoder output: {enc_out.shape}")
    print("  ✓ MAE test passed!")
