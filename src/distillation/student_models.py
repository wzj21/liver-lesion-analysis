"""
轻量级学生模型
Lightweight Student Models

为知识蒸馏设计的轻量化模型架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import math


class ConvNeXtFemtoBlock(nn.Module):
    """
    ConvNeXt Femto Block
    
    轻量化的ConvNeXt块
    """
    
    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init: float = 1e-6
    ):
        super().__init__()
        
        # Depthwise convolution
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
        # Pointwise convolutions (减小扩展比例: 4 -> 2)
        self.pwconv1 = nn.Linear(dim, 2 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(2 * dim, dim)
        
        # Layer scale
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim)) if layer_scale_init > 0 else None
        
        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = x
        
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1)  # [B, C, D, H, W] -> [B, D, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        if self.gamma is not None:
            x = self.gamma * x
        
        x = x.permute(0, 4, 1, 2, 3)  # [B, D, H, W, C] -> [B, C, D, H, W]
        
        x = input + self.drop_path(x)
        
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth)"""
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        
        return output


class LightweightConvNeXt3D(nn.Module):
    """
    轻量化3D ConvNeXt骨干网络
    
    参数量约为完整版的40%
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        dims: List[int] = [48, 96, 192, 384],  # 原: [96, 192, 384, 768]
        depths: List[int] = [2, 2, 4, 2],  # 原: [3, 3, 9, 3]
        drop_path_rate: float = 0.1,
        return_features: bool = True
    ):
        """
        Args:
            in_channels: 输入通道数
            dims: 各stage的通道数
            depths: 各stage的块数
            drop_path_rate: Drop path率
            return_features: 是否返回中间特征
        """
        super().__init__()
        
        self.dims = dims
        self.return_features = return_features
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, dims[0], kernel_size=4, stride=4),
            nn.LayerNorm(dims[0], eps=1e-6, elementwise_affine=True)
        )
        # 修复LayerNorm的维度问题
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, dims[0], kernel_size=4, stride=4),
            Permute(),
            nn.LayerNorm(dims[0], eps=1e-6),
            Permute(reverse=True)
        )
        
        # Drop path schedule
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # Stages
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        cur = 0
        for i in range(4):
            # Stage blocks
            stage = nn.Sequential(*[
                ConvNeXtFemtoBlock(dims[i], dp_rates[cur + j])
                for j in range(depths[i])
            ])
            self.stages.append(stage)
            cur += depths[i]
            
            # Downsample (except last stage)
            if i < 3:
                downsample = nn.Sequential(
                    Permute(),
                    nn.LayerNorm(dims[i], eps=1e-6),
                    Permute(reverse=True),
                    nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2)
                )
                self.downsamples.append(downsample)
        
        # Final norm
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        features = []
        
        x = self.stem(x)
        
        for i, stage in enumerate(self.stages):
            x = stage(x)
            
            if self.return_features:
                features.append(x)
            
            if i < len(self.downsamples):
                x = self.downsamples[i](x)
        
        # Global average pooling
        x_pool = x.mean(dim=(2, 3, 4))
        x_pool = self.norm(x_pool)
        
        if self.return_features:
            return {'features': features, 'global_features': x_pool}
        else:
            return {'global_features': x_pool}


class Permute(nn.Module):
    """维度置换层"""
    
    def __init__(self, reverse: bool = False):
        super().__init__()
        self.reverse = reverse
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.reverse:
            return x.permute(0, 4, 1, 2, 3)  # [B, D, H, W, C] -> [B, C, D, H, W]
        else:
            return x.permute(0, 2, 3, 4, 1)  # [B, C, D, H, W] -> [B, D, H, W, C]


class LightweightSegmentationHead(nn.Module):
    """
    轻量化分割头
    """
    
    def __init__(
        self,
        encoder_channels: List[int] = [48, 96, 192, 384],
        decoder_channels: List[int] = [192, 96, 48],
        num_classes: int = 2,
        use_evidential: bool = True
    ):
        super().__init__()
        
        self.use_evidential = use_evidential
        
        # 解码器
        self.decoders = nn.ModuleList()
        
        in_channels = encoder_channels[-1]
        for i, out_channels in enumerate(decoder_channels):
            skip_channels = encoder_channels[-(i + 2)] if i < len(encoder_channels) - 1 else 0
            
            self.decoders.append(
                LightweightDecoderBlock(
                    in_channels + skip_channels,
                    out_channels
                )
            )
            in_channels = out_channels
        
        # 输出头
        if use_evidential:
            self.output_head = nn.Conv3d(decoder_channels[-1], num_classes, 1)
            self.evidence_activation = nn.Softplus()
        else:
            self.output_head = nn.Conv3d(decoder_channels[-1], num_classes, 1)
    
    def forward(
        self,
        features: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        x = features[-1]
        
        for i, decoder in enumerate(self.decoders):
            # 上采样
            x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
            
            # 跳跃连接
            if i < len(features) - 1:
                skip = features[-(i + 2)]
                # 尺寸对齐
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
                x = torch.cat([x, skip], dim=1)
            
            x = decoder(x)
        
        # 输出
        logits = self.output_head(x)
        
        result = {'logits': logits}
        
        if self.use_evidential:
            alpha = self.evidence_activation(logits) + 1
            S = alpha.sum(dim=1, keepdim=True)
            result['alpha'] = alpha
            result['uncertainty'] = alpha.shape[1] / S.squeeze(1)
        
        return result


class LightweightDecoderBlock(nn.Module):
    """轻量化解码器块"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class LightweightDetectionHead(nn.Module):
    """
    轻量化检测头
    
    简化的DINO风格检测器
    """
    
    def __init__(
        self,
        in_channels: int = 384,
        hidden_dim: int = 128,  # 原: 256
        num_queries: int = 50,  # 原: 100
        num_classes: int = 2,
        num_decoder_layers: int = 3,  # 原: 6
        num_heads: int = 4,  # 原: 8
        deformable_points: int = 2  # 原: 4
    ):
        super().__init__()
        
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        
        # 输入投影
        self.input_proj = nn.Conv3d(in_channels, hidden_dim, 1)
        
        # 查询嵌入
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # Transformer解码器（简化版）
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,  # 原: * 4
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # 预测头
        self.class_head = nn.Linear(hidden_dim, num_classes)
        self.box_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6)  # 3D bbox: x, y, z, w, h, d
        )
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = features.shape[0]
        
        # 投影
        x = self.input_proj(features)
        
        # Flatten空间维度
        x = x.flatten(2).permute(0, 2, 1)  # [B, D*H*W, C]
        
        # 查询
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        
        # 解码
        output = self.decoder(queries, x)
        
        # 预测
        cls_logits = self.class_head(output)
        box_pred = self.box_head(output).sigmoid()
        
        return {
            'cls_logits': cls_logits,
            'box_pred': box_pred,
            'query_features': output
        }


class LightweightClassificationHead(nn.Module):
    """
    轻量化分类头
    
    简化的时序Transformer
    """
    
    def __init__(
        self,
        in_channels: int = 384,
        hidden_dim: int = 256,  # 原: 512
        num_classes: int = 4,
        num_slices: int = 8,
        num_transformer_layers: int = 2,  # 原: 4
        num_heads: int = 4,  # 原: 8
        use_evidential: bool = True
    ):
        super().__init__()
        
        self.use_evidential = use_evidential
        
        # 特征投影
        self.proj = nn.Linear(in_channels, hidden_dim)
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, num_slices, hidden_dim))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        # 分类头
        if use_evidential:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, num_classes)
            )
            self.evidence_activation = nn.Softplus()
        else:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, num_classes)
            )
        
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [B, K, C] 切片特征
        """
        B, K, C = features.shape
        
        # 投影
        x = self.proj(features)
        
        # 添加位置编码
        x = x + self.pos_embed[:, :K, :]
        
        # Transformer
        x = self.transformer(x)
        
        # 全局平均池化
        x = x.mean(dim=1)
        
        # 分类
        logits = self.classifier(x)
        
        result = {'logits': logits, 'global_features': x}
        
        if self.use_evidential:
            alpha = self.evidence_activation(logits) + 1
            S = alpha.sum(dim=1, keepdim=True)
            result['alpha'] = alpha
            result['uncertainty'] = logits.shape[1] / S.squeeze(1)
        
        return result


class LightweightLiverLesionModel(nn.Module):
    """
    完整的轻量化肝脏病灶分析模型
    
    参数量: ~25M (原完整版: ~62M)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_seg_classes: int = 2,
        num_det_classes: int = 2,
        num_cls_classes: int = 4,
        use_evidential: bool = True
    ):
        super().__init__()
        
        # 共享骨干网络
        self.backbone = LightweightConvNeXt3D(
            in_channels=in_channels,
            dims=[48, 96, 192, 384],
            depths=[2, 2, 4, 2]
        )
        
        # Stage1: 分割头
        self.seg_head = LightweightSegmentationHead(
            encoder_channels=[48, 96, 192, 384],
            decoder_channels=[192, 96, 48],
            num_classes=num_seg_classes,
            use_evidential=use_evidential
        )
        
        # Stage2: 检测头
        self.det_head = LightweightDetectionHead(
            in_channels=384,
            hidden_dim=128,
            num_queries=50,
            num_classes=num_det_classes
        )
        
        # Stage3: 分类头
        self.cls_head = LightweightClassificationHead(
            in_channels=384,
            hidden_dim=256,
            num_classes=num_cls_classes,
            use_evidential=use_evidential
        )
    
    def forward(
        self,
        x: torch.Tensor,
        task: str = "all"
    ) -> Dict[str, Any]:
        """
        前向传播
        
        Args:
            x: 输入图像 [B, C, D, H, W]
            task: 任务类型 ("seg", "det", "cls", "all")
            
        Returns:
            各任务的输出
        """
        # 骨干网络
        backbone_out = self.backbone(x)
        features = backbone_out['features']
        global_features = backbone_out['global_features']
        
        outputs = {
            'features': features,
            'global_features': global_features
        }
        
        if task in ["seg", "all"]:
            seg_out = self.seg_head(features)
            outputs['segmentation'] = seg_out
        
        if task in ["det", "all"]:
            det_out = self.det_head(features[-1])
            outputs['detection'] = det_out
        
        if task in ["cls", "all"]:
            # 使用全局特征进行分类
            cls_features = global_features.unsqueeze(1)  # [B, 1, C]
            cls_out = self.cls_head(cls_features)
            outputs['classification'] = cls_out
        
        return outputs
    
    def count_parameters(self) -> int:
        """统计参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_student_model(config: Dict) -> LightweightLiverLesionModel:
    """
    根据配置创建学生模型
    
    Args:
        config: 模型配置
        
    Returns:
        学生模型
    """
    model = LightweightLiverLesionModel(
        in_channels=config.get('in_channels', 1),
        num_seg_classes=config.get('num_seg_classes', 2),
        num_det_classes=config.get('num_det_classes', 2),
        num_cls_classes=config.get('num_cls_classes', 4),
        use_evidential=config.get('use_evidential', True)
    )
    
    print(f"Student model created with {model.count_parameters():,} parameters")
    
    return model
