"""
注意力可视化模块
Attention Visualization Module

包含:
- Transformer自注意力可视化
- Deformable Attention采样点可视化
- 切片注意力权重可视化
- 空间注意力图
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import matplotlib.colors as mcolors


class AttentionExtractor:
    """
    注意力提取器
    
    从模型中提取各种注意力权重
    """
    
    def __init__(self, model: nn.Module, device: torch.device = torch.device('cuda')):
        self.model = model
        self.device = device
        self.attention_maps = {}
        self.hooks = []
    
    def register_attention_hooks(self, layer_names: List[str]):
        """注册注意力钩子"""
        for name in layer_names:
            layer = self._get_layer(name)
            if layer is not None:
                hook = layer.register_forward_hook(self._attention_hook(name))
                self.hooks.append(hook)
    
    def _get_layer(self, layer_name: str) -> Optional[nn.Module]:
        """获取层"""
        parts = layer_name.split('.')
        layer = self.model
        for part in parts:
            if hasattr(layer, part):
                layer = getattr(layer, part)
            elif part.isdigit():
                layer = layer[int(part)]
            else:
                return None
        return layer
    
    def _attention_hook(self, name: str):
        """注意力钩子"""
        def hook(module, input, output):
            # 尝试提取注意力权重
            if hasattr(module, 'attention_weights'):
                self.attention_maps[name] = module.attention_weights.detach()
            elif isinstance(output, tuple) and len(output) > 1:
                # 很多Transformer返回 (output, attention_weights)
                if isinstance(output[1], torch.Tensor):
                    self.attention_maps[name] = output[1].detach()
            # 对于MultiheadAttention
            elif hasattr(module, 'attn_output_weights'):
                self.attention_maps[name] = module.attn_output_weights.detach()
        return hook
    
    def extract(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """提取注意力图"""
        self.attention_maps = {}
        self.model.eval()
        
        with torch.no_grad():
            _ = self.model(input_tensor.to(self.device))
        
        return self.attention_maps
    
    def cleanup(self):
        """清理钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.attention_maps = {}


class TransformerAttentionVisualizer:
    """
    Transformer注意力可视化
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.head_fusion = config.get('head_fusion', 'mean')
    
    def visualize_self_attention(
        self,
        attention_weights: torch.Tensor,
        tokens: Optional[List[str]] = None,
        layer_idx: int = -1,
        head_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        可视化自注意力矩阵
        
        Args:
            attention_weights: 注意力权重 [B, H, N, N] 或 [H, N, N]
            tokens: token标签
            layer_idx: 层索引
            head_idx: 注意力头索引（None则融合所有头）
            
        Returns:
            注意力热图
        """
        if attention_weights.dim() == 4:
            attn = attention_weights[0]  # 取第一个batch
        else:
            attn = attention_weights
        
        # 选择/融合注意力头
        if head_idx is not None:
            attn = attn[head_idx]
        elif self.head_fusion == 'mean':
            attn = attn.mean(dim=0)
        elif self.head_fusion == 'max':
            attn = attn.max(dim=0)[0]
        else:
            attn = attn.mean(dim=0)
        
        attn = attn.cpu().numpy()
        
        return attn
    
    def visualize_attention_flow(
        self,
        attention_weights: List[torch.Tensor],
        method: str = 'rollout'
    ) -> np.ndarray:
        """
        可视化注意力流（跨层）
        
        Args:
            attention_weights: 各层注意力权重列表
            method: 方法 (rollout, flow)
            
        Returns:
            融合的注意力图
        """
        if method == 'rollout':
            return self._attention_rollout(attention_weights)
        elif method == 'flow':
            return self._attention_flow(attention_weights)
        else:
            return self._attention_rollout(attention_weights)
    
    def _attention_rollout(
        self,
        attention_weights: List[torch.Tensor]
    ) -> np.ndarray:
        """
        Attention Rollout: 累积注意力
        """
        result = None
        
        for attn in attention_weights:
            if attn.dim() == 4:
                attn = attn[0]  # 取第一个batch
            
            # 融合多头
            attn = attn.mean(dim=0)
            
            # 添加残差连接
            attn = attn + torch.eye(attn.shape[0], device=attn.device)
            attn = attn / attn.sum(dim=-1, keepdim=True)
            
            if result is None:
                result = attn
            else:
                result = torch.matmul(attn, result)
        
        return result.cpu().numpy()
    
    def _attention_flow(
        self,
        attention_weights: List[torch.Tensor]
    ) -> np.ndarray:
        """
        Attention Flow: 基于图的分析
        """
        # 简化实现：最后一层注意力
        if attention_weights:
            attn = attention_weights[-1]
            if attn.dim() == 4:
                attn = attn[0]
            attn = attn.mean(dim=0)
            return attn.cpu().numpy()
        return None
    
    def plot_attention_matrix(
        self,
        attention: np.ndarray,
        tokens: Optional[List[str]] = None,
        title: str = "Attention Matrix",
        figsize: Tuple[int, int] = (10, 10),
        cmap: str = 'viridis'
    ) -> plt.Figure:
        """绘制注意力矩阵"""
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(attention, cmap=cmap)
        
        if tokens is not None:
            ax.set_xticks(range(len(tokens)))
            ax.set_yticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=45, ha='right')
            ax.set_yticklabels(tokens)
        
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        
        return fig


class DeformableAttentionVisualizer:
    """
    Deformable Attention可视化
    
    可视化可学习的采样点和注意力权重
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.point_size = config.get('point_size', 5)
        self.line_width = config.get('line_width', 1)
    
    def extract_sampling_locations(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        layer_name: str
    ) -> Dict[str, torch.Tensor]:
        """
        提取Deformable Attention的采样位置
        
        Returns:
            包含参考点、采样偏移和注意力权重的字典
        """
        sampling_info = {}
        
        def hook(module, input, output):
            # 假设Deformable Attention模块有这些属性
            if hasattr(module, 'sampling_offsets'):
                sampling_info['offsets'] = module.sampling_offsets.detach()
            if hasattr(module, 'attention_weights'):
                sampling_info['weights'] = module.attention_weights.detach()
            if hasattr(module, 'reference_points'):
                sampling_info['reference'] = module.reference_points.detach()
        
        # 注册钩子
        layer = self._get_layer(model, layer_name)
        if layer is not None:
            handle = layer.register_forward_hook(hook)
            
            model.eval()
            with torch.no_grad():
                _ = model(input_tensor)
            
            handle.remove()
        
        return sampling_info
    
    def _get_layer(self, model: nn.Module, layer_name: str):
        """获取层"""
        parts = layer_name.split('.')
        layer = model
        for part in parts:
            if hasattr(layer, part):
                layer = getattr(layer, part)
            elif part.isdigit():
                layer = layer[int(part)]
            else:
                return None
        return layer
    
    def visualize_sampling_points(
        self,
        image: np.ndarray,
        reference_points: np.ndarray,
        sampling_offsets: np.ndarray,
        attention_weights: Optional[np.ndarray] = None,
        query_idx: int = 0,
        figsize: Tuple[int, int] = (12, 12)
    ) -> plt.Figure:
        """
        可视化采样点
        
        Args:
            image: 原始图像（2D切片）
            reference_points: 参考点 [N_query, 2]
            sampling_offsets: 采样偏移 [N_query, N_heads, N_levels, N_points, 2]
            attention_weights: 注意力权重 [N_query, N_heads, N_levels, N_points]
            query_idx: 要可视化的query索引
            
        Returns:
            matplotlib图
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # 显示图像
        ax.imshow(image, cmap='gray')
        
        H, W = image.shape[:2]
        
        # 参考点
        ref = reference_points[query_idx]
        ref_pixel = ref * np.array([W, H])
        ax.scatter(ref_pixel[0], ref_pixel[1], c='red', s=100, marker='*', 
                   label='Reference Point', zorder=5)
        
        # 采样点
        offsets = sampling_offsets[query_idx]  # [N_heads, N_levels, N_points, 2]
        
        if attention_weights is not None:
            weights = attention_weights[query_idx]  # [N_heads, N_levels, N_points]
        else:
            weights = None
        
        # 颜色映射
        colors = plt.cm.rainbow(np.linspace(0, 1, offsets.shape[0]))
        
        for head_idx in range(offsets.shape[0]):
            for level_idx in range(offsets.shape[1]):
                for point_idx in range(offsets.shape[2]):
                    offset = offsets[head_idx, level_idx, point_idx]
                    sample_point = ref + offset
                    sample_pixel = sample_point * np.array([W, H])
                    
                    # 根据注意力权重调整大小
                    if weights is not None:
                        weight = weights[head_idx, level_idx, point_idx]
                        size = self.point_size * (1 + weight * 10)
                        alpha = 0.3 + 0.7 * weight
                    else:
                        size = self.point_size
                        alpha = 0.7
                    
                    # 绘制采样点
                    ax.scatter(sample_pixel[0], sample_pixel[1], 
                              c=[colors[head_idx]], s=size, alpha=alpha)
                    
                    # 绘制连线
                    ax.plot([ref_pixel[0], sample_pixel[0]], 
                           [ref_pixel[1], sample_pixel[1]],
                           c=colors[head_idx], linewidth=self.line_width, 
                           alpha=alpha*0.5)
        
        ax.set_title(f'Deformable Attention Sampling (Query {query_idx})')
        ax.legend()
        
        return fig


class SliceAttentionVisualizer:
    """
    切片注意力可视化
    
    用于Stage3时序分类的切片重要性可视化
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.top_k = config.get('highlight_top_k', 3)
    
    def extract_slice_attention(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        提取切片级注意力权重
        
        Args:
            model: 分类模型
            input_tensor: 输入切片序列 [B, K, C, H, W]
            
        Returns:
            切片注意力权重 [B, K]
        """
        attention_weights = None
        
        def hook(module, input, output):
            nonlocal attention_weights
            # 假设聚合模块输出注意力权重
            if isinstance(output, tuple) and len(output) > 1:
                attention_weights = output[1].detach()
            elif hasattr(module, 'attention_weights'):
                attention_weights = module.attention_weights.detach()
        
        # 找到聚合层并注册钩子
        for name, module in model.named_modules():
            if 'aggregator' in name.lower() or 'attention_pool' in name.lower():
                handle = module.register_forward_hook(hook)
                
                model.eval()
                with torch.no_grad():
                    _ = model(input_tensor)
                
                handle.remove()
                break
        
        return attention_weights
    
    def visualize_slice_importance(
        self,
        slices: np.ndarray,
        attention_weights: np.ndarray,
        figsize: Tuple[int, int] = (16, 4)
    ) -> plt.Figure:
        """
        可视化切片重要性
        
        Args:
            slices: 切片图像 [K, H, W]
            attention_weights: 注意力权重 [K]
            
        Returns:
            matplotlib图
        """
        K = len(slices)
        
        fig, axes = plt.subplots(2, K, figsize=figsize)
        
        # 归一化权重
        weights_norm = (attention_weights - attention_weights.min()) / \
                       (attention_weights.max() - attention_weights.min() + 1e-8)
        
        # 找到top-k切片
        top_k_indices = np.argsort(attention_weights)[-self.top_k:]
        
        # 上行：显示切片
        for i, (slice_img, weight) in enumerate(zip(slices, weights_norm)):
            ax = axes[0, i]
            ax.imshow(slice_img, cmap='gray')
            
            # 高亮重要切片
            if i in top_k_indices:
                for spine in ax.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(3)
            
            ax.set_title(f'Slice {i}\nWeight: {attention_weights[i]:.3f}')
            ax.axis('off')
        
        # 下行：显示权重条形图
        ax_bar = fig.add_subplot(2, 1, 2)
        colors = ['red' if i in top_k_indices else 'steelblue' for i in range(K)]
        ax_bar.bar(range(K), attention_weights, color=colors)
        ax_bar.set_xlabel('Slice Index')
        ax_bar.set_ylabel('Attention Weight')
        ax_bar.set_title('Slice Attention Weights')
        
        plt.tight_layout()
        return fig
    
    def create_attention_overlay(
        self,
        slices: np.ndarray,
        attention_weights: np.ndarray,
        cmap: str = 'Reds'
    ) -> np.ndarray:
        """
        创建注意力叠加的切片可视化
        
        Args:
            slices: 切片图像 [K, H, W]
            attention_weights: 注意力权重 [K]
            cmap: 颜色映射
            
        Returns:
            叠加后的图像 [K, H, W, 3]
        """
        K, H, W = slices.shape
        
        # 归一化权重
        weights_norm = (attention_weights - attention_weights.min()) / \
                       (attention_weights.max() - attention_weights.min() + 1e-8)
        
        # 创建彩色输出
        output = np.zeros((K, H, W, 3))
        
        for i in range(K):
            # 归一化切片
            slice_norm = (slices[i] - slices[i].min()) / (slices[i].max() - slices[i].min() + 1e-8)
            
            # 创建RGB图像
            gray_rgb = np.stack([slice_norm] * 3, axis=-1)
            
            # 创建注意力颜色叠加
            attention_color = plt.cm.get_cmap(cmap)(weights_norm[i])[:3]
            attention_overlay = np.ones((H, W, 3)) * attention_color
            
            # 混合
            alpha = weights_norm[i] * 0.5
            output[i] = (1 - alpha) * gray_rgb + alpha * attention_overlay
        
        return np.clip(output, 0, 1)
