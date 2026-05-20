"""
Grad-CAM可视化模块
Grad-CAM Visualization Module

包含:
- Grad-CAM
- Grad-CAM++
- Score-CAM
- Eigen-CAM
- 3D医学图像适配
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import cv2


class GradCAM:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping
    
    生成分类决策的空间热图，显示模型关注的区域
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layers: List[str],
        use_cuda: bool = True
    ):
        """
        Args:
            model: 目标模型
            target_layers: 目标层名称列表
            use_cuda: 是否使用GPU
        """
        self.model = model
        self.target_layers = target_layers
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        
        self.gradients = {}
        self.activations = {}
        self.hooks = []
        
        self._register_hooks()
    
    def _register_hooks(self):
        """注册前向和反向钩子"""
        for layer_name in self.target_layers:
            layer = self._get_layer(layer_name)
            if layer is not None:
                # 前向钩子：保存激活
                forward_hook = layer.register_forward_hook(
                    self._get_activation_hook(layer_name)
                )
                # 反向钩子：保存梯度
                backward_hook = layer.register_full_backward_hook(
                    self._get_gradient_hook(layer_name)
                )
                self.hooks.extend([forward_hook, backward_hook])
    
    def _get_layer(self, layer_name: str) -> Optional[nn.Module]:
        """根据名称获取层"""
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
    
    def _get_activation_hook(self, name: str):
        """创建前向钩子"""
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            self.activations[name] = output.detach()
        return hook
    
    def _get_gradient_hook(self, name: str):
        """创建反向钩子"""
        def hook(module, grad_input, grad_output):
            if isinstance(grad_output, tuple):
                grad_output = grad_output[0]
            self.gradients[name] = grad_output.detach()
        return hook
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        task: str = "classification"
    ) -> Dict[str, np.ndarray]:
        """
        生成Grad-CAM热图
        
        Args:
            input_tensor: 输入张量 [B, C, D, H, W] 或 [B, C, H, W]
            target_class: 目标类别（None则使用预测类别）
            task: 任务类型 (classification, segmentation, detection)
            
        Returns:
            各目标层的热图字典
        """
        self.model.eval()
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True
        
        # 前向传播
        output = self.model(input_tensor)
        
        # 获取目标分数
        if task == "classification":
            if isinstance(output, dict):
                logits = output.get('logits', output.get('classification', {}).get('logits'))
            else:
                logits = output
            
            if target_class is None:
                target_class = logits.argmax(dim=1)
            
            # 选择目标类别的分数
            if logits.dim() == 2:
                score = logits[range(len(logits)), target_class].sum()
            else:
                score = logits[:, target_class].sum()
                
        elif task == "segmentation":
            if isinstance(output, dict):
                logits = output.get('logits', output.get('segmentation', {}).get('logits'))
            else:
                logits = output
            
            if target_class is None:
                target_class = 1  # 默认前景类
            
            # 使用目标类别的平均分数
            score = logits[:, target_class].mean()
            
        else:
            raise ValueError(f"Unknown task: {task}")
        
        # 反向传播
        self.model.zero_grad()
        score.backward(retain_graph=True)
        
        # 生成热图
        cams = {}
        for layer_name in self.target_layers:
            if layer_name in self.activations and layer_name in self.gradients:
                cam = self._compute_cam(
                    self.activations[layer_name],
                    self.gradients[layer_name]
                )
                cams[layer_name] = cam
        
        return cams
    
    def _compute_cam(
        self,
        activations: torch.Tensor,
        gradients: torch.Tensor
    ) -> np.ndarray:
        """
        计算CAM
        
        Args:
            activations: 激活值 [B, C, ...]
            gradients: 梯度 [B, C, ...]
            
        Returns:
            CAM热图
        """
        # 全局平均池化梯度得到权重
        if activations.dim() == 5:  # 3D: [B, C, D, H, W]
            weights = gradients.mean(dim=(2, 3, 4), keepdim=True)
        else:  # 2D: [B, C, H, W]
            weights = gradients.mean(dim=(2, 3), keepdim=True)
        
        # 加权求和
        cam = (weights * activations).sum(dim=1)  # [B, ...]
        
        # ReLU
        cam = F.relu(cam)
        
        # 归一化
        cam = cam.cpu().numpy()
        cam = self._normalize(cam)
        
        return cam
    
    def _normalize(self, cam: np.ndarray) -> np.ndarray:
        """归一化到[0, 1]"""
        cam_min = cam.min(axis=tuple(range(1, cam.ndim)), keepdims=True)
        cam_max = cam.max(axis=tuple(range(1, cam.ndim)), keepdims=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam
    
    def visualize(
        self,
        image: np.ndarray,
        cam: np.ndarray,
        colormap: str = 'jet',
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        将CAM叠加到原始图像
        
        Args:
            image: 原始图像
            cam: CAM热图
            colormap: 颜色映射
            alpha: 透明度
            
        Returns:
            叠加后的图像
        """
        # 调整CAM大小以匹配图像
        if cam.shape != image.shape[-len(cam.shape):]:
            if len(cam.shape) == 3:  # 3D
                from scipy.ndimage import zoom
                scale = [s / c for s, c in zip(image.shape[-3:], cam.shape)]
                cam = zoom(cam, scale, order=1)
            else:  # 2D
                cam = cv2.resize(cam, (image.shape[-1], image.shape[-2]))
        
        # 应用颜色映射
        if colormap == 'jet':
            cmap = cv2.COLORMAP_JET
        elif colormap == 'hot':
            cmap = cv2.COLORMAP_HOT
        else:
            cmap = cv2.COLORMAP_JET
        
        if len(cam.shape) == 2:  # 2D
            cam_uint8 = (cam * 255).astype(np.uint8)
            cam_colored = cv2.applyColorMap(cam_uint8, cmap)
            cam_colored = cam_colored.astype(np.float32) / 255
            
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)
            
            overlay = alpha * cam_colored + (1 - alpha) * image
            overlay = np.clip(overlay, 0, 1)
            
        else:  # 3D - 逐切片处理
            overlay = np.zeros((*cam.shape, 3))
            for i in range(cam.shape[0]):
                cam_slice = (cam[i] * 255).astype(np.uint8)
                cam_colored = cv2.applyColorMap(cam_slice, cmap)
                cam_colored = cam_colored.astype(np.float32) / 255
                
                if len(image.shape) == 3:
                    img_slice = image[i]
                    if len(img_slice.shape) == 2:
                        img_slice = np.stack([img_slice] * 3, axis=-1)
                else:
                    img_slice = np.stack([image[i]] * 3, axis=-1)
                
                overlay[i] = alpha * cam_colored + (1 - alpha) * img_slice
            
            overlay = np.clip(overlay, 0, 1)
        
        return overlay
    
    def remove_hooks(self):
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.gradients = {}
        self.activations = {}


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++: 改进版Grad-CAM
    
    对正梯度进行加权，更好地处理多目标情况
    """
    
    def _compute_cam(
        self,
        activations: torch.Tensor,
        gradients: torch.Tensor
    ) -> np.ndarray:
        """计算Grad-CAM++"""
        # 计算alpha
        grad_2 = gradients ** 2
        grad_3 = grad_2 * gradients
        
        if activations.dim() == 5:  # 3D
            sum_activations = activations.sum(dim=(2, 3, 4), keepdim=True)
        else:  # 2D
            sum_activations = activations.sum(dim=(2, 3), keepdim=True)
        
        alpha_num = grad_2
        alpha_denom = 2 * grad_2 + sum_activations * grad_3 + 1e-8
        alpha = alpha_num / alpha_denom
        
        # 计算权重
        weights = (alpha * F.relu(gradients))
        if activations.dim() == 5:
            weights = weights.sum(dim=(2, 3, 4), keepdim=True)
        else:
            weights = weights.sum(dim=(2, 3), keepdim=True)
        
        # 加权求和
        cam = (weights * activations).sum(dim=1)
        cam = F.relu(cam)
        
        # 归一化
        cam = cam.cpu().numpy()
        cam = self._normalize(cam)
        
        return cam


class ScoreCAM(GradCAM):
    """
    Score-CAM: 基于分数的CAM
    
    不使用梯度，而是使用激活图作为掩码评估重要性
    """
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        task: str = "classification"
    ) -> Dict[str, np.ndarray]:
        """生成Score-CAM热图"""
        self.model.eval()
        input_tensor = input_tensor.to(self.device)
        
        # 前向传播获取激活
        with torch.no_grad():
            output = self.model(input_tensor)
        
        cams = {}
        for layer_name in self.target_layers:
            if layer_name in self.activations:
                cam = self._compute_score_cam(
                    input_tensor,
                    self.activations[layer_name],
                    target_class,
                    task
                )
                cams[layer_name] = cam
        
        return cams
    
    def _compute_score_cam(
        self,
        input_tensor: torch.Tensor,
        activations: torch.Tensor,
        target_class: Optional[int],
        task: str
    ) -> np.ndarray:
        """计算Score-CAM"""
        B, C = activations.shape[:2]
        spatial_shape = activations.shape[2:]
        input_shape = input_tensor.shape[2:]
        
        # 上采样激活图到输入大小
        if activations.dim() == 5:  # 3D
            upsampled = F.interpolate(
                activations, size=input_shape, mode='trilinear', align_corners=False
            )
        else:  # 2D
            upsampled = F.interpolate(
                activations, size=input_shape, mode='bilinear', align_corners=False
            )
        
        # 归一化每个通道
        upsampled_min = upsampled.min()
        upsampled_max = upsampled.max()
        upsampled = (upsampled - upsampled_min) / (upsampled_max - upsampled_min + 1e-8)
        
        # 计算每个激活通道的贡献
        scores = []
        with torch.no_grad():
            for c in range(C):
                # 使用激活图作为掩码
                masked_input = input_tensor * upsampled[:, c:c+1]
                
                # 获取预测分数
                output = self.model(masked_input)
                
                if task == "classification":
                    if isinstance(output, dict):
                        logits = output.get('logits', output.get('classification', {}).get('logits'))
                    else:
                        logits = output
                    
                    if target_class is None:
                        target_class = logits.argmax(dim=1)
                    
                    score = F.softmax(logits, dim=1)[:, target_class]
                else:
                    score = output.mean()
                
                scores.append(score)
        
        # 计算加权CAM
        scores = torch.stack(scores, dim=1)  # [B, C]
        scores = scores.unsqueeze(-1).unsqueeze(-1)
        if activations.dim() == 5:
            scores = scores.unsqueeze(-1)
        
        cam = (scores * activations).sum(dim=1)
        cam = F.relu(cam)
        
        # 归一化
        cam = cam.cpu().numpy()
        cam = self._normalize(cam)
        
        return cam


class LayerCAM(GradCAM):
    """
    Layer-CAM: 针对浅层的改进CAM
    
    使用逐像素的梯度加权，适用于浅层特征
    """
    
    def _compute_cam(
        self,
        activations: torch.Tensor,
        gradients: torch.Tensor
    ) -> np.ndarray:
        """计算Layer-CAM"""
        # 逐像素ReLU加权
        spatial_weights = F.relu(gradients * activations)
        
        # 沿通道维度求和
        cam = spatial_weights.sum(dim=1)
        
        # 归一化
        cam = cam.cpu().numpy()
        cam = self._normalize(cam)
        
        return cam


class CAMGenerator:
    """
    CAM生成器 - 统一接口
    
    支持多种CAM方法
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layers: List[str],
        method: str = "grad_cam",
        use_cuda: bool = True
    ):
        """
        Args:
            model: 模型
            target_layers: 目标层
            method: CAM方法 (grad_cam, grad_cam_plus_plus, score_cam, layer_cam)
            use_cuda: 是否使用GPU
        """
        self.method = method
        
        if method == "grad_cam":
            self.cam = GradCAM(model, target_layers, use_cuda)
        elif method == "grad_cam_plus_plus":
            self.cam = GradCAMPlusPlus(model, target_layers, use_cuda)
        elif method == "score_cam":
            self.cam = ScoreCAM(model, target_layers, use_cuda)
        elif method == "layer_cam":
            self.cam = LayerCAM(model, target_layers, use_cuda)
        else:
            raise ValueError(f"Unknown CAM method: {method}")
    
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        task: str = "classification"
    ) -> Dict[str, np.ndarray]:
        """生成CAM"""
        return self.cam(input_tensor, target_class, task)
    
    def visualize(
        self,
        image: np.ndarray,
        cam: np.ndarray,
        colormap: str = 'jet',
        alpha: float = 0.5
    ) -> np.ndarray:
        """可视化CAM"""
        return self.cam.visualize(image, cam, colormap, alpha)
    
    def generate_multi_class(
        self,
        input_tensor: torch.Tensor,
        classes: List[int],
        task: str = "classification"
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """为多个类别生成CAM"""
        results = {}
        for cls in classes:
            results[cls] = self.generate(input_tensor, cls, task)
        return results
    
    def cleanup(self):
        """清理资源"""
        self.cam.remove_hooks()
