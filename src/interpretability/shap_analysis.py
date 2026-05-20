"""
SHAP分析和特征贡献模块
SHAP Analysis and Feature Contribution Module

包含:
- Deep SHAP
- Gradient SHAP
- 特征重要性分析
- 特征贡献可视化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import matplotlib.pyplot as plt


class DeepSHAP:
    """
    Deep SHAP: 基于DeepLIFT的SHAP值计算
    
    计算每个特征对预测的贡献
    """
    
    def __init__(
        self,
        model: nn.Module,
        background_data: torch.Tensor,
        device: torch.device = torch.device('cuda')
    ):
        """
        Args:
            model: 目标模型
            background_data: 背景数据样本
            device: 设备
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        
        # 计算背景数据的预测作为参考
        with torch.no_grad():
            self.background_data = background_data.to(device)
            self.background_outputs = self._get_model_output(self.background_data)
    
    def _get_model_output(self, x: torch.Tensor) -> torch.Tensor:
        """获取模型输出"""
        output = self.model(x)
        if isinstance(output, dict):
            output = output.get('logits', output.get('classification', {}).get('logits'))
        return output
    
    def explain(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        num_samples: int = 100
    ) -> np.ndarray:
        """
        计算SHAP值
        
        Args:
            input_tensor: 输入张量
            target_class: 目标类别
            num_samples: 采样数量
            
        Returns:
            SHAP值数组，形状与输入相同
        """
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True
        
        # 获取预测
        output = self._get_model_output(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        # 计算梯度
        if isinstance(target_class, int):
            score = output[:, target_class].sum()
        else:
            score = output[range(len(output)), target_class].sum()
        
        self.model.zero_grad()
        score.backward()
        
        gradients = input_tensor.grad.detach()
        
        # DeepSHAP: 使用梯度和输入差异计算贡献
        # 简化版本：使用积分梯度的近似
        shap_values = self._compute_shap_values(
            input_tensor.detach(),
            gradients,
            num_samples
        )
        
        return shap_values.cpu().numpy()
    
    def _compute_shap_values(
        self,
        input_tensor: torch.Tensor,
        gradients: torch.Tensor,
        num_samples: int
    ) -> torch.Tensor:
        """
        计算SHAP值
        
        使用积分梯度近似
        """
        # 选择背景样本
        if len(self.background_data) > num_samples:
            indices = torch.randperm(len(self.background_data))[:num_samples]
            background = self.background_data[indices]
        else:
            background = self.background_data
        
        # 计算平均背景
        baseline = background.mean(dim=0, keepdim=True)
        
        # 输入与背景的差异
        diff = input_tensor - baseline
        
        # 积分梯度近似
        shap_values = diff * gradients
        
        return shap_values
    
    def explain_batch(
        self,
        input_batch: torch.Tensor,
        target_classes: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """批量计算SHAP值"""
        shap_values_list = []
        
        for i in range(len(input_batch)):
            target = target_classes[i].item() if target_classes is not None else None
            shap = self.explain(input_batch[i:i+1], target)
            shap_values_list.append(shap)
        
        return np.concatenate(shap_values_list, axis=0)


class GradientSHAP:
    """
    Gradient SHAP: 基于梯度的SHAP值计算
    
    使用随机基线和梯度计算SHAP值
    """
    
    def __init__(
        self,
        model: nn.Module,
        background_data: torch.Tensor,
        device: torch.device = torch.device('cuda'),
        num_samples: int = 50
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.background_data = background_data.to(device)
        self.num_samples = num_samples
    
    def _get_model_output(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        if isinstance(output, dict):
            output = output.get('logits', output.get('classification', {}).get('logits'))
        return output
    
    def explain(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        计算Gradient SHAP值
        """
        input_tensor = input_tensor.to(self.device)
        
        # 采样基线
        indices = torch.randperm(len(self.background_data))[:self.num_samples]
        baselines = self.background_data[indices]
        
        shap_values = torch.zeros_like(input_tensor)
        
        for baseline in baselines:
            baseline = baseline.unsqueeze(0)
            
            # 随机采样插值点
            alpha = torch.rand(1, device=self.device)
            interpolated = baseline + alpha * (input_tensor - baseline)
            interpolated.requires_grad = True
            
            # 计算梯度
            output = self._get_model_output(interpolated)
            
            if target_class is None:
                target_class = output.argmax(dim=1).item()
            
            score = output[:, target_class].sum()
            self.model.zero_grad()
            score.backward()
            
            gradients = interpolated.grad.detach()
            
            # 累加贡献
            shap_values += gradients * (input_tensor - baseline)
        
        # 平均
        shap_values = shap_values / self.num_samples
        
        return shap_values.cpu().numpy()


class FeatureContributionAnalyzer:
    """
    特征贡献分析器
    
    分析各类特征对预测的贡献
    """
    
    def __init__(
        self,
        feature_groups: Dict[str, List[str]] = None
    ):
        """
        Args:
            feature_groups: 特征分组
        """
        self.feature_groups = feature_groups or {
            'morphology': ['volume', 'surface_area', 'sphericity', 'compactness'],
            'intensity': ['mean_hu', 'std_hu', 'min_hu', 'max_hu'],
            'texture': ['contrast', 'homogeneity', 'energy', 'correlation'],
            'boundary': ['boundary_sharpness', 'irregularity', 'roughness']
        }
    
    def analyze(
        self,
        features: Dict[str, float],
        shap_values: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        分析特征贡献
        
        Args:
            features: 特征字典
            shap_values: SHAP值字典
            
        Returns:
            分析结果
        """
        results = {
            'individual_contributions': {},
            'group_contributions': {},
            'top_positive': [],
            'top_negative': []
        }
        
        # 个体贡献
        for name, value in shap_values.items():
            results['individual_contributions'][name] = {
                'feature_value': features.get(name, None),
                'shap_value': value,
                'direction': 'positive' if value > 0 else 'negative'
            }
        
        # 分组贡献
        for group_name, group_features in self.feature_groups.items():
            group_shap = sum(shap_values.get(f, 0) for f in group_features)
            results['group_contributions'][group_name] = group_shap
        
        # 排序找top贡献
        sorted_contributions = sorted(
            shap_values.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        for name, value in sorted_contributions:
            if value > 0:
                results['top_positive'].append((name, value))
            else:
                results['top_negative'].append((name, value))
        
        results['top_positive'] = results['top_positive'][:5]
        results['top_negative'] = results['top_negative'][:5]
        
        return results
    
    def plot_feature_importance(
        self,
        shap_values: Dict[str, float],
        figsize: Tuple[int, int] = (10, 6),
        max_features: int = 15
    ) -> plt.Figure:
        """
        绘制特征重要性图
        """
        # 按绝对值排序
        sorted_items = sorted(
            shap_values.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:max_features]
        
        names = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = ['#ff6b6b' if v > 0 else '#4ecdc4' for v in values]
        
        bars = ax.barh(range(len(names)), values, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel('SHAP Value (Impact on Prediction)')
        ax.set_title('Feature Importance')
        ax.axvline(x=0, color='black', linewidth=0.5)
        
        # 添加图例
        ax.bar([], [], color='#ff6b6b', label='Positive Impact')
        ax.bar([], [], color='#4ecdc4', label='Negative Impact')
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_group_contributions(
        self,
        group_contributions: Dict[str, float],
        figsize: Tuple[int, int] = (8, 8)
    ) -> plt.Figure:
        """
        绘制分组贡献饼图
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        names = list(group_contributions.keys())
        values = [abs(v) for v in group_contributions.values()]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
        
        wedges, texts, autotexts = ax.pie(
            values,
            labels=names,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90
        )
        
        ax.set_title('Feature Group Contributions')
        
        return fig
    
    def plot_waterfall(
        self,
        base_value: float,
        shap_values: Dict[str, float],
        feature_values: Dict[str, float],
        figsize: Tuple[int, int] = (12, 6),
        max_features: int = 10
    ) -> plt.Figure:
        """
        绘制瀑布图
        """
        # 排序
        sorted_items = sorted(
            shap_values.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:max_features]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        cumulative = base_value
        positions = []
        widths = []
        labels = []
        colors = []
        
        positions.append(cumulative)
        widths.append(0)
        labels.append(f'Base Value\n{base_value:.3f}')
        colors.append('gray')
        
        for name, shap_value in sorted_items:
            positions.append(cumulative)
            widths.append(shap_value)
            cumulative += shap_value
            
            feat_val = feature_values.get(name, '')
            labels.append(f'{name}\n={feat_val}')
            colors.append('#ff6b6b' if shap_value > 0 else '#4ecdc4')
        
        positions.append(cumulative)
        widths.append(0)
        labels.append(f'Prediction\n{cumulative:.3f}')
        colors.append('gray')
        
        # 绘制瀑布
        y = range(len(positions))
        
        for i in range(len(positions)):
            left = min(positions[i], positions[i] + widths[i])
            width = abs(widths[i])
            ax.barh(i, width, left=left, color=colors[i], edgecolor='black')
        
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel('Model Output')
        ax.set_title('SHAP Waterfall Plot')
        ax.axvline(x=base_value, color='gray', linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        return fig


class ContrastiveExplanation:
    """
    对比解释
    
    解释"为什么是A而不是B"
    """
    
    def __init__(
        self,
        model: nn.Module,
        class_names: List[str],
        device: torch.device = torch.device('cuda')
    ):
        self.model = model.to(device)
        self.class_names = class_names
        self.device = device
    
    def explain_contrast(
        self,
        input_tensor: torch.Tensor,
        predicted_class: int,
        contrast_class: int
    ) -> Dict[str, Any]:
        """
        生成对比解释
        
        Args:
            input_tensor: 输入
            predicted_class: 预测类别
            contrast_class: 对比类别
            
        Returns:
            解释结果
        """
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True
        
        # 获取两个类别的输出
        output = self.model(input_tensor)
        if isinstance(output, dict):
            logits = output.get('logits', output.get('classification', {}).get('logits'))
        else:
            logits = output
        
        # 计算差异分数
        diff_score = logits[:, predicted_class] - logits[:, contrast_class]
        
        self.model.zero_grad()
        diff_score.sum().backward()
        
        gradients = input_tensor.grad.detach().cpu().numpy()
        
        # 找到关键区别
        positive_contributions = np.maximum(gradients, 0)
        negative_contributions = np.minimum(gradients, 0)
        
        explanation = {
            'predicted_class': self.class_names[predicted_class],
            'contrast_class': self.class_names[contrast_class],
            'score_difference': diff_score.item(),
            'supporting_features': self._find_key_regions(positive_contributions),
            'opposing_features': self._find_key_regions(negative_contributions),
            'gradient_map': gradients
        }
        
        return explanation
    
    def _find_key_regions(
        self,
        gradient_map: np.ndarray,
        threshold: float = 0.1
    ) -> List[Dict]:
        """找到关键区域"""
        abs_gradient = np.abs(gradient_map)
        max_val = abs_gradient.max()
        
        if max_val == 0:
            return []
        
        # 归一化
        normalized = abs_gradient / max_val
        
        # 找到超过阈值的区域
        key_regions = []
        
        if len(normalized.shape) == 5:  # 3D
            # 找到最大位置
            max_indices = np.unravel_index(
                np.argsort(normalized.ravel())[-10:],
                normalized.shape
            )
            
            for i in range(len(max_indices[0])):
                if normalized[max_indices[0][i], max_indices[1][i], 
                             max_indices[2][i], max_indices[3][i], max_indices[4][i]] > threshold:
                    key_regions.append({
                        'location': (max_indices[2][i], max_indices[3][i], max_indices[4][i]),
                        'importance': float(normalized[max_indices[0][i], max_indices[1][i],
                                                       max_indices[2][i], max_indices[3][i], max_indices[4][i]])
                    })
        
        return key_regions
    
    def generate_natural_language_explanation(
        self,
        explanation: Dict[str, Any]
    ) -> str:
        """
        生成自然语言解释
        """
        pred_class = explanation['predicted_class']
        contrast_class = explanation['contrast_class']
        
        text = f"模型预测为「{pred_class}」而非「{contrast_class}」，"
        text += f"预测分数差异为 {explanation['score_difference']:.3f}。\n\n"
        
        if explanation['supporting_features']:
            text += "支持该预测的主要特征区域：\n"
            for i, region in enumerate(explanation['supporting_features'][:3], 1):
                text += f"  {i}. 位置 {region['location']}，重要性 {region['importance']:.3f}\n"
        
        return text
