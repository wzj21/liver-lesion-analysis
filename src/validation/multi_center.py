"""
多中心验证模块
Multi-Center Validation Module

包含:
- 域偏移分析
- 数据协调化
- 多中心性能评估
- 泛化能力测试
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import os
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt


@dataclass
class CenterInfo:
    """中心信息"""
    center_id: str
    center_name: str
    location: str
    scanner_manufacturer: str
    scanner_model: str
    scan_protocol: str
    num_samples: int = 0
    
    # 数据统计
    mean_hu: Optional[float] = None
    std_hu: Optional[float] = None
    spacing: Optional[Tuple[float, float, float]] = None
    
    # 标签分布
    class_distribution: Dict[str, int] = field(default_factory=dict)


class DomainShiftAnalyzer:
    """
    域偏移分析器
    
    分析不同中心数据之间的分布差异
    """
    
    def __init__(self, feature_extractor: nn.Module, device: torch.device = torch.device('cuda')):
        self.feature_extractor = feature_extractor.to(device)
        self.feature_extractor.eval()
        self.device = device
    
    @torch.no_grad()
    def extract_features(self, dataloader, max_samples: int = 1000) -> np.ndarray:
        """提取特征"""
        features_list = []
        count = 0
        
        for batch in dataloader:
            if count >= max_samples:
                break
            
            if isinstance(batch, dict):
                images = batch['image']
            else:
                images = batch[0]
            
            images = images.to(self.device)
            
            outputs = self.feature_extractor(images)
            
            if isinstance(outputs, dict):
                features = outputs.get('global_features', outputs.get('features', None))
                if isinstance(features, list):
                    features = features[-1]
            else:
                features = outputs
            
            # 全局平均池化
            if features.dim() > 2:
                features = features.mean(dim=tuple(range(2, features.dim())))
            
            features_list.append(features.cpu().numpy())
            count += len(images)
        
        return np.concatenate(features_list, axis=0)
    
    def compute_domain_distance(
        self,
        features_source: np.ndarray,
        features_target: np.ndarray,
        method: str = 'mmd'  # mmd, cosine, kl
    ) -> float:
        """
        计算域间距离
        
        Args:
            features_source: 源域特征
            features_target: 目标域特征
            method: 距离度量方法
            
        Returns:
            域间距离
        """
        if method == 'mmd':
            return self._compute_mmd(features_source, features_target)
        elif method == 'cosine':
            return self._compute_cosine_distance(features_source, features_target)
        elif method == 'kl':
            return self._compute_kl_divergence(features_source, features_target)
        else:
            return self._compute_mmd(features_source, features_target)
    
    def _compute_mmd(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        kernel: str = 'rbf',
        gamma: float = None
    ) -> float:
        """计算最大均值差异 (MMD)"""
        if gamma is None:
            gamma = 1.0 / X.shape[1]
        
        def rbf_kernel(X, Y, gamma):
            XX = np.sum(X**2, axis=1, keepdims=True)
            YY = np.sum(Y**2, axis=1, keepdims=True)
            distances = XX + YY.T - 2 * np.dot(X, Y.T)
            return np.exp(-gamma * distances)
        
        K_XX = rbf_kernel(X, X, gamma)
        K_YY = rbf_kernel(Y, Y, gamma)
        K_XY = rbf_kernel(X, Y, gamma)
        
        m = X.shape[0]
        n = Y.shape[0]
        
        mmd = (K_XX.sum() / (m * m) + 
               K_YY.sum() / (n * n) - 
               2 * K_XY.sum() / (m * n))
        
        return float(np.sqrt(max(mmd, 0)))
    
    def _compute_cosine_distance(self, X: np.ndarray, Y: np.ndarray) -> float:
        """计算余弦距离"""
        mean_X = X.mean(axis=0)
        mean_Y = Y.mean(axis=0)
        
        cosine_sim = np.dot(mean_X, mean_Y) / (np.linalg.norm(mean_X) * np.linalg.norm(mean_Y) + 1e-8)
        return float(1 - cosine_sim)
    
    def _compute_kl_divergence(self, X: np.ndarray, Y: np.ndarray) -> float:
        """计算KL散度（基于高斯假设）"""
        mean_X, cov_X = X.mean(axis=0), np.cov(X.T)
        mean_Y, cov_Y = Y.mean(axis=0), np.cov(Y.T)
        
        # 正则化
        cov_X = cov_X + np.eye(cov_X.shape[0]) * 1e-6
        cov_Y = cov_Y + np.eye(cov_Y.shape[0]) * 1e-6
        
        try:
            inv_cov_Y = np.linalg.inv(cov_Y)
            diff_mean = mean_Y - mean_X
            
            kl = 0.5 * (np.trace(inv_cov_Y @ cov_X) + 
                       diff_mean @ inv_cov_Y @ diff_mean -
                       X.shape[1] + 
                       np.log(np.linalg.det(cov_Y) / np.linalg.det(cov_X)))
            
            return float(max(kl, 0))
        except:
            return float('inf')
    
    def visualize_domain_shift(
        self,
        features_dict: Dict[str, np.ndarray],
        output_path: Optional[str] = None,
        method: str = 'tsne'
    ):
        """
        可视化域偏移
        
        Args:
            features_dict: {center_id: features}字典
            output_path: 输出路径
            method: 可视化方法 (tsne, pca)
        """
        # 合并所有特征
        all_features = []
        labels = []
        center_names = []
        
        for center_id, features in features_dict.items():
            all_features.append(features)
            labels.extend([center_id] * len(features))
            center_names.append(center_id)
        
        all_features = np.concatenate(all_features, axis=0)
        
        # 降维
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_features)-1))
        else:
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
        
        embedded = reducer.fit_transform(all_features)
        
        # 可视化
        plt.figure(figsize=(10, 8))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(center_names)))
        
        for i, center_id in enumerate(center_names):
            mask = np.array(labels) == center_id
            plt.scatter(
                embedded[mask, 0], embedded[mask, 1],
                c=[colors[i]], label=center_id, alpha=0.6, s=20
            )
        
        plt.legend()
        plt.title('Domain Shift Visualization (t-SNE)')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            return output_path
        
        return plt.gcf()
    
    def analyze_multi_center(
        self,
        dataloaders: Dict[str, Any],
        center_infos: Dict[str, CenterInfo]
    ) -> Dict[str, Any]:
        """
        多中心分析
        
        Returns:
            分析报告
        """
        # 提取各中心特征
        features_dict = {}
        for center_id, dataloader in dataloaders.items():
            features_dict[center_id] = self.extract_features(dataloader)
        
        # 计算两两域间距离
        domain_distances = {}
        center_ids = list(features_dict.keys())
        
        for i, c1 in enumerate(center_ids):
            for c2 in center_ids[i+1:]:
                dist = self.compute_domain_distance(
                    features_dict[c1], features_dict[c2]
                )
                domain_distances[f"{c1}-{c2}"] = dist
        
        # 生成报告
        report = {
            'num_centers': len(center_ids),
            'center_infos': {k: v.__dict__ if hasattr(v, '__dict__') else v 
                           for k, v in center_infos.items()},
            'domain_distances': domain_distances,
            'max_distance': max(domain_distances.values()) if domain_distances else 0,
            'min_distance': min(domain_distances.values()) if domain_distances else 0,
            'mean_distance': np.mean(list(domain_distances.values())) if domain_distances else 0,
            'recommendation': ''
        }
        
        # 生成建议
        if report['max_distance'] > 0.5:
            report['recommendation'] = "检测到显著的域偏移，建议使用域适应技术或数据协调化"
        elif report['max_distance'] > 0.3:
            report['recommendation'] = "存在中等程度的域偏移，建议进行数据增强或微调"
        else:
            report['recommendation'] = "域偏移较小，模型可能具有良好的泛化性"
        
        return report


class DataHarmonizer:
    """
    数据协调化器
    
    减少多中心数据的批次效应
    """
    
    def __init__(self, method: str = 'histogram_matching'):
        """
        Args:
            method: 协调化方法
                - histogram_matching: 直方图匹配
                - z_score: Z-score标准化
                - combat: ComBat批次校正
        """
        self.method = method
        self.reference_stats = None
    
    def fit(self, reference_data: np.ndarray):
        """
        学习参考分布
        
        Args:
            reference_data: 参考中心的数据
        """
        if self.method == 'histogram_matching':
            # 计算参考直方图
            self.reference_hist, self.reference_bins = np.histogram(
                reference_data.flatten(), bins=256, density=True
            )
            self.reference_cdf = np.cumsum(self.reference_hist)
            self.reference_cdf = self.reference_cdf / self.reference_cdf[-1]
            
        elif self.method == 'z_score':
            self.reference_stats = {
                'mean': reference_data.mean(),
                'std': reference_data.std()
            }
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        转换数据以匹配参考分布
        
        Args:
            data: 待转换的数据
            
        Returns:
            协调化后的数据
        """
        if self.method == 'histogram_matching':
            return self._histogram_matching(data)
        elif self.method == 'z_score':
            return self._z_score_normalize(data)
        else:
            return data
    
    def _histogram_matching(self, data: np.ndarray) -> np.ndarray:
        """直方图匹配"""
        if self.reference_cdf is None:
            return data
        
        # 计算源数据的CDF
        source_hist, source_bins = np.histogram(data.flatten(), bins=256, density=True)
        source_cdf = np.cumsum(source_hist)
        source_cdf = source_cdf / source_cdf[-1]
        
        # 创建映射
        mapping = np.interp(source_cdf, self.reference_cdf, self.reference_bins[:-1])
        
        # 应用映射
        data_flat = data.flatten()
        data_indices = np.searchsorted(source_bins[:-1], data_flat) - 1
        data_indices = np.clip(data_indices, 0, 254)
        
        transformed = mapping[data_indices].reshape(data.shape)
        
        return transformed
    
    def _z_score_normalize(self, data: np.ndarray) -> np.ndarray:
        """Z-score标准化到参考分布"""
        if self.reference_stats is None:
            return data
        
        # 先标准化到标准正态分布
        normalized = (data - data.mean()) / (data.std() + 1e-8)
        
        # 再转换到参考分布
        transformed = normalized * self.reference_stats['std'] + self.reference_stats['mean']
        
        return transformed


class MultiCenterEvaluator:
    """
    多中心评估器
    
    评估模型在多个中心的表现
    """
    
    def __init__(self, model: nn.Module, device: torch.device = torch.device('cuda')):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
    
    @torch.no_grad()
    def evaluate_center(
        self,
        dataloader,
        center_id: str,
        task: str = 'classification'
    ) -> Dict[str, float]:
        """
        评估单个中心
        
        Args:
            dataloader: 数据加载器
            center_id: 中心ID
            task: 任务类型
            
        Returns:
            评估指标
        """
        all_preds = []
        all_labels = []
        all_probs = []
        
        for batch in dataloader:
            if isinstance(batch, dict):
                images = batch['image'].to(self.device)
                labels = batch.get('label', batch.get('mask'))
            else:
                images, labels = batch[0].to(self.device), batch[1]
            
            outputs = self.model(images)
            
            if isinstance(outputs, dict):
                logits = outputs.get('logits', outputs.get('classification', {}).get('logits'))
            else:
                logits = outputs
            
            if task == 'classification':
                probs = torch.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy() if isinstance(labels, torch.Tensor) else labels)
                all_probs.extend(probs.cpu().numpy())
            
            elif task == 'segmentation':
                preds = logits.argmax(dim=1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.numpy() if isinstance(labels, torch.Tensor) else labels)
        
        # 计算指标
        metrics = {'center_id': center_id}
        
        if task == 'classification':
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            all_probs = np.array(all_probs)
            
            metrics['accuracy'] = accuracy_score(all_labels, all_preds)
            
            try:
                if len(np.unique(all_labels)) > 2:
                    metrics['auc'] = roc_auc_score(all_labels, all_probs, multi_class='ovr')
                else:
                    metrics['auc'] = roc_auc_score(all_labels, all_probs[:, 1])
            except:
                metrics['auc'] = 0.0
                
        elif task == 'segmentation':
            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)
            
            # Dice
            intersection = ((all_preds == 1) & (all_labels == 1)).sum()
            union = (all_preds == 1).sum() + (all_labels == 1).sum()
            metrics['dice'] = 2 * intersection / (union + 1e-8)
        
        return metrics
    
    def evaluate_all_centers(
        self,
        dataloaders: Dict[str, Any],
        task: str = 'classification'
    ) -> Dict[str, Any]:
        """
        评估所有中心
        
        Returns:
            综合评估报告
        """
        results = {}
        all_metrics = []
        
        for center_id, dataloader in dataloaders.items():
            metrics = self.evaluate_center(dataloader, center_id, task)
            results[center_id] = metrics
            all_metrics.append(metrics)
        
        # 计算汇总统计
        metric_names = [k for k in all_metrics[0].keys() if k != 'center_id']
        
        summary = {}
        for metric in metric_names:
            values = [m[metric] for m in all_metrics if metric in m]
            summary[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        # 识别表现差异大的中心
        worst_centers = []
        if 'accuracy' in summary:
            threshold = summary['accuracy']['mean'] - 2 * summary['accuracy']['std']
            for center_id, metrics in results.items():
                if metrics.get('accuracy', 1.0) < threshold:
                    worst_centers.append(center_id)
        
        return {
            'per_center_results': results,
            'summary': summary,
            'worst_centers': worst_centers,
            'generalization_score': 1 - (summary.get('accuracy', {}).get('std', 0) / 
                                         (summary.get('accuracy', {}).get('mean', 1) + 1e-8))
        }
    
    def generate_report(self, evaluation_results: Dict[str, Any]) -> str:
        """生成评估报告"""
        lines = [
            "=" * 60,
            "多中心验证评估报告",
            "=" * 60,
            ""
        ]
        
        # 汇总统计
        lines.append("【汇总统计】")
        for metric, stats in evaluation_results['summary'].items():
            lines.append(f"  {metric}:")
            lines.append(f"    均值: {stats['mean']:.4f}")
            lines.append(f"    标准差: {stats['std']:.4f}")
            lines.append(f"    范围: [{stats['min']:.4f}, {stats['max']:.4f}]")
        lines.append("")
        
        # 各中心结果
        lines.append("【各中心结果】")
        for center_id, metrics in evaluation_results['per_center_results'].items():
            metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items() if k != 'center_id'])
            lines.append(f"  {center_id}: {metric_str}")
        lines.append("")
        
        # 泛化能力评分
        lines.append(f"【泛化能力评分】: {evaluation_results['generalization_score']:.4f}")
        
        if evaluation_results['worst_centers']:
            lines.append(f"【需要关注的中心】: {', '.join(evaluation_results['worst_centers'])}")
        
        return "\n".join(lines)
