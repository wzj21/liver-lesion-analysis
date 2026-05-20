"""
主动学习模块
Active Learning Module

包含:
- 不确定性采样
- 多样性采样
- 查询策略
- 标注优先级排序
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum
import heapq


class QueryStrategy(Enum):
    """查询策略"""
    UNCERTAINTY = "uncertainty"
    ENTROPY = "entropy"
    MARGIN = "margin"
    DIVERSITY = "diversity"
    BADGE = "badge"
    CORESET = "coreset"
    HYBRID = "hybrid"


@dataclass
class SampleInfo:
    """样本信息"""
    sample_id: str
    uncertainty_score: float
    diversity_score: float = 0.0
    combined_score: float = 0.0
    predicted_class: Optional[int] = None
    prediction_confidence: float = 0.0
    features: Optional[np.ndarray] = None


class UncertaintySampler:
    """
    不确定性采样器
    
    基于模型不确定性选择最有价值的样本进行标注
    """
    
    def __init__(
        self,
        model: nn.Module,
        strategy: QueryStrategy = QueryStrategy.UNCERTAINTY,
        device: torch.device = torch.device('cuda')
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.strategy = strategy
    
    @torch.no_grad()
    def compute_uncertainty(
        self,
        dataloader,
        method: str = 'evidential'  # evidential, entropy, mc_dropout
    ) -> List[SampleInfo]:
        """
        计算样本不确定性
        
        Args:
            dataloader: 未标注数据加载器
            method: 不确定性估计方法
            
        Returns:
            样本信息列表（按不确定性排序）
        """
        samples = []
        
        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, dict):
                images = batch['image'].to(self.device)
                sample_ids = batch.get('id', [f"{batch_idx}_{i}" for i in range(len(images))])
            else:
                images = batch[0].to(self.device)
                sample_ids = [f"{batch_idx}_{i}" for i in range(len(images))]
            
            if method == 'evidential':
                uncertainties, predictions, confidences = self._evidential_uncertainty(images)
            elif method == 'entropy':
                uncertainties, predictions, confidences = self._entropy_uncertainty(images)
            elif method == 'mc_dropout':
                uncertainties, predictions, confidences = self._mc_dropout_uncertainty(images)
            else:
                uncertainties, predictions, confidences = self._entropy_uncertainty(images)
            
            for i, (sid, unc, pred, conf) in enumerate(zip(
                sample_ids, uncertainties, predictions, confidences
            )):
                samples.append(SampleInfo(
                    sample_id=str(sid),
                    uncertainty_score=float(unc),
                    predicted_class=int(pred),
                    prediction_confidence=float(conf)
                ))
        
        # 按不确定性排序
        samples.sort(key=lambda x: x.uncertainty_score, reverse=True)
        
        return samples
    
    def _evidential_uncertainty(
        self,
        images: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evidential不确定性"""
        outputs = self.model(images)
        
        if isinstance(outputs, dict):
            if 'alpha' in outputs:
                alpha = outputs['alpha']
                S = alpha.sum(dim=1)
                uncertainty = alpha.shape[1] / S
                
                probs = alpha / S.unsqueeze(1)
                confidence, predictions = probs.max(dim=1)
                
                return (
                    uncertainty.cpu().numpy(),
                    predictions.cpu().numpy(),
                    confidence.cpu().numpy()
                )
            elif 'uncertainty' in outputs:
                uncertainty = outputs['uncertainty']
                logits = outputs.get('logits')
                
                probs = torch.softmax(logits, dim=1)
                confidence, predictions = probs.max(dim=1)
                
                return (
                    uncertainty.cpu().numpy(),
                    predictions.cpu().numpy(),
                    confidence.cpu().numpy()
                )
        
        # 回退到熵
        return self._entropy_uncertainty(images)
    
    def _entropy_uncertainty(
        self,
        images: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """熵不确定性"""
        outputs = self.model(images)
        
        if isinstance(outputs, dict):
            logits = outputs.get('logits', outputs.get('classification', {}).get('logits'))
        else:
            logits = outputs
        
        probs = torch.softmax(logits, dim=1)
        
        # 熵
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        max_entropy = np.log(probs.shape[1])
        uncertainty = entropy / max_entropy
        
        confidence, predictions = probs.max(dim=1)
        
        return (
            uncertainty.cpu().numpy(),
            predictions.cpu().numpy(),
            confidence.cpu().numpy()
        )
    
    def _mc_dropout_uncertainty(
        self,
        images: torch.Tensor,
        num_samples: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MC Dropout不确定性"""
        # 启用dropout
        self.model.train()
        
        all_probs = []
        
        for _ in range(num_samples):
            outputs = self.model(images)
            
            if isinstance(outputs, dict):
                logits = outputs.get('logits')
            else:
                logits = outputs
            
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.unsqueeze(0))
        
        self.model.eval()
        
        # 计算预测方差
        all_probs = torch.cat(all_probs, dim=0)  # [T, B, C]
        mean_probs = all_probs.mean(dim=0)  # [B, C]
        
        # 预测熵
        predictive_entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=1)
        
        # 期望熵
        expected_entropy = -torch.sum(
            all_probs * torch.log(all_probs + 1e-8), dim=2
        ).mean(dim=0)
        
        # 互信息作为认知不确定性
        uncertainty = predictive_entropy - expected_entropy
        
        # 归一化
        max_entropy = np.log(mean_probs.shape[1])
        uncertainty = uncertainty / max_entropy
        
        confidence, predictions = mean_probs.max(dim=1)
        
        return (
            uncertainty.cpu().numpy(),
            predictions.cpu().numpy(),
            confidence.cpu().numpy()
        )


class DiversitySampler:
    """
    多样性采样器
    
    选择具有代表性的多样化样本
    """
    
    def __init__(
        self,
        feature_extractor: nn.Module,
        device: torch.device = torch.device('cuda')
    ):
        self.feature_extractor = feature_extractor.to(device)
        self.feature_extractor.eval()
        self.device = device
    
    @torch.no_grad()
    def extract_features(self, dataloader) -> Tuple[np.ndarray, List[str]]:
        """提取特征"""
        features_list = []
        sample_ids = []
        
        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, dict):
                images = batch['image'].to(self.device)
                ids = batch.get('id', [f"{batch_idx}_{i}" for i in range(len(images))])
            else:
                images = batch[0].to(self.device)
                ids = [f"{batch_idx}_{i}" for i in range(len(images))]
            
            outputs = self.feature_extractor(images)
            
            if isinstance(outputs, dict):
                features = outputs.get('global_features', outputs.get('features'))
                if isinstance(features, list):
                    features = features[-1]
            else:
                features = outputs
            
            if features.dim() > 2:
                features = features.mean(dim=tuple(range(2, features.dim())))
            
            features_list.append(features.cpu().numpy())
            sample_ids.extend([str(i) for i in ids])
        
        return np.concatenate(features_list, axis=0), sample_ids
    
    def select_diverse_samples(
        self,
        features: np.ndarray,
        sample_ids: List[str],
        num_samples: int,
        method: str = 'coreset'  # coreset, kmeans_pp
    ) -> List[str]:
        """
        选择多样化样本
        
        Args:
            features: 特征矩阵 [N, D]
            sample_ids: 样本ID列表
            num_samples: 选择数量
            method: 选择方法
            
        Returns:
            选中的样本ID列表
        """
        if method == 'coreset':
            return self._coreset_selection(features, sample_ids, num_samples)
        elif method == 'kmeans_pp':
            return self._kmeans_pp_selection(features, sample_ids, num_samples)
        else:
            return self._coreset_selection(features, sample_ids, num_samples)
    
    def _coreset_selection(
        self,
        features: np.ndarray,
        sample_ids: List[str],
        num_samples: int
    ) -> List[str]:
        """Coreset选择 - 贪婪最远点采样"""
        N = len(features)
        selected_indices = []
        
        # 随机选择第一个点
        first_idx = np.random.randint(N)
        selected_indices.append(first_idx)
        
        # 计算到已选点的最小距离
        min_distances = np.full(N, np.inf)
        
        for _ in range(num_samples - 1):
            # 更新最小距离
            last_selected = features[selected_indices[-1]]
            distances = np.linalg.norm(features - last_selected, axis=1)
            min_distances = np.minimum(min_distances, distances)
            
            # 选择最远点
            min_distances[selected_indices] = -1  # 已选择的设为-1
            next_idx = np.argmax(min_distances)
            selected_indices.append(next_idx)
        
        return [sample_ids[i] for i in selected_indices]
    
    def _kmeans_pp_selection(
        self,
        features: np.ndarray,
        sample_ids: List[str],
        num_samples: int
    ) -> List[str]:
        """K-means++初始化策略"""
        N = len(features)
        selected_indices = []
        
        # 随机选择第一个点
        first_idx = np.random.randint(N)
        selected_indices.append(first_idx)
        
        for _ in range(num_samples - 1):
            # 计算到最近已选点的距离
            min_distances = np.full(N, np.inf)
            for idx in selected_indices:
                distances = np.linalg.norm(features - features[idx], axis=1)
                min_distances = np.minimum(min_distances, distances)
            
            # 按距离的平方作为概率采样
            min_distances[selected_indices] = 0
            probs = min_distances ** 2
            probs = probs / probs.sum()
            
            next_idx = np.random.choice(N, p=probs)
            selected_indices.append(next_idx)
        
        return [sample_ids[i] for i in selected_indices]


class ActiveLearner:
    """
    主动学习器
    
    整合不确定性采样和多样性采样
    """
    
    def __init__(
        self,
        model: nn.Module,
        feature_extractor: Optional[nn.Module] = None,
        strategy: QueryStrategy = QueryStrategy.HYBRID,
        uncertainty_weight: float = 0.7,
        diversity_weight: float = 0.3,
        device: torch.device = torch.device('cuda')
    ):
        self.model = model
        self.feature_extractor = feature_extractor or model
        self.strategy = strategy
        self.uncertainty_weight = uncertainty_weight
        self.diversity_weight = diversity_weight
        self.device = device
        
        self.uncertainty_sampler = UncertaintySampler(model, device=device)
        self.diversity_sampler = DiversitySampler(feature_extractor or model, device=device)
        
        # 已标注样本
        self.labeled_samples = set()
        # 查询历史
        self.query_history = []
    
    def query(
        self,
        unlabeled_dataloader,
        num_samples: int,
        labeled_features: Optional[np.ndarray] = None
    ) -> List[SampleInfo]:
        """
        查询最有价值的样本
        
        Args:
            unlabeled_dataloader: 未标注数据加载器
            num_samples: 查询数量
            labeled_features: 已标注样本的特征（用于多样性计算）
            
        Returns:
            推荐标注的样本列表
        """
        if self.strategy == QueryStrategy.UNCERTAINTY:
            return self._uncertainty_query(unlabeled_dataloader, num_samples)
        elif self.strategy == QueryStrategy.DIVERSITY:
            return self._diversity_query(unlabeled_dataloader, num_samples)
        elif self.strategy == QueryStrategy.HYBRID:
            return self._hybrid_query(unlabeled_dataloader, num_samples, labeled_features)
        elif self.strategy == QueryStrategy.BADGE:
            return self._badge_query(unlabeled_dataloader, num_samples)
        else:
            return self._uncertainty_query(unlabeled_dataloader, num_samples)
    
    def _uncertainty_query(
        self,
        dataloader,
        num_samples: int
    ) -> List[SampleInfo]:
        """不确定性查询"""
        samples = self.uncertainty_sampler.compute_uncertainty(dataloader)
        
        # 过滤已标注样本
        samples = [s for s in samples if s.sample_id not in self.labeled_samples]
        
        return samples[:num_samples]
    
    def _diversity_query(
        self,
        dataloader,
        num_samples: int
    ) -> List[SampleInfo]:
        """多样性查询"""
        features, sample_ids = self.diversity_sampler.extract_features(dataloader)
        
        # 过滤已标注样本
        mask = [i for i, sid in enumerate(sample_ids) if sid not in self.labeled_samples]
        features = features[mask]
        sample_ids = [sample_ids[i] for i in mask]
        
        selected_ids = self.diversity_sampler.select_diverse_samples(
            features, sample_ids, num_samples
        )
        
        return [SampleInfo(sample_id=sid, uncertainty_score=0, diversity_score=1) 
                for sid in selected_ids]
    
    def _hybrid_query(
        self,
        dataloader,
        num_samples: int,
        labeled_features: Optional[np.ndarray] = None
    ) -> List[SampleInfo]:
        """混合查询 - 结合不确定性和多样性"""
        # 获取不确定性分数
        uncertainty_samples = self.uncertainty_sampler.compute_uncertainty(dataloader)
        uncertainty_dict = {s.sample_id: s for s in uncertainty_samples}
        
        # 获取特征
        features, sample_ids = self.diversity_sampler.extract_features(dataloader)
        
        # 过滤已标注样本
        mask = [i for i, sid in enumerate(sample_ids) if sid not in self.labeled_samples]
        features = features[mask]
        sample_ids = [sample_ids[i] for i in mask]
        
        # 计算多样性分数
        diversity_scores = self._compute_diversity_scores(features, labeled_features)
        
        # 归一化
        uncertainty_scores = np.array([uncertainty_dict.get(sid, SampleInfo(sid, 0)).uncertainty_score 
                                       for sid in sample_ids])
        
        if uncertainty_scores.max() > 0:
            uncertainty_scores = uncertainty_scores / uncertainty_scores.max()
        if diversity_scores.max() > 0:
            diversity_scores = diversity_scores / diversity_scores.max()
        
        # 计算组合分数
        combined_scores = (self.uncertainty_weight * uncertainty_scores + 
                          self.diversity_weight * diversity_scores)
        
        # 选择top样本
        top_indices = np.argsort(combined_scores)[::-1][:num_samples]
        
        results = []
        for idx in top_indices:
            sid = sample_ids[idx]
            sample_info = uncertainty_dict.get(sid, SampleInfo(sid, uncertainty_scores[idx]))
            sample_info.diversity_score = diversity_scores[idx]
            sample_info.combined_score = combined_scores[idx]
            results.append(sample_info)
        
        return results
    
    def _compute_diversity_scores(
        self,
        features: np.ndarray,
        labeled_features: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """计算多样性分数"""
        if labeled_features is None or len(labeled_features) == 0:
            # 没有已标注样本，使用特征的范数作为代理
            return np.linalg.norm(features, axis=1)
        
        # 计算到已标注样本的最小距离
        min_distances = np.full(len(features), np.inf)
        
        for labeled_feat in labeled_features:
            distances = np.linalg.norm(features - labeled_feat, axis=1)
            min_distances = np.minimum(min_distances, distances)
        
        return min_distances
    
    def _badge_query(
        self,
        dataloader,
        num_samples: int
    ) -> List[SampleInfo]:
        """
        BADGE: Batch Active learning by Diverse Gradient Embeddings
        
        结合不确定性和多样性的梯度嵌入方法
        """
        # 简化实现：使用最后一层梯度
        gradient_embeddings = []
        sample_ids = []
        
        self.model.train()  # 需要梯度
        
        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, dict):
                images = batch['image'].to(self.device)
                ids = batch.get('id', [f"{batch_idx}_{i}" for i in range(len(images))])
            else:
                images = batch[0].to(self.device)
                ids = [f"{batch_idx}_{i}" for i in range(len(images))]
            
            images.requires_grad = True
            
            outputs = self.model(images)
            if isinstance(outputs, dict):
                logits = outputs.get('logits')
            else:
                logits = outputs
            
            # 使用预测类别的梯度
            probs = torch.softmax(logits, dim=1)
            predictions = probs.argmax(dim=1)
            
            for i in range(len(images)):
                self.model.zero_grad()
                logits[i, predictions[i]].backward(retain_graph=True)
                
                # 获取梯度作为嵌入
                grad = images.grad[i].flatten().cpu().numpy()
                gradient_embeddings.append(grad)
                sample_ids.append(str(ids[i]))
        
        self.model.eval()
        
        gradient_embeddings = np.array(gradient_embeddings)
        
        # 过滤已标注样本
        mask = [i for i, sid in enumerate(sample_ids) if sid not in self.labeled_samples]
        gradient_embeddings = gradient_embeddings[mask]
        sample_ids = [sample_ids[i] for i in mask]
        
        # 使用k-means++选择多样化样本
        selected_ids = self.diversity_sampler._kmeans_pp_selection(
            gradient_embeddings, sample_ids, num_samples
        )
        
        return [SampleInfo(sample_id=sid, uncertainty_score=0, diversity_score=1) 
                for sid in selected_ids]
    
    def update_labeled(self, sample_ids: List[str]):
        """更新已标注样本集"""
        self.labeled_samples.update(sample_ids)
        self.query_history.append({
            'num_samples': len(sample_ids),
            'total_labeled': len(self.labeled_samples)
        })
    
    def get_labeling_priority(
        self,
        samples: List[SampleInfo],
        format: str = 'table'
    ) -> str:
        """
        生成标注优先级报告
        
        Args:
            samples: 样本列表
            format: 输出格式
            
        Returns:
            优先级报告
        """
        if format == 'table':
            lines = [
                "=" * 70,
                "标注优先级排序",
                "=" * 70,
                f"{'排名':<6}{'样本ID':<15}{'不确定性':<12}{'多样性':<12}{'综合分':<12}{'预测':<8}",
                "-" * 70
            ]
            
            for i, sample in enumerate(samples, 1):
                lines.append(
                    f"{i:<6}{sample.sample_id:<15}{sample.uncertainty_score:<12.4f}"
                    f"{sample.diversity_score:<12.4f}{sample.combined_score:<12.4f}"
                    f"{sample.predicted_class if sample.predicted_class is not None else '-':<8}"
                )
            
            return "\n".join(lines)
        
        else:
            return str([s.__dict__ for s in samples])
