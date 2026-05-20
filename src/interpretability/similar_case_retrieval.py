"""
相似病例检索模块
Similar Case Retrieval Module

包含:
- 特征库构建
- FAISS向量检索
- 相似病例展示
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import os
import pickle
import json
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class CaseInfo:
    """病例信息"""
    case_id: str
    diagnosis: str
    diagnosis_confidence: float
    lesion_type: str  # 良性/恶性/囊型包虫/泡型包虫
    is_active: Optional[bool]  # 活性判定（包虫病）
    volume_mm3: float
    location: str  # 肝脏位置
    patient_age: Optional[int]
    patient_gender: Optional[str]
    scan_date: Optional[str]
    image_path: Optional[str]
    features: Optional[Dict[str, float]]


class FeatureExtractor:
    """
    特征提取器
    
    从模型中提取用于检索的特征向量
    """
    
    def __init__(
        self,
        model: nn.Module,
        feature_layer: str = 'global_features',
        device: torch.device = torch.device('cuda')
    ):
        """
        Args:
            model: 特征提取模型
            feature_layer: 特征层名称
            device: 设备
        """
        self.model = model.to(device)
        self.model.eval()
        self.feature_layer = feature_layer
        self.device = device
        
        self.features = None
        self._register_hook()
    
    def _register_hook(self):
        """注册特征钩子"""
        def hook(module, input, output):
            if isinstance(output, dict):
                self.features = output.get(self.feature_layer, output.get('global_features'))
            elif isinstance(output, torch.Tensor):
                self.features = output
        
        # 找到目标层
        for name, module in self.model.named_modules():
            if self.feature_layer in name:
                module.register_forward_hook(hook)
                break
    
    @torch.no_grad()
    def extract(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        提取特征向量
        
        Args:
            input_tensor: 输入图像
            
        Returns:
            特征向量
        """
        input_tensor = input_tensor.to(self.device)
        _ = self.model(input_tensor)
        
        if self.features is not None:
            features = self.features.cpu().numpy()
            
            # 如果是多维的，展平
            if len(features.shape) > 2:
                features = features.reshape(features.shape[0], -1)
            
            # L2归一化
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            features = features / (norms + 1e-8)
            
            return features
        
        return None
    
    def extract_batch(self, dataloader) -> np.ndarray:
        """批量提取特征"""
        all_features = []
        
        for batch in dataloader:
            if isinstance(batch, dict):
                images = batch['image']
            else:
                images = batch[0]
            
            features = self.extract(images)
            if features is not None:
                all_features.append(features)
        
        return np.concatenate(all_features, axis=0)


class SimilarCaseDatabase:
    """
    相似病例数据库
    
    存储和管理病例特征
    """
    
    def __init__(
        self,
        database_path: str,
        feature_dim: int = 512
    ):
        """
        Args:
            database_path: 数据库路径
            feature_dim: 特征维度
        """
        self.database_path = database_path
        self.feature_dim = feature_dim
        
        self.features = None
        self.case_infos: List[CaseInfo] = []
        
        # 加载已有数据
        if os.path.exists(database_path):
            self.load()
    
    def add_case(
        self,
        features: np.ndarray,
        case_info: CaseInfo
    ):
        """
        添加病例
        
        Args:
            features: 特征向量 [D]
            case_info: 病例信息
        """
        features = features.reshape(1, -1)
        
        if self.features is None:
            self.features = features
        else:
            self.features = np.concatenate([self.features, features], axis=0)
        
        self.case_infos.append(case_info)
    
    def add_cases_batch(
        self,
        features: np.ndarray,
        case_infos: List[CaseInfo]
    ):
        """批量添加病例"""
        if self.features is None:
            self.features = features
        else:
            self.features = np.concatenate([self.features, features], axis=0)
        
        self.case_infos.extend(case_infos)
    
    def save(self):
        """保存数据库"""
        os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
        
        data = {
            'features': self.features,
            'case_infos': [asdict(info) for info in self.case_infos]
        }
        
        with open(self.database_path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self):
        """加载数据库"""
        if os.path.exists(self.database_path):
            with open(self.database_path, 'rb') as f:
                data = pickle.load(f)
            
            self.features = data['features']
            self.case_infos = [CaseInfo(**info) for info in data['case_infos']]
    
    def __len__(self):
        return len(self.case_infos)


class FAISSIndex:
    """
    FAISS向量索引
    
    高效的相似度检索
    """
    
    def __init__(
        self,
        feature_dim: int = 512,
        index_type: str = 'IVFFlat',  # Flat, IVFFlat, IVFPQ
        metric: str = 'cosine',  # cosine, euclidean
        nlist: int = 100,  # IVF聚类数
        nprobe: int = 10  # 搜索探测数
    ):
        """
        Args:
            feature_dim: 特征维度
            index_type: 索引类型
            metric: 距离度量
            nlist: IVF聚类数
            nprobe: 搜索时探测的聚类数
        """
        self.feature_dim = feature_dim
        self.index_type = index_type
        self.metric = metric
        self.nlist = nlist
        self.nprobe = nprobe
        
        self.index = None
        self._build_index()
    
    def _build_index(self):
        """构建索引"""
        try:
            import faiss
            
            # 选择度量
            if self.metric == 'cosine':
                # 对于cosine，使用内积（需要归一化向量）
                metric_type = faiss.METRIC_INNER_PRODUCT
            else:
                metric_type = faiss.METRIC_L2
            
            # 构建索引
            if self.index_type == 'Flat':
                self.index = faiss.IndexFlatIP(self.feature_dim) if self.metric == 'cosine' \
                    else faiss.IndexFlatL2(self.feature_dim)
            elif self.index_type == 'IVFFlat':
                quantizer = faiss.IndexFlatIP(self.feature_dim) if self.metric == 'cosine' \
                    else faiss.IndexFlatL2(self.feature_dim)
                self.index = faiss.IndexIVFFlat(quantizer, self.feature_dim, self.nlist, metric_type)
            elif self.index_type == 'IVFPQ':
                quantizer = faiss.IndexFlatL2(self.feature_dim)
                self.index = faiss.IndexIVFPQ(quantizer, self.feature_dim, self.nlist, 8, 8)
            else:
                self.index = faiss.IndexFlatL2(self.feature_dim)
            
        except ImportError:
            print("FAISS not installed, using numpy-based search")
            self.index = None
    
    def train(self, features: np.ndarray):
        """训练索引（IVF需要）"""
        if self.index is not None and hasattr(self.index, 'train'):
            import faiss
            if not self.index.is_trained:
                self.index.train(features.astype(np.float32))
    
    def add(self, features: np.ndarray):
        """添加向量"""
        features = features.astype(np.float32)
        
        if self.index is not None:
            if hasattr(self.index, 'is_trained') and not self.index.is_trained:
                self.train(features)
            self.index.add(features)
        else:
            # Numpy fallback
            if not hasattr(self, '_features'):
                self._features = features
            else:
                self._features = np.concatenate([self._features, features], axis=0)
    
    def search(
        self,
        query: np.ndarray,
        k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        搜索相似向量
        
        Args:
            query: 查询向量 [N, D] 或 [D]
            k: 返回数量
            
        Returns:
            (距离, 索引)
        """
        if len(query.shape) == 1:
            query = query.reshape(1, -1)
        
        query = query.astype(np.float32)
        
        if self.index is not None:
            import faiss
            if hasattr(self.index, 'nprobe'):
                self.index.nprobe = self.nprobe
            
            distances, indices = self.index.search(query, k)
            
            # 如果使用内积，转换为相似度
            if self.metric == 'cosine':
                similarities = distances  # 内积就是余弦相似度（归一化后）
            else:
                similarities = 1 / (1 + distances)  # 转换为相似度
            
            return similarities, indices
        else:
            # Numpy fallback
            if self.metric == 'cosine':
                similarities = np.dot(query, self._features.T)
            else:
                distances = np.linalg.norm(self._features - query, axis=1)
                similarities = 1 / (1 + distances)
            
            indices = np.argsort(-similarities, axis=1)[:, :k]
            top_similarities = np.take_along_axis(
                similarities.reshape(len(query), -1), indices, axis=1
            )
            
            return top_similarities, indices
    
    def save(self, path: str):
        """保存索引"""
        if self.index is not None:
            import faiss
            faiss.write_index(self.index, path)
        else:
            np.save(path, self._features)
    
    def load(self, path: str):
        """加载索引"""
        if os.path.exists(path):
            try:
                import faiss
                self.index = faiss.read_index(path)
            except:
                self._features = np.load(path)


class SimilarCaseRetriever:
    """
    相似病例检索器
    
    完整的检索管道
    """
    
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        database: SimilarCaseDatabase,
        index: FAISSIndex,
        top_k: int = 5
    ):
        """
        Args:
            feature_extractor: 特征提取器
            database: 病例数据库
            index: 向量索引
            top_k: 返回数量
        """
        self.feature_extractor = feature_extractor
        self.database = database
        self.index = index
        self.top_k = top_k
        
        # 如果数据库有数据，添加到索引
        if database.features is not None and len(database.features) > 0:
            self.index.add(database.features)
    
    def retrieve(
        self,
        input_tensor: torch.Tensor,
        filter_same_diagnosis: bool = False,
        predicted_diagnosis: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        检索相似病例
        
        Args:
            input_tensor: 输入图像
            filter_same_diagnosis: 是否只返回相同诊断的病例
            predicted_diagnosis: 预测诊断（用于过滤）
            
        Returns:
            相似病例列表
        """
        # 提取特征
        query_features = self.feature_extractor.extract(input_tensor)
        
        if query_features is None:
            return []
        
        # 搜索
        k = self.top_k * 2 if filter_same_diagnosis else self.top_k
        similarities, indices = self.index.search(query_features[0], k)
        
        # 构建结果
        results = []
        
        for sim, idx in zip(similarities[0], indices[0]):
            if idx < 0 or idx >= len(self.database):
                continue
            
            case_info = self.database.case_infos[idx]
            
            # 过滤
            if filter_same_diagnosis and predicted_diagnosis:
                if case_info.diagnosis != predicted_diagnosis:
                    continue
            
            results.append({
                'case_info': asdict(case_info),
                'similarity': float(sim),
                'rank': len(results) + 1
            })
            
            if len(results) >= self.top_k:
                break
        
        return results
    
    def add_new_case(
        self,
        input_tensor: torch.Tensor,
        case_info: CaseInfo
    ):
        """添加新病例到数据库"""
        features = self.feature_extractor.extract(input_tensor)
        
        if features is not None:
            self.database.add_case(features[0], case_info)
            self.index.add(features)
    
    def format_results_for_display(
        self,
        results: List[Dict[str, Any]]
    ) -> str:
        """格式化结果用于显示"""
        if not results:
            return "未找到相似病例"
        
        text = "=== 相似病例检索结果 ===\n\n"
        
        for result in results:
            info = result['case_info']
            text += f"排名 #{result['rank']} (相似度: {result['similarity']:.3f})\n"
            text += f"  病例ID: {info['case_id']}\n"
            text += f"  诊断: {info['diagnosis']} (置信度: {info['diagnosis_confidence']:.2%})\n"
            text += f"  病灶类型: {info['lesion_type']}\n"
            text += f"  体积: {info['volume_mm3']:.1f} mm³\n"
            text += f"  位置: {info['location']}\n"
            if info.get('is_active') is not None:
                text += f"  活性: {'是' if info['is_active'] else '否'}\n"
            text += "\n"
        
        return text
    
    def visualize_similar_cases(
        self,
        query_image: np.ndarray,
        results: List[Dict[str, Any]],
        load_image_fn=None
    ):
        """
        可视化相似病例
        
        Args:
            query_image: 查询图像
            results: 检索结果
            load_image_fn: 图像加载函数
        """
        import matplotlib.pyplot as plt
        
        n_results = min(len(results), 5)
        fig, axes = plt.subplots(1, n_results + 1, figsize=(4 * (n_results + 1), 4))
        
        # 查询图像
        axes[0].imshow(query_image, cmap='gray')
        axes[0].set_title('Query Image')
        axes[0].axis('off')
        
        # 相似病例
        for i, result in enumerate(results[:n_results]):
            ax = axes[i + 1]
            
            info = result['case_info']
            
            if load_image_fn and info.get('image_path'):
                try:
                    similar_image = load_image_fn(info['image_path'])
                    ax.imshow(similar_image, cmap='gray')
                except:
                    ax.text(0.5, 0.5, 'Image\nNot Available', 
                           ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, 'Image\nNot Available', 
                       ha='center', va='center', transform=ax.transAxes)
            
            ax.set_title(f"#{result['rank']} ({result['similarity']:.2f})\n{info['diagnosis']}")
            ax.axis('off')
        
        plt.tight_layout()
        return fig
