"""
可解释性模块
Interpretability Module

本模块提供完整的AI模型可解释性解决方案，帮助临床医生理解模型决策：

主要功能:
1. Grad-CAM可视化
   - Grad-CAM: 基础梯度加权类激活映射
   - Grad-CAM++: 改进版，更好处理多目标
   - Score-CAM: 无梯度版本
   - Layer-CAM: 适用于浅层

2. 注意力可视化
   - Transformer自注意力矩阵可视化
   - Deformable Attention采样点可视化
   - 切片级注意力权重可视化
   - 注意力流动分析

3. SHAP特征贡献分析
   - Deep SHAP
   - Gradient SHAP
   - 特征重要性排序
   - 特征组贡献分析
   - 对比解释

4. 相似病例检索
   - 特征提取与存储
   - FAISS向量索引
   - 高效相似度检索
   - 可视化对比

5. 解释报告生成
   - 自然语言解释
   - 不确定性分析
   - 临床建议生成
   - HTML/JSON报告导出

使用示例:
---------
# 1. Grad-CAM可视化
from interpretability import CAMGenerator

cam_gen = CAMGenerator(model, target_layers=['backbone.stages.3'], method='grad_cam_plus_plus')
cams = cam_gen.generate(input_tensor)
visualization = cam_gen.visualize(image, cams['backbone.stages.3'])

# 2. 相似病例检索
from interpretability import SimilarCaseRetriever, FeatureExtractor, SimilarCaseDatabase, FAISSIndex

extractor = FeatureExtractor(model)
database = SimilarCaseDatabase('cases.pkl')
index = FAISSIndex(feature_dim=512)
retriever = SimilarCaseRetriever(extractor, database, index)

similar_cases = retriever.retrieve(input_tensor)

# 3. 完整分析管道
from interpretability import InterpretabilityPipeline

pipeline = InterpretabilityPipeline(
    model=model,
    cam_generator=cam_gen,
    case_retriever=retriever
)

report = pipeline.analyze(input_tensor, patient_id='P001')
html_path = pipeline.generate_report(report, format='html')
"""

from .grad_cam import (
    GradCAM,
    GradCAMPlusPlus,
    ScoreCAM,
    LayerCAM,
    CAMGenerator
)

from .attention_visualization import (
    AttentionExtractor,
    TransformerAttentionVisualizer,
    DeformableAttentionVisualizer,
    SliceAttentionVisualizer
)

from .shap_analysis import (
    DeepSHAP,
    GradientSHAP,
    FeatureContributionAnalyzer,
    ContrastiveExplanation
)

from .similar_case_retrieval import (
    CaseInfo,
    FeatureExtractor,
    SimilarCaseDatabase,
    FAISSIndex,
    SimilarCaseRetriever
)

from .report_generator import (
    InterpretabilityReport,
    ExplanationGenerator,
    RecommendationEngine,
    ReportExporter,
    InterpretabilityPipeline
)


__all__ = [
    # Grad-CAM
    'GradCAM',
    'GradCAMPlusPlus',
    'ScoreCAM',
    'LayerCAM',
    'CAMGenerator',
    
    # Attention Visualization
    'AttentionExtractor',
    'TransformerAttentionVisualizer',
    'DeformableAttentionVisualizer',
    'SliceAttentionVisualizer',
    
    # SHAP Analysis
    'DeepSHAP',
    'GradientSHAP',
    'FeatureContributionAnalyzer',
    'ContrastiveExplanation',
    
    # Similar Case Retrieval
    'CaseInfo',
    'FeatureExtractor',
    'SimilarCaseDatabase',
    'FAISSIndex',
    'SimilarCaseRetriever',
    
    # Report Generation
    'InterpretabilityReport',
    'ExplanationGenerator',
    'RecommendationEngine',
    'ReportExporter',
    'InterpretabilityPipeline'
]

__version__ = '1.0.0'
