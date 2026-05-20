"""
肝脏病灶智能分析系统
Liver Lesion Intelligent Analysis System

一个完整的基于深度学习的肝脏病灶检测、分割、分类和活性判定系统。

主要功能:
- Stage 1: 肝脏分割 (U-Net + Multi-Phase Fusion)
- Stage 2: 病灶检测与分割 (Deformable DETR)  
- Stage 3: 病灶分类 (Evidential Deep Learning)
- Stage 4: 包虫病活性判定 (Boundary Analysis)

技术特性:
- 3D ConvNeXt骨干网络
- Evidential不确定性估计
- 半监督学习支持
- 知识蒸馏与模型轻量化
- 完整的可解释性分析
- 多中心验证与主动学习
- Docker容器化部署

Version: 1.0.0
"""

__version__ = '1.0.0'
__author__ = 'AI Medical Imaging Team'

__all__ = [
    'semi_supervised',
    'distillation', 
    'interpretability',
    'visualization',
    'validation',
    'deployment',
    'software',
]


def __getattr__(name):
    """Lazy-load heavy subpackages.

    Importing ``src`` should not immediately require torch or optional
    deployment dependencies. Subpackages are imported only when requested.
    """
    if name in __all__:
        import importlib

        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
