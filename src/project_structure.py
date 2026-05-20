"""
肝脏病灶智能分析系统 - 项目结构
Liver Lesion Intelligent Analysis System - Project Structure

Version: 1.0.0
"""

PROJECT_STRUCTURE = """
liver_lesion_analysis_system/
│
├── configs/                              # 配置文件
│   ├── model_config.yaml                 # 模型配置
│   ├── training_config.yaml              # 训练配置
│   ├── semi_supervised_config.yaml       # 半监督学习配置
│   ├── distillation_config.yaml          # 知识蒸馏配置
│   ├── interpretability_config.yaml      # 可解释性配置
│   └── deployment_config.yaml            # 部署配置
│
├── src/                                  # 源代码
│   │
│   ├── models/                           # 模型定义
│   │   ├── __init__.py
│   │   ├── backbone/                     # 骨干网络
│   │   │   ├── convnext_3d.py           # 3D ConvNeXt
│   │   │   └── swin_transformer_3d.py   # 3D Swin Transformer
│   │   ├── stage1_segmentation/          # Stage1: 肝脏分割
│   │   │   ├── liver_segmentor.py
│   │   │   └── multi_phase_fusion.py
│   │   ├── stage2_detection/             # Stage2: 病灶检测分割
│   │   │   ├── deformable_detr.py
│   │   │   └── lesion_segmentor.py
│   │   ├── stage3_classification/        # Stage3: 病灶分类
│   │   │   ├── evidential_classifier.py
│   │   │   └── temporal_transformer.py
│   │   ├── stage4_activity/              # Stage4: 活性判定
│   │   │   ├── activity_classifier.py
│   │   │   └── boundary_analyzer.py
│   │   └── full_model.py                 # 完整模型集成
│   │
│   ├── losses/                           # 损失函数
│   │   ├── __init__.py
│   │   ├── dice_loss.py
│   │   ├── focal_loss.py
│   │   ├── evidential_loss.py
│   │   └── contrastive_loss.py
│   │
│   ├── data/                             # 数据处理
│   │   ├── __init__.py
│   │   ├── dataset.py                    # 数据集定义
│   │   ├── preprocessing.py              # 预处理
│   │   ├── augmentation.py               # 数据增强
│   │   └── data_loader.py                # 数据加载器
│   │
│   ├── training/                         # 训练模块
│   │   ├── __init__.py
│   │   ├── trainer.py                    # 训练器
│   │   ├── evaluator.py                  # 评估器
│   │   └── callbacks.py                  # 回调函数
│   │
│   ├── semi_supervised/                  # 半监督学习
│   │   ├── __init__.py
│   │   ├── base.py                       # 基础组件
│   │   ├── teacher_student.py            # Teacher-Student
│   │   ├── fixmatch.py                   # FixMatch
│   │   ├── cross_pseudo_supervision.py   # CPS
│   │   ├── augmentation.py               # 数据增强
│   │   └── trainer.py                    # 训练器
│   │
│   ├── distillation/                     # 知识蒸馏
│   │   ├── __init__.py
│   │   ├── losses.py                     # 蒸馏损失
│   │   ├── student_models.py             # 轻量级模型
│   │   ├── trainer.py                    # 蒸馏训练器
│   │   └── quantization.py               # 量化优化
│   │
│   ├── interpretability/                 # 可解释性
│   │   ├── __init__.py
│   │   ├── grad_cam.py                   # Grad-CAM
│   │   ├── attention_visualization.py   # 注意力可视化
│   │   ├── shap_analysis.py              # SHAP分析
│   │   ├── similar_case_retrieval.py     # 相似病例检索
│   │   └── report_generator.py           # 报告生成
│   │
│   ├── visualization/                    # 可视化
│   │   ├── __init__.py
│   │   ├── volume_rendering.py           # 3D渲染
│   │   └── structured_report.py          # 结构化报告
│   │
│   ├── validation/                       # 验证模块
│   │   ├── __init__.py
│   │   ├── multi_center.py               # 多中心验证
│   │   └── active_learning.py            # 主动学习
│   │
│   ├── deployment/                       # 部署模块
│   │   ├── __init__.py
│   │   ├── server.py                     # 模型服务器
│   │   └── api.py                        # REST API
│   │
│   └── utils/                            # 工具函数
│       ├── __init__.py
│       ├── metrics.py                    # 评估指标
│       ├── logger.py                     # 日志工具
│       └── io.py                         # IO工具
│
├── scripts/                              # 脚本文件
│   ├── train.py                          # 训练脚本
│   ├── evaluate.py                       # 评估脚本
│   ├── inference.py                      # 推理脚本
│   ├── export_model.py                   # 模型导出
│   └── generate_report.py                # 报告生成
│
├── tests/                                # 测试代码
│   ├── test_models.py
│   ├── test_losses.py
│   └── test_inference.py
│
├── notebooks/                            # Jupyter笔记本
│   ├── data_exploration.ipynb
│   ├── model_analysis.ipynb
│   └── visualization_demo.ipynb
│
├── deployment/                           # 部署文件
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── requirements.txt
│   ├── app.py
│   └── README.md
│
├── docs/                                 # 文档
│   ├── architecture.md                   # 架构文档
│   ├── api_reference.md                  # API参考
│   ├── user_guide.md                     # 用户指南
│   └── deployment_guide.md               # 部署指南
│
├── checkpoints/                          # 模型检查点
│   ├── stage1/
│   ├── stage2/
│   ├── stage3/
│   ├── stage4/
│   └── full_model/
│
├── logs/                                 # 日志文件
│
├── outputs/                              # 输出结果
│   ├── predictions/
│   ├── visualizations/
│   └── reports/
│
├── requirements.txt                      # Python依赖
├── setup.py                              # 安装脚本
├── README.md                             # 项目说明
└── LICENSE                               # 许可证
"""

# 模块说明
MODULE_DESCRIPTIONS = {
    "models": "模型定义模块，包含4个Stage的完整模型架构",
    "losses": "损失函数模块，包含Dice、Focal、Evidential等损失",
    "data": "数据处理模块，包含数据集、预处理、增强",
    "training": "训练模块，包含训练器、评估器、回调",
    "semi_supervised": "半监督学习模块，支持CPS、FixMatch、Teacher-Student",
    "distillation": "知识蒸馏模块，支持特征蒸馏、输出蒸馏、模型量化",
    "interpretability": "可解释性模块，支持Grad-CAM、SHAP、相似病例检索",
    "visualization": "可视化模块，支持3D渲染、结构化报告",
    "validation": "验证模块，支持多中心验证、主动学习",
    "deployment": "部署模块，支持REST API、Docker部署",
    "utils": "工具模块，包含评估指标、日志、IO等"
}

# 模型版本
MODEL_VERSIONS = {
    "full": {
        "params": "62M",
        "description": "完整版模型，最高精度",
        "inference_time_gpu": "<4s",
        "inference_time_cpu": "<10s"
    },
    "lightweight_fp32": {
        "params": "25M", 
        "description": "轻量版FP32模型",
        "inference_time_gpu": "<2s",
        "inference_time_cpu": "<5s"
    },
    "lightweight_fp16": {
        "params": "25M",
        "description": "轻量版FP16模型",
        "inference_time_gpu": "<1.5s",
        "inference_time_cpu": "<4s"
    },
    "lightweight_int8": {
        "params": "25M",
        "description": "轻量版INT8量化模型",
        "inference_time_gpu": "<1s",
        "inference_time_cpu": "<3s"
    }
}

def print_structure():
    """打印项目结构"""
    print(PROJECT_STRUCTURE)

def print_module_descriptions():
    """打印模块说明"""
    print("\n模块说明:")
    print("=" * 60)
    for module, desc in MODULE_DESCRIPTIONS.items():
        print(f"  {module}: {desc}")

def print_model_versions():
    """打印模型版本"""
    print("\n模型版本:")
    print("=" * 60)
    for version, info in MODEL_VERSIONS.items():
        print(f"\n  {version}:")
        for key, value in info.items():
            print(f"    {key}: {value}")


if __name__ == "__main__":
    print_structure()
    print_module_descriptions()
    print_model_versions()
