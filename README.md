# 肝脏病灶智能分析系统 (Liver Lesion Analysis System)

## 概述

基于深度学习的肝脏病灶智能分析系统，支持**肝脏分割、病灶检测分类、包虫病分型与活性判定**。
采用4阶段级联架构，全流程端到端推理，自动输出结构化报告与可解释性分析。

## 核心特性

- **自监督预训练**: 3D MAE + 对比学习 (SimCLR/MoCo)
- **骨干网络**: 3D ConvNeXt (支持 tiny/small/base/large)
- **级联分割**: 粗-细级联 U-Net + Attention Gate + 边界增强
- **病灶检测**: 3D Deformable DETR + 可变形分割头
- **时序分类**: Mask-Guided Encoder + Temporal Transformer + 形态特征融合
- **不确定性估计**: Evidential Learning (Dirichlet分布)
- **可解释性**: Grad-CAM、SHAP、注意力可视化、相似病例检索
- **半监督学习**: Teacher-Student / FixMatch / Cross Pseudo Supervision
- **知识蒸馏**: 特征蒸馏 + 量化 (INT8/FP16)

## 系统流程

```
Phase 0: 自监督预训练 (3D MAE / 对比学习)
    ↓
CT扫描输入 → 预处理 (窗宽窗位标准化 + 重采样)
    ↓
Stage1: 3D肝脏分割 (粗-细级联, Dice>0.97)
    ↓
Stage2: 病灶检测 + Deformable分割 (mAP@0.5>0.90)
    ↓
Stage3: 时序增强的四分类 (良性/恶性/囊型包虫/泡型包虫, Acc>0.93)
    ↓
Stage4: 包虫病活性判定 (活动性/非活动性, Acc>0.91)
    ↓
输出: 结构化报告 + 3D可视化 + 可解释性分析 + 不确定性评估
```

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

### 推理

```python
from src.pipeline import LiverLesionAnalysisPipeline

pipeline = LiverLesionAnalysisPipeline(config_path='configs/global_config.yaml')
pipeline.load_weights(
    stage1_path='checkpoints/stage1/best_model.pth',
    stage2_path='checkpoints/stage2/best_model.pth',
    stage3_path='checkpoints/stage3/best_model.pth',
    stage4_path='checkpoints/stage4/best_model.pth',
)

results = pipeline.analyze(ct_path='path/to/ct.nii.gz')
report = pipeline.generate_report(results, output_path='report.txt')
print(report)
```

### 软件化DICOM推理

训练好四阶段权重后，可以直接输入DICOM序列文件夹并导出分割、分类和报告：

```powershell
python scripts\run_clinical_inference.py `
  --input D:\CT_cases\case_001_dicom `
  --output D:\liver_outputs\case_001 `
  --config configs\software_inference.yaml `
  --patient_id case_001
```

输出包含 `result.json`、`report.md`、`report.html` 和预处理空间下的 NIfTI mask。
详细说明见 `docs/REAL_WORLD_SOFTWARE_GUIDE.md`。

### Windows桌面版 / EXE

开发环境可直接启动桌面界面：

```powershell
python scripts\run_desktop_app.py
```

打包为可双击运行的软件：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_windows_exe.ps1
```

生成文件位于 `dist\LiverLesionAI\LiverLesionAI.exe`。详细说明见
`docs/WINDOWS_EXE_GUIDE.md`。

### 自监督预训练

```bash
python scripts/pretrain.py --method mae --data_dir /data/unlabeled_ct \
    --output_dir checkpoints/pretrained --epochs 400
```

### 分阶段训练

```bash
python scripts/train.py --stage 1 --config configs/stage1_liver_seg.yaml
python scripts/train.py --stage 2 --config configs/stage2_det_cls_deform_seg.yaml
python scripts/train.py --stage 3 --config configs/stage3_temporal_cls.yaml
python scripts/train.py --stage 4 --config configs/stage4_activity.yaml
```

## 性能指标

| Stage | 任务 | 指标 | 目标值 |
|-------|------|------|--------|
| Stage1 | 肝脏分割 | Dice | >0.975 |
| Stage2 | 病灶检测 | mAP@0.5 | >0.90 |
| Stage2 | 病灶分割 | Dice | >0.88 |
| Stage3 | 四分类 | Accuracy | >0.93 |
| Stage4 | 活性判定 | Accuracy | >0.91 |

## 项目结构

```
├── configs/                          # 各阶段配置文件
├── scripts/
│   ├── prepare_data.py              # 数据准备
│   ├── pretrain.py                  # 自监督预训练
│   └── train.py                     # 分阶段训练
├── src/
│   ├── backbones/                   # 3D ConvNeXt骨干网络
│   ├── models/
│   │   ├── stage1_liver_seg/        # 粗-细级联肝脏分割
│   │   ├── stage2_det_cls_deform_seg/ # Deformable DETR检测分割
│   │   ├── stage3_temporal_cls/     # 时序增强四分类
│   │   └── stage4_activity/         # 包虫病活性判定
│   ├── losses/                      # Dice/Tversky/HD/Evidential/Detection损失
│   ├── preprocessing/               # 窗宽窗位/重采样/预处理流水线
│   ├── pretraining/                 # MAE + 对比学习
│   ├── semi_supervised/             # Teacher-Student/FixMatch/CPS
│   ├── uncertainty/                 # Evidential不确定性
│   ├── interpretability/            # Grad-CAM/SHAP/注意力/相似病例
│   ├── distillation/                # 知识蒸馏 + 量化
│   ├── visualization/               # 3D渲染 + 结构化报告
│   ├── validation/                  # 多中心验证 + 主动学习
│   ├── data/                        # 数据集 + 增强
│   ├── deployment/                  # 推理服务部署
│   └── pipeline.py                  # 端到端推理流水线
├── docs/                            # 数据准备指南
├── requirements.txt
├── setup.py
└── test_system.py                   # 系统测试
```

## 许可证

MIT License
