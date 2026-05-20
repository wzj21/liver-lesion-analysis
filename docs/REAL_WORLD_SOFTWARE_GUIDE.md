# 真实世界软件化使用指南

本项目现在包含一层软件化封装，用于把研究模型包装成可部署的工作流：

```
DICOM序列 / DICOM单文件 / NIfTI
  -> 预处理
  -> 肝脏分割
  -> 病灶检测与分割
  -> 病灶分类
  -> 包虫病活性判断
  -> mask + JSON + Markdown/HTML报告
```

## 1. 数据输入

推荐输入是一个完整腹部CT DICOM序列文件夹：

```text
case_001_dicom/
  IM0001.dcm
  IM0002.dcm
  ...
```

也支持：

- 单个 DICOM 文件：`.dcm`
- NIfTI：`.nii` 或 `.nii.gz`

真实应用中建议优先使用同一检查、同一序列、同一时相的DICOM目录。若是多期增强CT，应先明确动脉期、门静脉期、延迟期的组织方式，再扩展多时相输入逻辑。

## 2. 权重要求

软件推理默认要求四阶段训练权重都存在：

```text
checkpoints/stage1/best_model.pth
checkpoints/stage2/best_model.pth
checkpoints/stage3/best_model.pth
checkpoints/stage4/best_model.pth
```

这是为了避免随机初始化模型被误当成诊断系统使用。仅做流程调试时，才可以加：

```powershell
--allow_missing_weights
```

## 3. 命令行推理

在项目目录运行：

```powershell
cd D:\liver_lesion_project\liver_lesion_analysis_system_complete

python scripts\run_clinical_inference.py `
  --input D:\CT_cases\case_001_dicom `
  --output D:\liver_outputs\case_001 `
  --config configs\software_inference.yaml `
  --patient_id case_001
```

输出目录包含：

```text
result.json
report.md
report.html
masks/
  liver_mask_preprocessed.nii.gz
  lesion_01_mask_preprocessed.nii.gz
  lesion_mask_combined_preprocessed.nii.gz
```

当前导出的mask位于预处理空间。若要回写到原始DICOM空间，需要进一步增加反向重采样和DICOM SEG导出。

## 4. 大模型报告建议

项目支持可选的大模型报告建议模块。它只负责把结构化AI结果整理成更自然的报告建议，不能替代医生诊断，也不能覆盖模型分类结果。

在 `configs/software_inference.yaml` 中配置：

```yaml
llm:
  enabled: true
  api_base: "https://your-private-llm.example.com/v1"
  api_key_env: "LLM_API_KEY"
  model: "your-approved-model"
```

设置环境变量：

```powershell
$env:LLM_API_KEY="你的密钥"
```

运行时也可以覆盖：

```powershell
python scripts\run_clinical_inference.py `
  --input D:\CT_cases\case_001_dicom `
  --output D:\liver_outputs\case_001 `
  --enable_llm `
  --llm_api_base https://your-private-llm.example.com/v1 `
  --llm_model your-approved-model
```

临床或科研环境中，建议使用院内部署或合规审批后的大模型服务，避免上传可识别患者身份的信息。

## 5. 标签体系

当前 Stage3 是四分类：

| code | 类别 |
|---:|---|
| 0 | 良性肝脏病变 |
| 1 | 恶性肝脏病变 |
| 2 | 囊型包虫病 |
| 3 | 泡型包虫病 |

如果你希望区分肝血管瘤、肝囊肿、肝癌、转移瘤等更多类别，需要同步修改：

- `configs/stage3_temporal_cls.yaml` 的 `num_classes`
- `src/pipeline.py` 的类别映射
- 分类头输出维度
- 损失函数类别权重
- `scripts/prepare_data.py` 标签模板
- 报告与规则建议映射

## 6. 推荐优化路线

真实世界落地建议按这个顺序推进：

1. 数据质控：统一DICOM序列、spacing、方向、增强时相和标签标准。
2. 训练闭环：补齐Stage1/Stage2/Stage3/Stage4的可运行训练入口和验证指标。
3. 后处理：加入连通域过滤、体积阈值、最大病灶选择、mask回原图空间。
4. 报告：输出JSON、HTML/PDF、DICOM SEG/DICOM SR。
5. 部署：FastAPI服务、Docker、GPU/CPU推理配置、日志审计。
6. 临床安全：置信度阈值、强制复核策略、异常输入拒绝、版本追踪。
7. 多中心验证：按医院/设备/扫描协议分层评估泛化性能。
