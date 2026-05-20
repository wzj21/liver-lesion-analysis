# Windows EXE 打包与使用指南

项目已经提供一个 Windows 桌面版入口，可以打包成双击运行的软件。

## 1. 开发环境直接打开

如果已经安装依赖，可以直接运行：

```powershell
cd D:\liver_lesion_project\liver_lesion_analysis_system_complete
python scripts\run_desktop_app.py
```

界面支持：

- 选择 DICOM 序列文件夹
- 选择 NIfTI / DICOM 单文件
- 选择输出目录
- 选择配置文件
- 启动分析
- 打开生成的 HTML 报告
- 打开输出目录

## 2. 打包成 EXE

推荐使用目录版打包，而不是单文件打包。PyTorch、SimpleITK、nibabel 等医学影像依赖较大，目录版更稳定。

```powershell
cd D:\liver_lesion_project\liver_lesion_analysis_system_complete
powershell -ExecutionPolicy Bypass -File scripts\build_windows_exe.ps1
```

生成位置：

```text
dist\LiverLesionAI\LiverLesionAI.exe
```

以后用户只需要双击：

```text
LiverLesionAI.exe
```

## 3. 模型权重

真实分析必须准备四阶段权重：

```text
checkpoints\stage1\best_model.pth
checkpoints\stage2\best_model.pth
checkpoints\stage3\best_model.pth
checkpoints\stage4\best_model.pth
```

打包后可以采用两种方式：

1. 把 `checkpoints` 文件夹复制到 `dist\LiverLesionAI\` 下。
2. 修改 `dist\LiverLesionAI\configs\software_inference.yaml`，把权重路径改成绝对路径。

默认配置会阻止缺失权重时运行，避免随机模型输出被误用。

## 4. 大模型接入

桌面版支持勾选“启用大模型报告建议”。需要先在配置中填写：

```yaml
llm:
  enabled: true
  api_base: "https://your-private-llm.example.com/v1"
  api_key_env: "LLM_API_KEY"
  model: "your-approved-model"
```

并设置环境变量：

```powershell
$env:LLM_API_KEY="你的密钥"
```

临床场景建议使用院内私有化或合规审批的大模型服务，避免上传患者可识别信息。

## 5. 注意事项

- EXE 只是软件封装，不会提升模型精度。
- 最终效果取决于训练数据质量、标注质量和多中心验证。
- 当前导出的 mask 位于预处理空间，若要对接 PACS，应继续增加原图空间回写、DICOM SEG 和 DICOM SR。
- 临床使用前必须经过验证、审批和医生复核流程。

## 6. 常见问题

### 打包后 exe 很大

正常。PyTorch、SimpleITK、医学影像库会让目录体积变大。后续可通过 ONNX Runtime、TensorRT、量化和轻量模型进一步压缩。

### 双击后报缺少权重

检查 `configs\software_inference.yaml` 中的权重路径，或把 `checkpoints` 文件夹放到 exe 同级目录。

### 只想测试界面

可以在界面里勾选“流程调试：允许缺失权重”。这只适合测试软件流程，不能用于真实分析。
