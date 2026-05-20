# GitHub 仓库连接指南

本项目可以提交到 GitHub，但请注意：医学影像数据、患者信息、模型权重和推理输出不应直接提交到公开仓库。

## 1. 推荐提交内容

可以提交：

- `src/`
- `scripts/`
- `configs/`
- `docs/`
- `packaging/`
- `README.md`
- `requirements.txt`
- `setup.py`
- 测试文件

默认 `.gitignore` 已排除：

- `data/`
- `checkpoints/`
- `outputs/`
- `logs/`
- DICOM/NIfTI/NumPy数据文件
- `.pth/.pt/.onnx` 等模型文件
- 打包目录 `dist/` / `build/`

## 2. 初始化并绑定远程仓库

在 GitHub 创建一个空仓库后，复制仓库地址，例如：

```text
https://github.com/<your-name>/liver-lesion-analysis.git
```

然后运行：

```powershell
cd D:\liver_lesion_project\liver_lesion_analysis_system_complete
git init
git checkout -b codex/software-inference
git remote add origin https://github.com/<your-name>/liver-lesion-analysis.git
```

## 3. 配置提交身份

如果还没有配置 Git 用户名和邮箱：

```powershell
git config user.name "Your Name"
git config user.email "your-email@example.com"
```

## 4. 首次提交

```powershell
git add .
git status --short
git commit -m "Add clinical desktop inference workflow"
git push -u origin codex/software-inference
```

如果使用 HTTPS 推送，Git for Windows 会弹出 GitHub 登录授权窗口。也可以使用 Personal Access Token。

## 5. 大文件建议

若后续确实需要版本化模型权重，建议：

- 使用 Git LFS 管理权重文件。
- 私有仓库保存训练权重。
- 数据集使用院内对象存储、DVC 或受控数据平台，不放入普通 Git 仓库。

## 6. 临床安全

提交到 GitHub 前请确认：

- 没有患者姓名、身份证、住院号、检查号等隐私字段。
- 没有原始 DICOM、NIfTI 或报告导出。
- 配置文件中没有 API Key、Token 或内网服务地址。
- README 中明确说明软件仅为 AI 辅助，不能替代医生诊断。
