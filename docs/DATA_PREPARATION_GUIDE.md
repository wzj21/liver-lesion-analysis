# 肝脏病灶智能分析系统 - 数据准备与使用指南

## 📁 推荐的项目目录结构

```
liver_lesion_project/
│
├── liver_lesion_analysis_system/     # 代码目录（解压后的代码）
│   ├── src/
│   ├── configs/
│   └── ...
│
├── data/                              # 数据目录
│   ├── raw/                           # 原始数据
│   │   ├── images/                    # CT图像
│   │   │   ├── patient_001.nii.gz
│   │   │   ├── patient_002.nii.gz
│   │   │   └── ...
│   │   └── labels/                    # 标注文件
│   │       ├── patient_001_liver.nii.gz      # 肝脏分割标注
│   │       ├── patient_001_lesion.nii.gz     # 病灶分割标注
│   │       └── ...
│   │
│   ├── processed/                     # 预处理后的数据
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   │
│   └── annotations/                   # 标注信息
│       ├── train.json
│       ├── val.json
│       └── test.json
│
├── checkpoints/                       # 模型检查点
│   ├── stage1/
│   ├── stage2/
│   ├── stage3/
│   ├── stage4/
│   └── pretrained/                    # 预训练模型
│
├── logs/                              # 训练日志
│   └── tensorboard/
│
├── outputs/                           # 输出结果
│   ├── predictions/
│   ├── visualizations/
│   └── reports/
│
└── scripts/                           # 运行脚本
    ├── train.sh
    ├── evaluate.sh
    └── inference.sh
```

## 📋 数据格式要求

### 1. CT图像格式

支持的格式：
- **NIfTI** (.nii, .nii.gz) - 推荐
- **DICOM** (.dcm) - 需要转换
- **NumPy** (.npy, .npz)

图像要求：
- 3D CT图像
- 推荐尺寸：512×512×(64-256)
- HU值范围：-1000 ~ 3000

### 2. 标注格式

#### 分割标注（NIfTI格式）
```
标注值说明：
- 0: 背景
- 1: 肝脏
- 2: 病灶（良性）
- 3: 病灶（恶性）
- 4: 囊型包虫病灶
- 5: 泡型包虫病灶
```

#### 标注JSON格式
```json
{
    "patient_001": {
        "image_path": "images/patient_001.nii.gz",
        "liver_mask_path": "labels/patient_001_liver.nii.gz",
        "lesion_mask_path": "labels/patient_001_lesion.nii.gz",
        "diagnosis": "囊型包虫病",
        "diagnosis_code": 2,
        "is_active": true,
        "lesions": [
            {
                "id": 1,
                "type": "囊型包虫",
                "location": "右叶",
                "volume_mm3": 1250.5,
                "bbox": [100, 120, 30, 50, 60, 40]
            }
        ],
        "patient_info": {
            "age": 45,
            "gender": "M",
            "scan_date": "2024-01-15"
        }
    }
}
```

## 🔧 数据准备步骤

### 步骤1：创建目录结构

```bash
# 在你的工作目录下执行
mkdir -p liver_lesion_project/{data/{raw/{images,labels},processed/{train,val,test},annotations},checkpoints/{stage1,stage2,stage3,stage4,pretrained},logs/tensorboard,outputs/{predictions,visualizations,reports},scripts}

cd liver_lesion_project

# 解压代码
unzip liver_lesion_analysis_system_fixed.zip
```

### 步骤2：放置原始数据

将你的CT图像放入 `data/raw/images/`：
```bash
# 示例：复制数据
cp /path/to/your/ct_images/*.nii.gz data/raw/images/
cp /path/to/your/labels/*.nii.gz data/raw/labels/
```

### 步骤3：DICOM转NIfTI（如果需要）

```python
# scripts/dicom_to_nifti.py
import os
import SimpleITK as sitk

def convert_dicom_to_nifti(dicom_dir, output_path):
    """将DICOM序列转换为NIfTI"""
    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_files)
    image = reader.Execute()
    sitk.WriteImage(image, output_path)

# 使用示例
dicom_dir = "data/raw/dicom/patient_001/"
output_path = "data/raw/images/patient_001.nii.gz"
convert_dicom_to_nifti(dicom_dir, output_path)
```

### 步骤4：创建标注文件

```python
# scripts/create_annotations.py
import json
import os
import glob

def create_annotation_file(data_dir, output_path, split_ratio=(0.7, 0.15, 0.15)):
    """创建训练/验证/测试标注文件"""
    
    # 获取所有图像
    image_files = sorted(glob.glob(os.path.join(data_dir, "images", "*.nii.gz")))
    
    annotations = {}
    for img_path in image_files:
        patient_id = os.path.basename(img_path).replace(".nii.gz", "")
        
        # 检查对应的标注文件
        liver_mask = os.path.join(data_dir, "labels", f"{patient_id}_liver.nii.gz")
        lesion_mask = os.path.join(data_dir, "labels", f"{patient_id}_lesion.nii.gz")
        
        annotations[patient_id] = {
            "image_path": img_path,
            "liver_mask_path": liver_mask if os.path.exists(liver_mask) else None,
            "lesion_mask_path": lesion_mask if os.path.exists(lesion_mask) else None,
            "diagnosis": "",  # 需要手动填写
            "diagnosis_code": -1,
            "is_active": None
        }
    
    # 划分数据集
    patient_ids = list(annotations.keys())
    n = len(patient_ids)
    n_train = int(n * split_ratio[0])
    n_val = int(n * split_ratio[1])
    
    import random
    random.shuffle(patient_ids)
    
    train_ids = patient_ids[:n_train]
    val_ids = patient_ids[n_train:n_train+n_val]
    test_ids = patient_ids[n_train+n_val:]
    
    # 保存
    for split, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        split_annotations = {pid: annotations[pid] for pid in ids}
        with open(os.path.join(output_path, f"{split}.json"), "w", encoding="utf-8") as f:
            json.dump(split_annotations, f, ensure_ascii=False, indent=2)
        print(f"{split}: {len(ids)} 个样本")

# 使用
create_annotation_file("data/raw", "data/annotations")
```

### 步骤5：数据预处理

```python
# scripts/preprocess_data.py
import sys
sys.path.append("liver_lesion_analysis_system/src")

from preprocessing import PreprocessingPipeline
import nibabel as nib
import numpy as np
import json
import os

def preprocess_dataset(annotation_file, output_dir, config=None):
    """预处理整个数据集"""
    
    config = config or {
        "window_width": 160,
        "window_level": 60,
        "target_spacing": [1.0, 1.0, 2.0],
        "target_size": [256, 256, 128]
    }
    
    # 加载标注
    with open(annotation_file, "r") as f:
        annotations = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for patient_id, info in annotations.items():
        print(f"处理: {patient_id}")
        
        # 加载图像
        image = nib.load(info["image_path"]).get_fdata()
        
        # 窗宽窗位调整
        min_hu = config["window_level"] - config["window_width"] / 2
        max_hu = config["window_level"] + config["window_width"] / 2
        image = np.clip(image, min_hu, max_hu)
        image = (image - min_hu) / (max_hu - min_hu)
        
        # 保存预处理后的数据
        output_path = os.path.join(output_dir, f"{patient_id}.npz")
        
        data = {"image": image.astype(np.float32)}
        
        # 加载标注
        if info.get("liver_mask_path") and os.path.exists(info["liver_mask_path"]):
            liver_mask = nib.load(info["liver_mask_path"]).get_fdata()
            data["liver_mask"] = liver_mask.astype(np.uint8)
        
        if info.get("lesion_mask_path") and os.path.exists(info["lesion_mask_path"]):
            lesion_mask = nib.load(info["lesion_mask_path"]).get_fdata()
            data["lesion_mask"] = lesion_mask.astype(np.uint8)
        
        np.savez_compressed(output_path, **data)
        print(f"  保存到: {output_path}")

# 使用
preprocess_dataset("data/annotations/train.json", "data/processed/train")
preprocess_dataset("data/annotations/val.json", "data/processed/val")
preprocess_dataset("data/annotations/test.json", "data/processed/test")
```

## 🚀 训练模型

### 安装依赖

```bash
# 创建虚拟环境（推荐）
conda create -n liver_lesion python=3.10
conda activate liver_lesion

# 安装PyTorch（根据你的CUDA版本）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install -r liver_lesion_analysis_system/requirements.txt
```

### 训练脚本

```python
# scripts/train_stage1.py
"""Stage1: 肝脏分割训练"""
import sys
sys.path.append("liver_lesion_analysis_system/src")

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os

# 数据集
class LiverDataset(Dataset):
    def __init__(self, data_dir, annotation_file):
        import json
        with open(annotation_file) as f:
            self.annotations = json.load(f)
        self.patient_ids = list(self.annotations.keys())
        self.data_dir = data_dir
    
    def __len__(self):
        return len(self.patient_ids)
    
    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        data = np.load(os.path.join(self.data_dir, f"{patient_id}.npz"))
        
        image = torch.from_numpy(data["image"]).unsqueeze(0).float()
        mask = torch.from_numpy(data.get("liver_mask", np.zeros_like(data["image"]))).long()
        
        return {"image": image, "mask": mask, "patient_id": patient_id}

# 训练
def train_stage1():
    from models.stage1_liver_seg import CascadeSegmentationPipeline
    from losses import DiceLoss
    
    # 配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 数据
    train_dataset = LiverDataset("data/processed/train", "data/annotations/train.json")
    val_dataset = LiverDataset("data/processed/val", "data/annotations/val.json")
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4)
    
    # 模型
    model = CascadeSegmentationPipeline(in_channels=1, num_classes=2).to(device)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = DiceLoss()
    
    # 训练循环
    num_epochs = 100
    best_dice = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs["logits"], masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 验证
        model.eval()
        val_dice = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                outputs = model(images)
                # 计算Dice
                pred = outputs["logits"].argmax(dim=1)
                dice = 2 * (pred * masks).sum() / (pred.sum() + masks.sum() + 1e-8)
                val_dice += dice.item()
        
        val_dice /= len(val_loader)
        train_loss /= len(train_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Val Dice: {val_dice:.4f}")
        
        # 保存最佳模型
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dice": best_dice
            }, "checkpoints/stage1/best_model.pth")
            print(f"  保存最佳模型，Dice: {best_dice:.4f}")

if __name__ == "__main__":
    train_stage1()
```

### 运行训练

```bash
# Stage 1: 肝脏分割
python scripts/train_stage1.py

# Stage 2: 病灶检测（类似方式）
python scripts/train_stage2.py

# Stage 3: 病灶分类
python scripts/train_stage3.py

# Stage 4: 活性判定
python scripts/train_stage4.py
```

## 🔍 推理使用

```python
# scripts/inference.py
import sys
sys.path.append("liver_lesion_analysis_system/src")

import torch
import nibabel as nib
import numpy as np

def run_inference(image_path, model_path):
    """运行完整推理"""
    from pipeline import LiverLesionAnalysisPipeline
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    pipeline = LiverLesionAnalysisPipeline(
        stage1_checkpoint="checkpoints/stage1/best_model.pth",
        stage2_checkpoint="checkpoints/stage2/best_model.pth",
        stage3_checkpoint="checkpoints/stage3/best_model.pth",
        stage4_checkpoint="checkpoints/stage4/best_model.pth",
        device=device
    )
    
    # 加载图像
    image = nib.load(image_path).get_fdata()
    
    # 运行分析
    results = pipeline.analyze(image)
    
    print("=" * 50)
    print("分析结果")
    print("=" * 50)
    print(f"诊断: {results['diagnosis']}")
    print(f"置信度: {results['confidence']:.2%}")
    print(f"不确定性: {results['uncertainty']:.2%}")
    print(f"病灶数量: {results['num_lesions']}")
    if results.get('is_active') is not None:
        print(f"活性状态: {'活动期' if results['is_active'] else '静止期'}")
    
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="输入CT图像路径")
    args = parser.parse_args()
    
    run_inference(args.image, None)
```

## 📊 数据标注工具推荐

1. **3D Slicer** - 免费，功能强大
   - 下载: https://www.slicer.org/
   - 支持NIfTI、DICOM
   - 有分割编辑功能

2. **ITK-SNAP** - 轻量级
   - 下载: http://www.itksnap.org/
   - 专注于医学图像分割

3. **MITK Workbench** - 专业
   - 下载: https://www.mitk.org/

## ❓ 常见问题

### Q: 我只有DICOM格式的数据怎么办？
A: 使用上面的`dicom_to_nifti.py`脚本转换，或使用3D Slicer导出为NIfTI。

### Q: 没有分割标注，只有诊断标签？
A: 可以只训练Stage 3分类模型，或使用半监督学习方法。

### Q: 数据量很少怎么办？
A: 
1. 使用数据增强
2. 使用预训练模型
3. 使用半监督学习（代码已包含）

### Q: 如何使用GPU训练？
A: 确保安装了CUDA版本的PyTorch：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 📞 下一步

1. 准备你的数据并按照上述结构组织
2. 运行预处理脚本
3. 开始训练
4. 如有问题，随时询问！
