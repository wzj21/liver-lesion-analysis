# 数据组织指南 - 有标签 + 无标签数据

## 📁 推荐目录结构

```
liver_lesion_project/
│
├── liver_lesion_analysis_system/      # 代码（解压后）
│
├── data/
│   │
│   ├── labeled/                       # ===== 有标签数据 =====
│   │   ├── images/                    # CT图像
│   │   │   ├── patient_001.nii.gz
│   │   │   ├── patient_002.nii.gz
│   │   │   └── ...
│   │   │
│   │   ├── liver_masks/               # 肝脏分割标注
│   │   │   ├── patient_001.nii.gz
│   │   │   ├── patient_002.nii.gz
│   │   │   └── ...
│   │   │
│   │   ├── lesion_masks/              # 病灶分割标注
│   │   │   ├── patient_001.nii.gz
│   │   │   ├── patient_002.nii.gz
│   │   │   └── ...
│   │   │
│   │   └── annotations/               # 标注信息
│   │       ├── train.json             # 训练集标注
│   │       ├── val.json               # 验证集标注
│   │       └── test.json              # 测试集标注
│   │
│   ├── unlabeled/                     # ===== 无标签数据（自监督预训练）=====
│   │   ├── images/                    # CT图像
│   │   │   ├── unlabeled_001.nii.gz
│   │   │   ├── unlabeled_002.nii.gz
│   │   │   ├── unlabeled_003.nii.gz
│   │   │   └── ...
│   │   │
│   │   └── file_list.txt              # 无标签数据列表
│   │
│   └── processed/                     # ===== 预处理后的数据 =====
│       ├── labeled/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── unlabeled/
│           └── pretrain/
│
├── checkpoints/                       # 模型保存
│   ├── pretrained/                    # 自监督预训练模型
│   ├── stage1/
│   ├── stage2/
│   ├── stage3/
│   └── stage4/
│
└── logs/                              # 日志
```

---

## 📝 文件格式说明

### 1. 有标签数据的标注JSON格式

**`data/labeled/annotations/train.json`** 示例：

```json
{
    "patient_001": {
        "image": "patient_001.nii.gz",
        "liver_mask": "patient_001.nii.gz",
        "lesion_mask": "patient_001.nii.gz",
        "diagnosis": "囊型包虫病",
        "diagnosis_code": 2,
        "is_active": true,
        "lesions": [
            {
                "id": 1,
                "type": "囊型包虫",
                "location": "肝右叶",
                "volume_mm3": 1520.5
            }
        ],
        "patient_info": {
            "age": 45,
            "gender": "M"
        }
    },
    "patient_002": {
        "image": "patient_002.nii.gz",
        "liver_mask": "patient_002.nii.gz",
        "lesion_mask": "patient_002.nii.gz",
        "diagnosis": "泡型包虫病",
        "diagnosis_code": 3,
        "is_active": false
    }
}
```

### 2. 无标签数据列表

**`data/unlabeled/file_list.txt`** 示例：

```
unlabeled_001.nii.gz
unlabeled_002.nii.gz
unlabeled_003.nii.gz
unlabeled_004.nii.gz
...
```

### 3. 诊断编码对照表

| diagnosis_code | 诊断 | 说明 |
|----------------|------|------|
| 0 | 良性肝脏病变 | 良性倾向病变，若要细分血管瘤/囊肿需扩展类别 |
| 1 | 恶性肝脏病变 | 恶性倾向病变 |
| 2 | 囊型包虫病 | CE分型 |
| 3 | 泡型包虫病 | AE分型 |

> 注意：当前 `Stage3` 模型是四分类。如果要训练 6 类或更多亚型，必须同步修改
> `configs/stage3_temporal_cls.yaml`、`src/pipeline.py` 的类别映射、分类头输出维度、
> 损失函数和报告映射。

---

## 🔧 数据放置步骤

### 步骤1：创建目录

```bash
# 在你选择的位置创建项目目录
mkdir -p liver_lesion_project
cd liver_lesion_project

# 创建数据目录结构
mkdir -p data/labeled/{images,liver_masks,lesion_masks,annotations}
mkdir -p data/unlabeled/images
mkdir -p data/processed/{labeled/{train,val,test},unlabeled/pretrain}
mkdir -p checkpoints/{pretrained,stage1,stage2,stage3,stage4}
mkdir -p logs

# 解压代码
unzip liver_lesion_analysis_system_complete.zip
```

### 步骤2：放置有标签数据

```bash
# 复制你的有标签CT图像
cp /你的数据路径/labeled_images/*.nii.gz data/labeled/images/

# 复制肝脏分割标注
cp /你的数据路径/liver_segmentations/*.nii.gz data/labeled/liver_masks/

# 复制病灶分割标注
cp /你的数据路径/lesion_segmentations/*.nii.gz data/labeled/lesion_masks/
```

### 步骤3：放置无标签数据

```bash
# 复制无标签CT图像
cp /你的数据路径/unlabeled_images/*.nii.gz data/unlabeled/images/

# 生成文件列表
ls data/unlabeled/images/ > data/unlabeled/file_list.txt
```

### 步骤4：创建标注文件

运行下面的脚本自动生成标注JSON：

---

## 🐍 数据准备脚本

将以下脚本保存并运行：
