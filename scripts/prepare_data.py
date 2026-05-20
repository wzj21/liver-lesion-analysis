#!/usr/bin/env python3
"""
数据准备脚本 - 支持Excel/CSV标签表格
Data Preparation Script - Support Excel/CSV Label Table

你只需要准备一个简单的Excel或CSV表格，包含诊断信息
脚本会自动扫描图像文件并与表格匹配
"""

import os
import sys
import json
import glob
import random
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


def load_label_table(label_file: str) -> Dict[str, Dict]:
    """
    加载标签表格 (Excel 或 CSV)
    
    表格格式示例:
    | patient_id | diagnosis | diagnosis_code | is_active | age | gender |
    |------------|-----------|----------------|-----------|-----|--------|
    | patient_001| 囊型包虫病 | 4              | TRUE      | 45  | M      |
    | patient_002| 泡型包虫病 | 5              | FALSE     | 52  | F      |
    """
    labels = {}
    
    try:
        import pandas as pd
    except ImportError:
        print("请安装pandas: pip install pandas openpyxl")
        return labels
    
    # 读取文件
    try:
        if label_file.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(label_file)
        else:
            try:
                df = pd.read_csv(label_file, encoding='utf-8')
            except:
                df = pd.read_csv(label_file, encoding='gbk')
    except Exception as e:
        print(f"读取文件失败: {e}")
        return labels
    
    # 标准化列名
    df.columns = df.columns.str.strip().str.lower()
    
    # 查找patient_id列
    id_col = None
    for col in ['patient_id', 'id', 'patientid', 'case_id', '患者id', '病例id', '编号', '文件名']:
        if col in df.columns:
            id_col = col
            break
    
    if id_col is None:
        print(f"错误: 未找到患者ID列。现有列: {list(df.columns)}")
        return labels
    
    # 解析每行
    for _, row in df.iterrows():
        patient_id = str(row[id_col]).strip()
        if not patient_id or patient_id == 'nan':
            continue
        
        # 去除可能的扩展名
        patient_id = patient_id.replace('.nii.gz', '').replace('.nii', '')
        
        label_info = {}
        
        # 诊断名称
        for col in ['diagnosis', '诊断', '诊断结果', 'label', '标签']:
            if col in df.columns and pd.notna(row.get(col)):
                label_info['diagnosis'] = str(row[col]).strip()
                break
        
        # 诊断编码
        for col in ['diagnosis_code', 'code', '编码', '诊断编码', 'label_code', '类别']:
            if col in df.columns and pd.notna(row.get(col)):
                try:
                    label_info['diagnosis_code'] = int(float(row[col]))
                except:
                    pass
                break
        
        # 活性状态
        for col in ['is_active', 'active', '活性', '是否活动', '活动期']:
            if col in df.columns and pd.notna(row.get(col)):
                val = row[col]
                if isinstance(val, bool):
                    label_info['is_active'] = val
                elif isinstance(val, str):
                    label_info['is_active'] = val.lower() in ['true', 'yes', '1', '是', '活动', '活动期']
                elif isinstance(val, (int, float)):
                    label_info['is_active'] = bool(val)
                break
        
        # 年龄和性别
        for col in ['age', '年龄']:
            if col in df.columns and pd.notna(row.get(col)):
                try:
                    label_info['age'] = int(float(row[col]))
                except:
                    pass
                break
        
        for col in ['gender', 'sex', '性别']:
            if col in df.columns and pd.notna(row.get(col)):
                label_info['gender'] = str(row[col]).strip()
                break
        
        labels[patient_id] = label_info
    
    print(f"✓ 从表格加载了 {len(labels)} 个患者的标签")
    return labels


def create_template_table(output_path: str, patient_ids: List[str]):
    """创建标签模板表格"""
    try:
        import pandas as pd
    except ImportError:
        print("请安装pandas: pip install pandas openpyxl")
        return
    
    df = pd.DataFrame({
        'patient_id': patient_ids,
        'diagnosis': [''] * len(patient_ids),
        'diagnosis_code': [''] * len(patient_ids),
        'is_active': [''] * len(patient_ids),
        'age': [''] * len(patient_ids),
        'gender': [''] * len(patient_ids)
    })
    
    # 添加说明sheet
    instructions = pd.DataFrame({
        '说明': [
            'patient_id: 患者ID，与图像文件名对应',
            'diagnosis: 诊断名称（可选）',
            'diagnosis_code: 诊断编码（必填）',
            'diagnosis_code采用当前Stage3四分类体系：',
            '  0 = 良性肝脏病变',
            '  1 = 恶性肝脏病变',
            '  2 = 囊型包虫病',
            '  3 = 泡型包虫病',
            '如需细分肝血管瘤、肝囊肿等亚型，请先扩展Stage3的num_classes和标签映射。',
            'is_active: 是否活动期（包虫病填写，TRUE/FALSE）',
            'age: 年龄（可选）',
            'gender: 性别 M/F（可选）'
        ]
    })
    
    if output_path.endswith('.xlsx'):
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='标签', index=False)
                instructions.to_excel(writer, sheet_name='说明', index=False)
            print(f"✓ 模板已保存: {output_path}")
        except:
            output_path = output_path.replace('.xlsx', '.csv')
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"✓ 模板已保存为CSV: {output_path}")
    else:
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✓ 模板已保存: {output_path}")


def scan_and_match_data(labeled_dir: str, label_table: Optional[str] = None) -> Dict[str, Dict]:
    """扫描数据目录并与标签表格匹配"""
    images_dir = os.path.join(labeled_dir, "images")
    liver_masks_dir = os.path.join(labeled_dir, "liver_masks")
    lesion_masks_dir = os.path.join(labeled_dir, "lesion_masks")
    
    image_files = glob.glob(os.path.join(images_dir, "*.nii.gz"))
    image_files += glob.glob(os.path.join(images_dir, "*.nii"))
    
    if not image_files:
        print(f"⚠ 未在 {images_dir} 找到图像文件")
        return {}
    
    # 加载标签
    labels = {}
    if label_table and os.path.exists(label_table):
        labels = load_label_table(label_table)
    
    annotations = {}
    matched = 0
    
    for img_path in sorted(image_files):
        filename = os.path.basename(img_path)
        patient_id = filename.replace(".nii.gz", "").replace(".nii", "")
        
        # 检查mask
        liver_mask = None
        lesion_mask = None
        
        for ext in [".nii.gz", ".nii"]:
            if os.path.exists(os.path.join(liver_masks_dir, patient_id + ext)):
                liver_mask = patient_id + ext
                break
        
        for ext in [".nii.gz", ".nii"]:
            if os.path.exists(os.path.join(lesion_masks_dir, patient_id + ext)):
                lesion_mask = patient_id + ext
                break
        
        label_info = labels.get(patient_id, {})
        if patient_id in labels:
            matched += 1
        
        annotations[patient_id] = {
            "image": filename,
            "liver_mask": liver_mask,
            "lesion_mask": lesion_mask,
            "diagnosis": label_info.get("diagnosis", ""),
            "diagnosis_code": label_info.get("diagnosis_code", -1),
            "is_active": label_info.get("is_active"),
            "patient_info": {
                "age": label_info.get("age"),
                "gender": label_info.get("gender")
            }
        }
    
    print(f"\n扫描结果: {len(annotations)} 个图像, {matched} 个有标签")
    return annotations


def split_dataset(annotations: Dict, train_ratio=0.7, val_ratio=0.15, seed=42):
    """划分数据集"""
    random.seed(seed)
    
    # 按类别分层
    groups = {}
    for pid, info in annotations.items():
        code = info.get('diagnosis_code', -1)
        if code not in groups:
            groups[code] = []
        groups[code].append(pid)
    
    train_ids, val_ids, test_ids = [], [], []
    
    for code, pids in groups.items():
        random.shuffle(pids)
        n = len(pids)
        n_train = max(1, int(n * train_ratio))
        n_val = max(0, int(n * val_ratio)) if n > 2 else 0
        
        train_ids.extend(pids[:n_train])
        val_ids.extend(pids[n_train:n_train + n_val])
        test_ids.extend(pids[n_train + n_val:])
    
    return (
        {pid: annotations[pid] for pid in train_ids},
        {pid: annotations[pid] for pid in val_ids},
        {pid: annotations[pid] for pid in test_ids}
    )


def preprocess_data(data_dir: str, annotation_file: str, output_dir: str):
    """预处理数据"""
    try:
        import nibabel as nib
    except ImportError:
        print("请安装nibabel: pip install nibabel")
        return
    
    with open(annotation_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (patient_id, info) in enumerate(annotations.items()):
        print(f"[{i+1}/{len(annotations)}] {patient_id}", end=" ")
        
        try:
            img_path = os.path.join(data_dir, "images", info["image"])
            image = nib.load(img_path).get_fdata().astype(np.float32)
            
            # 窗宽窗位 (肝窗)
            image = np.clip(image, -20, 140)
            image = (image + 20) / 160
            
            result = {"image": image}
            
            if info.get("liver_mask"):
                mask_path = os.path.join(data_dir, "liver_masks", info["liver_mask"])
                if os.path.exists(mask_path):
                    result["liver_mask"] = nib.load(mask_path).get_fdata().astype(np.uint8)
            
            if info.get("lesion_mask"):
                mask_path = os.path.join(data_dir, "lesion_masks", info["lesion_mask"])
                if os.path.exists(mask_path):
                    result["lesion_mask"] = nib.load(mask_path).get_fdata().astype(np.uint8)
            
            np.savez_compressed(os.path.join(output_dir, f"{patient_id}.npz"), **result)
            print("✓")
        except Exception as e:
            print(f"✗ {e}")


def main():
    parser = argparse.ArgumentParser(description="数据准备（支持Excel/CSV标签）")
    parser.add_argument("--base_dir", type=str, default=".", help="项目根目录")
    parser.add_argument("--label_table", type=str, help="标签表格 (.xlsx/.csv)")
    parser.add_argument("--create_template", action="store_true", help="创建模板")
    parser.add_argument("--skip_preprocess", action="store_true", help="跳过预处理")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    
    args = parser.parse_args()
    base_dir = os.path.abspath(args.base_dir)
    labeled_dir = os.path.join(base_dir, "data/labeled")
    
    print("=" * 60)
    print("数据准备工具（支持Excel/CSV标签表格）")
    print("=" * 60)
    
    # 创建目录
    for d in ["data/labeled/images", "data/labeled/liver_masks", "data/labeled/lesion_masks",
              "data/labeled/annotations", "data/unlabeled/images",
              "data/processed/labeled/train", "data/processed/labeled/val",
              "data/processed/labeled/test", "data/processed/unlabeled/pretrain",
              "checkpoints/pretrained"]:
        os.makedirs(os.path.join(base_dir, d), exist_ok=True)
    
    # 扫描图像
    images_dir = os.path.join(labeled_dir, "images")
    image_files = glob.glob(os.path.join(images_dir, "*.nii*"))
    patient_ids = [os.path.basename(f).replace(".nii.gz", "").replace(".nii", "") 
                   for f in sorted(image_files)]
    
    if not patient_ids:
        print(f"\n⚠ 未找到图像，请将CT放入: {images_dir}")
        return
    
    print(f"\n发现 {len(patient_ids)} 个图像")
    
    # 创建模板
    if args.create_template:
        template_path = os.path.join(labeled_dir, "labels_template.xlsx")
        create_template_table(template_path, patient_ids)
        print(f"\n请填写后运行: python {sys.argv[0]} --label_table {template_path}")
        return
    
    # 查找标签文件
    label_table = args.label_table
    if not label_table:
        for name in ["labels.xlsx", "labels.csv", "标签.xlsx", "标签.csv"]:
            path = os.path.join(labeled_dir, name)
            if os.path.exists(path):
                label_table = path
                print(f"找到标签文件: {label_table}")
                break
    
    if not label_table:
        print("\n未找到标签文件，创建模板...")
        template_path = os.path.join(labeled_dir, "labels_template.xlsx")
        create_template_table(template_path, patient_ids)
        print(f"\n请填写 {template_path} 后重新运行")
        return
    
    # 处理数据
    annotations = scan_and_match_data(labeled_dir, label_table)
    
    if not annotations:
        return
    
    # 划分
    train_ann, val_ann, test_ann = split_dataset(annotations, args.train_ratio, args.val_ratio)
    
    # 保存JSON
    ann_dir = os.path.join(labeled_dir, "annotations")
    for name, ann in [("train", train_ann), ("val", val_ann), ("test", test_ann)]:
        with open(os.path.join(ann_dir, f"{name}.json"), 'w', encoding='utf-8') as f:
            json.dump(ann, f, ensure_ascii=False, indent=2)
        print(f"✓ {name}.json: {len(ann)} 个样本")
    
    # 预处理
    if not args.skip_preprocess:
        for split in ["train", "val", "test"]:
            print(f"\n预处理 {split}...")
            preprocess_data(
                labeled_dir,
                os.path.join(ann_dir, f"{split}.json"),
                os.path.join(base_dir, f"data/processed/labeled/{split}")
            )
    
    print("\n" + "=" * 60)
    print("✓ 完成!")


if __name__ == "__main__":
    main()
