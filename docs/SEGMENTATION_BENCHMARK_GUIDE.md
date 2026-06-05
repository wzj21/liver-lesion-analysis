# Segmentation Benchmark Guide

The project should not choose a segmentation algorithm only because a paper or
repository looks strong. Liver and lesion CT segmentation performance depends
on local scanner protocols, annotation quality, lesion size, phase timing,
spacing, and class imbalance. The safer engineering path is to train several
models on the same data split, export their masks, and rank them with identical
metrics.

## Candidate Algorithms

Use these candidates as the first benchmark set:

| Candidate | Why include it | How to use it |
| --- | --- | --- |
| `project_cascade_liver` | Current coarse-to-fine ConvNeXt3D cascade in this repo. | Native project baseline. |
| `external_nnunet` | Strong self-configuring medical segmentation baseline. | Train with the official nnU-Net repository, then export masks. |
| `monai_segresnet` | Strong residual 3D CNN baseline with good speed and stability. | Build through `src.segmentation.build_segmentation_model`. |
| `monai_dynunet` | MONAI configurable U-Net inspired by nnU-Net planning. | Build through the model zoo and tune kernels/strides. |
| `monai_unet` | Simple 3D U-Net sanity baseline. | Use to detect data or training pipeline problems. |
| `monai_unetr` | ViT-based 3D transformer candidate. | Use when enough data and GPU memory are available. |
| `monai_swinunetr` | Windowed-transformer 3D candidate. | Strong modern candidate, but memory-sensitive. |
| `external_mednext` | ConvNeXt-style recent medical segmentation candidate. | Train externally and evaluate exported masks. |
| `external_nnformer` | Transformer-based volumetric segmentation candidate. | Treat as a research comparison until locally validated. |
| `external_segformer3d` | Efficient transformer-style 3D segmentation candidate. | Optional recent research comparison. |

Reference starting points:

- nnU-Net: <https://github.com/MIC-DKFZ/nnUNet>
- MONAI networks: <https://docs.monai.io/en/stable/networks.html>
- MedNeXt: <https://github.com/MIC-DKFZ/MedNeXt>
- nnFormer: <https://github.com/282857341/nnFormer>
- nnFormer paper: <https://arxiv.org/abs/2109.03201>
- SegFormer3D: <https://github.com/OSUPCVLab/SegFormer3D>
- SegFormer3D paper: <https://arxiv.org/abs/2404.10156>

## Fair Experiment Protocol

Use one fixed train/validation/test split for every model. Do not let one model
train on cases that another model only sees at validation time.

Use the same input spacing, intensity windowing, label convention, and
post-processing rules. If nnU-Net performs its own preprocessing, record that
clearly and still evaluate its final exported masks in the same output space.

Report both overlap and boundary metrics:

- Dice and IoU for mask overlap.
- Precision and recall for false-positive and false-negative balance.
- HD95 for boundary quality.
- Volume similarity for clinically relevant volume agreement.
- Runtime, peak GPU memory, and model size before selecting a production model.

For lesions, stratify results by lesion size and lesion count. A model can look
good on mean Dice while still missing small lesions, which is unsafe for a real
application.

## Directory Layout

The evaluator expects one prediction folder per model:

```text
outputs/segmentation_benchmark/
  ground_truth/
    case_001.nii.gz
    case_002.nii.gz
  predictions/
    monai_segresnet/
      case_001.nii.gz
      case_002.nii.gz
    external_nnunet/
      case_001.nii.gz
      case_002.nii.gz
  metrics/
```

Supported mask formats are `.nii.gz`, `.nii`, `.npy`, and `.npz`.

## Commands

List candidate algorithms:

```powershell
python scripts\evaluate_segmentation_benchmark.py --list-models
```

Evaluate all model folders:

```powershell
python scripts\evaluate_segmentation_benchmark.py `
  --gt-dir outputs\segmentation_benchmark\ground_truth `
  --pred-root outputs\segmentation_benchmark\predictions `
  --output-dir outputs\segmentation_benchmark\metrics `
  --spacing 1.0 1.0 1.0
```

For a multi-class mask, evaluate only one label:

```powershell
python scripts\evaluate_segmentation_benchmark.py `
  --gt-dir outputs\segmentation_benchmark\ground_truth `
  --pred-root outputs\segmentation_benchmark\predictions `
  --output-dir outputs\segmentation_benchmark\metrics `
  --label 2
```

The output files are:

- `case_metrics.csv`: one row per case and model.
- `summary.csv`: one row per model, ranked by mean Dice and then mean HD95.
- `summary.json`: machine-readable ranking and skipped-case details.

## Choosing The Production Segmentor

Pick the model with the best validation and external-test evidence, not the
model with the newest architecture. A good release gate is:

1. Highest mean Dice on the fixed validation set.
2. Lowest mean HD95 if Dice is close.
3. No unacceptable small-lesion failure mode.
4. Stable runtime and memory on the target hospital workstation.
5. Same or better performance on an external test center.

After selecting the winner, wire that checkpoint into the desktop/EXE inference
configuration and keep the full benchmark outputs as model-card evidence.
