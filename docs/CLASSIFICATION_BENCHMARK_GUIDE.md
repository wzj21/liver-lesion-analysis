# Classification Benchmark Guide

The project already contains classification models:

- Stage3: `TemporalLesionClassifier` for four-class lesion diagnosis.
- Stage4: `EchinococcosisActivityNet` for active vs inactive echinococcosis.

The missing piece was a fair comparison framework. The current classifier should
be treated as the project baseline, then compared against 3D CNN, medical
pretraining, 2.5D slice aggregation, and radiomics baselines on the same
patient-level split.

## Candidate Algorithms

| Candidate | Why include it | How to use it |
| --- | --- | --- |
| `project_stage3_temporal_evidential` | Current mask-guided temporal model with uncertainty. | Native Stage3 baseline. |
| `project_stage4_activity_evidential` | Current activity classifier using boundary/internal features. | Native Stage4 baseline. |
| `monai_densenet121_3d` | Stable 3D CNN volume classifier. | Build through `src.classification.build_classification_model`. |
| `monai_resnet18_3d` | Lightweight 3D ResNet baseline. | Build through the model zoo. |
| `monai_resnet50_3d` | Higher-capacity 3D ResNet baseline. | Build through the model zoo. |
| `torchvision_r3d18` | Generic 3D ResNet video backbone. | Useful non-medical-pretraining baseline. |
| `torchvision_r2plus1d18` | 2+1D convolution baseline for CT slice stacks. | Compare with key-slice temporal modeling. |
| `external_medicalnet_resnet` | Medical-pretrained 3D ResNet transfer learning. | Train externally and export predictions. |
| `external_radimagenet_25d` | Medical-pretrained 2D/2.5D slice baseline. | Aggregate slice probabilities at patient level. |
| `external_foundation_2d_mil` | Modern 2D foundation encoder plus MIL. | Use strict patient-level split control. |
| `external_radiomics_ml` | Radiomics plus logistic/SVM/XGBoost baseline. | Important interpretable clinical baseline. |

Reference starting points:

- MONAI networks: <https://docs.monai.io/en/latest/networks.html>
- Torchvision video models: <https://docs.pytorch.org/vision/stable/models/video.html>
- MedicalNet: <https://github.com/Tencent/MedicalNet>
- RadImageNet: <https://www.radimagenet.com/>
- DINOv2: <https://github.com/facebookresearch/dinov2>
- PyRadiomics: <https://pyradiomics.readthedocs.io/>

## Prediction Format

Each model should export a `predictions.csv` file:

```text
case_id,pred,prob_0,prob_1,prob_2,prob_3
case_001,1,0.03,0.91,0.04,0.02
case_002,2,0.05,0.04,0.88,0.03
```

The labels file should contain:

```text
case_id,label
case_001,1
case_002,2
```

For Stage4 activity classification, use two probability columns:

```text
case_id,pred,prob_0,prob_1
case_010,0,0.84,0.16
```

## Directory Layout

```text
outputs/classification_benchmark/
  stage3/
    labels.csv
    predictions/
      project_stage3_temporal_evidential/
        predictions.csv
      monai_densenet121_3d/
        predictions.csv
    metrics/
  stage4/
    labels.csv
    predictions/
      project_stage4_activity_evidential/
        predictions.csv
    metrics/
```

## Commands

List candidate algorithms:

```powershell
python scripts\evaluate_classification_benchmark.py --list-models
```

Evaluate Stage3 four-class diagnosis:

```powershell
python scripts\evaluate_classification_benchmark.py `
  --labels-csv outputs\classification_benchmark\stage3\labels.csv `
  --pred-root outputs\classification_benchmark\stage3\predictions `
  --output-dir outputs\classification_benchmark\stage3\metrics `
  --num-classes 4 `
  --class-names benign,malignant,cystic_echinococcosis,alveolar_echinococcosis
```

Evaluate Stage4 activity:

```powershell
python scripts\evaluate_classification_benchmark.py `
  --labels-csv outputs\classification_benchmark\stage4\labels.csv `
  --pred-root outputs\classification_benchmark\stage4\predictions `
  --output-dir outputs\classification_benchmark\stage4\metrics `
  --num-classes 2 `
  --class-names active,inactive
```

The output files are:

- `case_predictions.csv`: one row per case and model.
- `summary.csv`: one row per model, ranked by macro AUC then macro F1.
- `summary.json`: machine-readable ranking, confusion matrices, and skipped cases.

## Choosing The Production Classifier

Do not choose the final classifier from accuracy alone. A model can look good
while missing malignant lesions or active echinococcosis cases. A safer release
gate is:

1. Highest macro AUC and macro F1 on the validation set.
2. No drop in malignant recall for Stage3.
3. No drop in active-case sensitivity for Stage4.
4. Acceptable calibration and uncertainty behavior.
5. Stable performance on an external test center.
6. Runtime compatible with the desktop/EXE workflow.

After a winner is selected, wire its checkpoint into the software inference
configuration and keep benchmark outputs as model-card evidence.
