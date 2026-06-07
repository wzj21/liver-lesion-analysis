"""
Model catalog for liver lesion classification experiments.

The project has several native classification tasks:

1. Stage3: four-class lesion diagnosis.
2. Echinococcosis subtype binary models:
   - hepatic cystic echinococcosis (CE) vs non-CE
   - hepatic alveolar echinococcosis (AE) vs non-AE
3. Optional Stage4: active vs inactive echinococcosis assessment.

This registry adds common 3D/2.5D comparison baselines while keeping imports
lazy so CI can list candidates without torch, torchvision, or MONAI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple


ConfigDict = Mapping[str, Any]


@dataclass(frozen=True)
class ClassificationModelSpec:
    """Metadata for one classification candidate."""

    name: str
    family: str
    implementation: str
    status: str
    recommended_task: str
    strengths: Tuple[str, ...]
    caveats: Tuple[str, ...]
    config: ConfigDict
    references: Tuple[str, ...] = ()


_MODEL_SPECS: Dict[str, ClassificationModelSpec] = {
    "project_stage3_temporal_evidential": ClassificationModelSpec(
        name="project_stage3_temporal_evidential",
        family="Mask-guided temporal evidential classifier",
        implementation="native",
        status="buildable",
        recommended_task="Stage3 four-class liver lesion diagnosis.",
        strengths=(
            "Uses lesion masks, key slices, morphology, and temporal attention.",
            "Outputs uncertainty through evidential learning for clinical review.",
        ),
        caveats=(
            "Needs complete training and external validation before clinical use.",
            "Performance depends on segmentation quality and morphology features.",
        ),
        config={
            "num_classes": 4,
            "slice_encoder_variant": "tiny",
            "include_boundary": True,
            "num_morphology_features": 16,
            "morphology_output_dim": 256,
            "temporal_hidden_dim": 512,
            "temporal_num_layers": 4,
            "temporal_num_heads": 8,
            "classifier_hidden_dims": (256, 128, 64),
        },
        references=("src.models.stage3_temporal_cls.TemporalLesionClassifier",),
    ),
    "project_ce_binary_evidential": ClassificationModelSpec(
        name="project_ce_binary_evidential",
        family="Mask-guided temporal evidential binary classifier",
        implementation="native",
        status="buildable",
        recommended_task="Hepatic cystic echinococcosis binary classification: CE vs non-CE.",
        strengths=(
            "Keeps the CE decision as its own calibrated binary model.",
            "Can optimize CE sensitivity without forcing a four-class trade-off.",
        ),
        caveats=(
            "Requires patient-level labels mapped to non-CE vs CE.",
            "Should be validated separately from the AE binary model.",
        ),
        config={
            "num_classes": 2,
            "positive_class": "cystic_echinococcosis",
            "class_names": ("non_cystic_echinococcosis", "cystic_echinococcosis"),
            "slice_encoder_variant": "tiny",
            "include_boundary": True,
            "num_morphology_features": 16,
            "morphology_output_dim": 256,
            "temporal_hidden_dim": 512,
            "temporal_num_layers": 4,
            "temporal_num_heads": 8,
            "classifier_hidden_dims": (256, 128, 64),
        },
        references=("src.models.stage3_temporal_cls.TemporalLesionClassifier",),
    ),
    "project_ae_binary_evidential": ClassificationModelSpec(
        name="project_ae_binary_evidential",
        family="Mask-guided temporal evidential binary classifier",
        implementation="native",
        status="buildable",
        recommended_task="Hepatic alveolar echinococcosis binary classification: AE vs non-AE.",
        strengths=(
            "Keeps the AE decision as its own calibrated binary model.",
            "Can optimize AE sensitivity for infiltrative lesions independently.",
        ),
        caveats=(
            "Requires patient-level labels mapped to non-AE vs AE.",
            "Should be validated separately from the CE binary model.",
        ),
        config={
            "num_classes": 2,
            "positive_class": "alveolar_echinococcosis",
            "class_names": ("non_alveolar_echinococcosis", "alveolar_echinococcosis"),
            "slice_encoder_variant": "tiny",
            "include_boundary": True,
            "num_morphology_features": 16,
            "morphology_output_dim": 256,
            "temporal_hidden_dim": 512,
            "temporal_num_layers": 4,
            "temporal_num_heads": 8,
            "classifier_hidden_dims": (256, 128, 64),
        },
        references=("src.models.stage3_temporal_cls.TemporalLesionClassifier",),
    ),
    "project_stage4_activity_evidential": ClassificationModelSpec(
        name="project_stage4_activity_evidential",
        family="Boundary/internal activity classifier",
        implementation="native",
        status="buildable",
        recommended_task="Optional active vs inactive echinococcosis assessment after CE/AE subtype classification.",
        strengths=(
            "Combines boundary, internal structure, lesion type, and Stage3 features.",
            "Clinically aligned with active vs inactive follow-up decisions.",
        ),
        caveats=(
            "This is an activity classifier, not the CE-vs-AE subtype classifier.",
            "Only applies after an echinococcosis class is suspected or confirmed.",
            "Needs WHO-IWGE/PNM-aligned labels for trustworthy training.",
        ),
        config={
            "num_classes": 2,
            "boundary_output_dim": 128,
            "internal_output_dim": 128,
            "stage3_feature_dim": 256,
            "classifier_hidden_dims": (128, 64),
            "use_3d": True,
        },
        references=("src.models.stage4_activity.EchinococcosisActivityNet",),
    ),
    "monai_densenet121_3d": ClassificationModelSpec(
        name="monai_densenet121_3d",
        family="3D DenseNet121",
        implementation="monai",
        status="buildable",
        recommended_task="Volume-level CT classification baseline.",
        strengths=(
            "Well-established CNN baseline for volumetric classification.",
            "Often stable on small to medium medical datasets.",
        ),
        caveats=(
            "Does not explicitly use lesion masks unless the input is cropped or masked.",
            "Needs careful foreground cropping to avoid learning scanner/background bias.",
        ),
        config={"spatial_dims": 3, "pretrained": False},
        references=("https://docs.monai.io/en/latest/networks.html",),
    ),
    "monai_resnet18_3d": ClassificationModelSpec(
        name="monai_resnet18_3d",
        family="3D ResNet18",
        implementation="monai",
        status="buildable",
        recommended_task="Lightweight 3D CNN baseline for CT volume or lesion ROI classification.",
        strengths=(
            "Fast, robust baseline for ablation experiments.",
            "Good first 3D model when data size or GPU memory is limited.",
        ),
        caveats=(
            "May underfit subtle enhancement patterns compared with richer models.",
            "Usually benefits from medical pretraining or strong augmentation.",
        ),
        config={"spatial_dims": 3, "shortcut_type": "B"},
        references=("https://docs.monai.io/en/latest/networks.html",),
    ),
    "monai_resnet50_3d": ClassificationModelSpec(
        name="monai_resnet50_3d",
        family="3D ResNet50",
        implementation="monai",
        status="buildable",
        recommended_task="Higher-capacity 3D CNN baseline for lesion ROI classification.",
        strengths=(
            "More capacity than ResNet18 for complex lesion appearance.",
            "A fair comparison point for MedicalNet-style ResNet transfer learning.",
        ),
        caveats=(
            "Higher overfitting risk on small datasets.",
            "GPU memory and batch size should be recorded in the benchmark.",
        ),
        config={"spatial_dims": 3, "shortcut_type": "B"},
        references=("https://docs.monai.io/en/latest/networks.html",),
    ),
    "torchvision_r3d18": ClassificationModelSpec(
        name="torchvision_r3d18",
        family="3D ResNet video backbone",
        implementation="torchvision",
        status="buildable",
        recommended_task="Generic 3D CNN baseline for stacked CT slices or lesion ROI clips.",
        strengths=(
            "Widely available and simple to reproduce.",
            "Useful non-medical-pretraining baseline.",
        ),
        caveats=(
            "Kinetics-style video pretraining is not medical-domain pretraining.",
            "Input spacing and slice sampling must be standardized.",
        ),
        config={"weights": None},
        references=("https://docs.pytorch.org/vision/stable/models/video.html",),
    ),
    "torchvision_r2plus1d18": ClassificationModelSpec(
        name="torchvision_r2plus1d18",
        family="R(2+1)D video backbone",
        implementation="torchvision",
        status="buildable",
        recommended_task="2+1D convolution baseline for multi-slice CT classification.",
        strengths=(
            "Separates spatial and through-plane modeling.",
            "Good comparison for 2.5D key-slice workflows.",
        ),
        caveats=(
            "Still not a medical-specific architecture.",
            "Must be trained on the same ROI and slice protocol as other candidates.",
        ),
        config={"weights": None},
        references=("https://docs.pytorch.org/vision/stable/models/video.html",),
    ),
    "external_medicalnet_resnet": ClassificationModelSpec(
        name="external_medicalnet_resnet",
        family="MedicalNet / Med3D 3D ResNet",
        implementation="external",
        status="external_predictions_only",
        recommended_task="Medical-pretrained 3D ResNet transfer-learning baseline.",
        strengths=(
            "Pretrained on multiple medical 3D datasets.",
            "Strong baseline when local labeled cases are limited.",
        ),
        caveats=(
            "Use official weights/code and export predictions for fair comparison.",
            "Check license and preprocessing assumptions before vendoring code.",
        ),
        config={"depths": (10, 18, 34, 50), "input": "3D CT ROI"},
        references=("https://github.com/Tencent/MedicalNet",),
    ),
    "external_radimagenet_25d": ClassificationModelSpec(
        name="external_radimagenet_25d",
        family="RadImageNet-pretrained 2.5D CNN",
        implementation="external",
        status="external_predictions_only",
        recommended_task="2D/2.5D transfer-learning baseline from medical image pretraining.",
        strengths=(
            "Medical-domain 2D pretraining can help when 3D labels are scarce.",
            "Useful comparison against the native key-slice temporal classifier.",
        ),
        caveats=(
            "Requires a clear slice aggregation rule such as mean, attention, or MIL.",
            "May miss volumetric context compared with true 3D models.",
        ),
        config={"slice_sampling": "top_k_lesion_slices", "aggregation": "attention_or_mil"},
        references=("https://www.radimagenet.com/",),
    ),
    "external_foundation_2d_mil": ClassificationModelSpec(
        name="external_foundation_2d_mil",
        family="2D foundation encoder + MIL",
        implementation="external",
        status="external_predictions_only",
        recommended_task="Modern slice-feature baseline for scalable CT classification.",
        strengths=(
            "Can exploit strong 2D encoders with limited medical labels.",
            "Efficient for long CT volumes when full 3D training is expensive.",
        ),
        caveats=(
            "Needs strict leakage control because slice-level features come from one patient.",
            "Must preserve patient-level train/validation/test splits.",
        ),
        config={"encoder": "DINOv2_or_medical_2d_encoder", "aggregator": "MIL_attention"},
        references=("https://github.com/facebookresearch/dinov2",),
    ),
    "external_radiomics_ml": ClassificationModelSpec(
        name="external_radiomics_ml",
        family="Radiomics + tabular machine learning",
        implementation="external",
        status="external_predictions_only",
        recommended_task="Classical baseline using morphology, texture, and clinical features.",
        strengths=(
            "Important clinical baseline for small datasets.",
            "More interpretable than deep models and useful for sanity checks.",
        ),
        caveats=(
            "Feature extraction must use masks from the same segmentation protocol.",
            "Needs nested cross-validation to avoid optimistic feature-selection bias.",
        ),
        config={"features": "radiomics_morphology_clinical", "models": "logistic_xgboost_svm"},
        references=("https://pyradiomics.readthedocs.io/",),
    ),
}


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def list_model_specs() -> Tuple[ClassificationModelSpec, ...]:
    """Return all candidate model specs sorted by name."""

    return tuple(_MODEL_SPECS[name] for name in sorted(_MODEL_SPECS))


def get_model_spec(name: str) -> ClassificationModelSpec:
    """Return a model spec by name, accepting hyphen/underscore variants."""

    key = _normalize_name(name)
    if key in _MODEL_SPECS:
        return _MODEL_SPECS[key]

    available = ", ".join(sorted(_MODEL_SPECS))
    raise KeyError(f"Unknown classification model '{name}'. Available models: {available}")


def build_classification_model(
    name: str,
    in_channels: int = 1,
    num_classes: int = 4,
    **overrides: Any,
) -> Any:
    """Instantiate a buildable classification candidate."""

    spec = get_model_spec(name)
    config = dict(spec.config)
    config.update(overrides)

    if spec.implementation == "native":
        return _build_native_model(spec.name, num_classes=num_classes, config=config)

    if spec.implementation == "monai":
        return _build_monai_model(spec.name, in_channels=in_channels, num_classes=num_classes, config=config)

    if spec.implementation == "torchvision":
        return _build_torchvision_model(spec.name, in_channels=in_channels, num_classes=num_classes, config=config)

    raise NotImplementedError(
        f"{spec.name} is registered as an external pipeline. Train it with its "
        "official tooling and place a predictions.csv file under "
        f"pred_root/{spec.name}/ for evaluation."
    )


def _build_native_model(name: str, num_classes: int, config: Mapping[str, Any]) -> Any:
    if name in {
        "project_stage3_temporal_evidential",
        "project_ce_binary_evidential",
        "project_ae_binary_evidential",
    }:
        return _build_temporal_classifier(config=config, default_num_classes=num_classes)

    if name == "project_stage4_activity_evidential":
        from src.models.stage4_activity import EchinococcosisActivityNet

        return EchinococcosisActivityNet(
            boundary_output_dim=int(config.get("boundary_output_dim", 128)),
            internal_output_dim=int(config.get("internal_output_dim", 128)),
            stage3_feature_dim=int(config.get("stage3_feature_dim", 256)),
            classifier_hidden_dims=list(config.get("classifier_hidden_dims", [128, 64])),
            dropout=float(config.get("dropout", 0.2)),
            use_3d=bool(config.get("use_3d", True)),
        )

    raise KeyError(f"No native builder is registered for {name!r}")


def _build_temporal_classifier(config: Mapping[str, Any], default_num_classes: int) -> Any:
    from src.models.stage3_temporal_cls import TemporalLesionClassifier

    return TemporalLesionClassifier(
        slice_encoder_variant=config.get("slice_encoder_variant", "tiny"),
        slice_encoder_pretrained=config.get("slice_encoder_pretrained"),
        include_boundary=bool(config.get("include_boundary", True)),
        num_morphology_features=int(config.get("num_morphology_features", 16)),
        morphology_hidden_dims=list(config.get("morphology_hidden_dims", [64, 128, 256])),
        morphology_output_dim=int(config.get("morphology_output_dim", 256)),
        temporal_hidden_dim=int(config.get("temporal_hidden_dim", 512)),
        temporal_num_layers=int(config.get("temporal_num_layers", 4)),
        temporal_num_heads=int(config.get("temporal_num_heads", 8)),
        temporal_dropout=float(config.get("temporal_dropout", 0.1)),
        num_classes=int(config.get("num_classes", default_num_classes)),
        classifier_hidden_dims=list(config.get("classifier_hidden_dims", [256, 128, 64])),
        classifier_dropout=float(config.get("classifier_dropout", 0.2)),
        fusion_method=config.get("fusion_method", "concat_linear"),
        aggregation_type=config.get("aggregation_type", "attention_pooling"),
    )


def _require_monai_nets():
    try:
        from monai.networks.nets import DenseNet121, resnet18, resnet50
    except ImportError as exc:
        raise ImportError(
            "MONAI is required to build this classification model. Install "
            "project dependencies with 'pip install -r requirements.txt'."
        ) from exc

    return {"DenseNet121": DenseNet121, "resnet18": resnet18, "resnet50": resnet50}


def _build_monai_model(
    name: str,
    in_channels: int,
    num_classes: int,
    config: Mapping[str, Any],
) -> Any:
    nets = _require_monai_nets()

    if name == "monai_densenet121_3d":
        return nets["DenseNet121"](
            spatial_dims=int(config.get("spatial_dims", 3)),
            in_channels=in_channels,
            out_channels=num_classes,
            pretrained=bool(config.get("pretrained", False)),
        )

    if name == "monai_resnet18_3d":
        return nets["resnet18"](
            spatial_dims=int(config.get("spatial_dims", 3)),
            n_input_channels=in_channels,
            num_classes=num_classes,
            shortcut_type=config.get("shortcut_type", "B"),
        )

    if name == "monai_resnet50_3d":
        return nets["resnet50"](
            spatial_dims=int(config.get("spatial_dims", 3)),
            n_input_channels=in_channels,
            num_classes=num_classes,
            shortcut_type=config.get("shortcut_type", "B"),
        )

    raise KeyError(f"No MONAI builder is registered for {name!r}")


def _build_torchvision_model(
    name: str,
    in_channels: int,
    num_classes: int,
    config: Mapping[str, Any],
) -> Any:
    try:
        import torch.nn as nn
        from torchvision.models.video import r2plus1d_18, r3d_18
    except ImportError as exc:
        raise ImportError(
            "torch and torchvision are required to build this classification "
            "model. Install project dependencies with 'pip install -r requirements.txt'."
        ) from exc

    if name == "torchvision_r3d18":
        model = r3d_18(weights=config.get("weights"))
    elif name == "torchvision_r2plus1d18":
        model = r2plus1d_18(weights=config.get("weights"))
    else:
        raise KeyError(f"No torchvision builder is registered for {name!r}")

    _replace_video_input_channels(model, in_channels=in_channels, nn=nn)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _replace_video_input_channels(model: Any, in_channels: int, nn: Any) -> None:
    first_conv = model.stem[0]
    if first_conv.in_channels == in_channels:
        return

    replacement = nn.Conv3d(
        in_channels=in_channels,
        out_channels=first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=first_conv.bias is not None,
    )
    model.stem[0] = replacement
