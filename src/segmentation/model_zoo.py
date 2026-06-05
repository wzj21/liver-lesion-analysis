"""
Model catalog for 3D liver and liver-lesion segmentation experiments.

The catalog deliberately separates two concepts:

1. Models this project can instantiate directly, such as MONAI networks.
2. External pipelines, such as nnU-Net, that should be trained with their own
   official tooling and then compared through exported prediction masks.

Keeping the registry dependency-light lets CI list the candidate algorithms
without importing torch or MONAI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Tuple


ConfigDict = Mapping[str, Any]


@dataclass(frozen=True)
class SegmentationModelSpec:
    """Metadata for one segmentation candidate."""

    name: str
    family: str
    implementation: str
    status: str
    recommended_task: str
    strengths: Tuple[str, ...]
    caveats: Tuple[str, ...]
    config: ConfigDict
    references: Tuple[str, ...] = ()


_MODEL_SPECS: Dict[str, SegmentationModelSpec] = {
    "project_cascade_liver": SegmentationModelSpec(
        name="project_cascade_liver",
        family="ConvNeXt3D cascade",
        implementation="native",
        status="buildable",
        recommended_task="Stage1 liver segmentation baseline already present in this repo.",
        strengths=(
            "Coarse-to-fine ROI workflow fits large abdominal CT volumes.",
            "Uses the existing ConvNeXt3D backbone and uncertainty-aware fine stage.",
        ),
        caveats=(
            "Needs project-specific training closure and validation.",
            "Not automatically tuned across datasets like nnU-Net.",
        ),
        config={
            "coarse_backbone": "tiny",
            "fine_backbone": "small",
            "coarse_input_size": (128, 128, 128),
            "roi_margin": 10,
        },
        references=("src.models.stage1_liver_seg.CascadeLiverSegmentation",),
    ),
    "monai_unet": SegmentationModelSpec(
        name="monai_unet",
        family="3D U-Net",
        implementation="monai",
        status="buildable",
        recommended_task="Simple 3D CNN baseline for liver or lesion masks.",
        strengths=(
            "Fast to train and easy to debug.",
            "Useful sanity-check baseline before larger architectures.",
        ),
        caveats=(
            "Usually weaker than tuned nnU-Net or modern residual/transformer models.",
            "Patch size and spacing choices strongly affect performance.",
        ),
        config={
            "channels": (16, 32, 64, 128, 256),
            "strides": (2, 2, 2, 2),
            "num_res_units": 2,
        },
        references=("https://docs.monai.io/en/stable/networks.html",),
    ),
    "monai_segresnet": SegmentationModelSpec(
        name="monai_segresnet",
        family="SegResNet",
        implementation="monai",
        status="buildable",
        recommended_task="Strong residual 3D CNN baseline for liver and lesion segmentation.",
        strengths=(
            "Good accuracy-speed trade-off on many volumetric medical tasks.",
            "Often easier to train than transformer-heavy models on limited data.",
        ),
        caveats=(
            "Still requires fair tuning of patch size, loss, augmentation, and spacing.",
            "Small lesions may need foreground-biased sampling.",
        ),
        config={
            "init_filters": 32,
            "blocks_down": (1, 2, 2, 4),
            "blocks_up": (1, 1, 1),
            "dropout_prob": 0.0,
        },
        references=("https://docs.monai.io/en/stable/networks.html",),
    ),
    "monai_dynunet": SegmentationModelSpec(
        name="monai_dynunet",
        family="DynUNet",
        implementation="monai",
        status="buildable",
        recommended_task="nnU-Net-inspired configurable 3D U-Net baseline inside MONAI.",
        strengths=(
            "Handles anisotropic kernels and strides better than a fixed U-Net.",
            "Good bridge between simple U-Net and full nnU-Net pipeline.",
        ),
        caveats=(
            "The provided defaults are only starter values.",
            "Best results need dataset-specific kernel/stride planning.",
        ),
        config={
            "kernel_size": (3, 3, 3, 3, 3),
            "strides": (1, 2, 2, 2, 2),
            "upsample_kernel_size": (2, 2, 2, 2),
            "deep_supervision": False,
        },
        references=("https://docs.monai.io/en/stable/networks.html",),
    ),
    "monai_unetr": SegmentationModelSpec(
        name="monai_unetr",
        family="UNETR",
        implementation="monai",
        status="buildable",
        recommended_task="Transformer-based 3D segmentation candidate for larger datasets.",
        strengths=(
            "Models long-range 3D context with a ViT encoder.",
            "Worth testing when lesions require global anatomical context.",
        ),
        caveats=(
            "Typically needs more data and memory than CNN baselines.",
            "Input roi_size must be fixed and compatible with the transformer patching.",
        ),
        config={
            "feature_size": 16,
            "hidden_size": 768,
            "mlp_dim": 3072,
            "num_heads": 12,
            "pos_embed": "perceptron",
            "norm_name": "instance",
            "res_block": True,
            "dropout_rate": 0.0,
        },
        references=("https://docs.monai.io/en/stable/networks.html",),
    ),
    "monai_swinunetr": SegmentationModelSpec(
        name="monai_swinunetr",
        family="SwinUNETR",
        implementation="monai",
        status="buildable",
        recommended_task="Windowed-transformer 3D candidate for high-quality volumetric segmentation.",
        strengths=(
            "Combines local windows and hierarchical global context.",
            "A strong modern candidate when GPU memory is sufficient.",
        ),
        caveats=(
            "More memory-sensitive than CNN baselines.",
            "MONAI constructor signatures differ slightly across versions.",
        ),
        config={
            "feature_size": 48,
            "use_checkpoint": True,
        },
        references=("https://docs.monai.io/en/stable/networks.html",),
    ),
    "external_nnunet": SegmentationModelSpec(
        name="external_nnunet",
        family="nnU-Net",
        implementation="external",
        status="external_predictions_only",
        recommended_task="Primary self-configuring baseline for liver and lesion segmentation.",
        strengths=(
            "Strong public baseline because preprocessing, architecture, and training are self-configured.",
            "Excellent first model to beat before trusting a custom architecture.",
        ),
        caveats=(
            "Train with the official nnU-Net tooling, then export masks into the benchmark folder.",
            "Do not mix its data split or post-processing with other candidates.",
        ),
        config={
            "trainer": "nnUNetTrainer",
            "folds": "0-4 or the same fixed split used by the project",
            "plans": "official nnU-Net dataset fingerprint/plans",
        },
        references=("https://github.com/MIC-DKFZ/nnUNet",),
    ),
    "external_mednext": SegmentationModelSpec(
        name="external_mednext",
        family="MedNeXt",
        implementation="external",
        status="external_predictions_only",
        recommended_task="Recent ConvNeXt-style medical segmentation candidate.",
        strengths=(
            "Designed around ConvNeXt blocks for medical segmentation.",
            "Useful comparison against this project's ConvNeXt3D cascade.",
        ),
        caveats=(
            "Keep third-party code and licenses separate unless formally vendored.",
            "Needs the same validation split and metrics as every other candidate.",
        ),
        config={
            "variant": "small_or_base_after_memory_test",
            "input_spacing": "same as benchmark preprocessing",
        },
        references=("https://github.com/MIC-DKFZ/MedNeXt",),
    ),
    "external_nnformer": SegmentationModelSpec(
        name="external_nnformer",
        family="nnFormer",
        implementation="external",
        status="external_predictions_only",
        recommended_task="Interleaved transformer candidate for 3D medical segmentation.",
        strengths=(
            "Transformer design targeted at volumetric segmentation.",
            "Useful to test whether transformer context helps lesion boundaries.",
        ),
        caveats=(
            "Can be heavier and harder to reproduce than MONAI baselines.",
            "Treat as a research candidate until validated on local data.",
        ),
        config={
            "input_spacing": "same as benchmark preprocessing",
            "folds": "same validation protocol as other candidates",
        },
        references=(
            "https://github.com/282857341/nnFormer",
            "https://arxiv.org/abs/2109.03201",
        ),
    ),
    "external_segformer3d": SegmentationModelSpec(
        name="external_segformer3d",
        family="SegFormer3D",
        implementation="external",
        status="external_predictions_only",
        recommended_task="Recent efficient transformer-style 3D segmentation candidate.",
        strengths=(
            "Adds a modern efficient-transformer comparison point.",
            "May be useful where CNNs miss long-range lesion context.",
        ),
        caveats=(
            "External implementation quality varies by repository.",
            "Benchmark only after matching preprocessing and label conventions.",
        ),
        config={
            "input_spacing": "same as benchmark preprocessing",
            "patch_size": "choose by GPU memory",
        },
        references=(
            "https://github.com/OSUPCVLab/SegFormer3D",
            "https://arxiv.org/abs/2404.10156",
        ),
    ),
}


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def list_model_specs() -> Tuple[SegmentationModelSpec, ...]:
    """Return all candidate model specs sorted by name."""

    return tuple(_MODEL_SPECS[name] for name in sorted(_MODEL_SPECS))


def get_model_spec(name: str) -> SegmentationModelSpec:
    """Return a model spec by name, accepting hyphen/underscore variants."""

    key = _normalize_name(name)
    if key in _MODEL_SPECS:
        return _MODEL_SPECS[key]

    available = ", ".join(sorted(_MODEL_SPECS))
    raise KeyError(f"Unknown segmentation model '{name}'. Available models: {available}")


def build_segmentation_model(
    name: str,
    in_channels: int = 1,
    out_channels: int = 2,
    roi_size: Iterable[int] = (96, 96, 96),
    **overrides: Any,
) -> Any:
    """Instantiate a buildable candidate model.

    External pipelines are intentionally not imported here. Train them with
    their official tooling and compare their exported prediction masks with the
    benchmark script.
    """

    spec = get_model_spec(name)
    config = dict(spec.config)
    config.update(overrides)
    roi_size_tuple = tuple(int(v) for v in roi_size)

    if spec.implementation == "native":
        return _build_native_model(spec.name, in_channels, out_channels, config)

    if spec.implementation == "monai":
        return _build_monai_model(spec.name, in_channels, out_channels, roi_size_tuple, config)

    raise NotImplementedError(
        f"{spec.name} is registered as an external pipeline. Train it with its "
        "official repository and place predictions under "
        f"pred_root/{spec.name}/ for evaluation."
    )


def _build_native_model(
    name: str,
    in_channels: int,
    out_channels: int,
    config: Mapping[str, Any],
) -> Any:
    if name == "project_cascade_liver":
        from src.models.stage1_liver_seg import CascadeLiverSegmentation

        return CascadeLiverSegmentation(
            in_channels=in_channels,
            num_classes=out_channels,
            coarse_backbone=config.get("coarse_backbone", "tiny"),
            fine_backbone=config.get("fine_backbone", "small"),
            coarse_input_size=tuple(config.get("coarse_input_size", (128, 128, 128))),
            roi_margin=int(config.get("roi_margin", 10)),
        )

    raise KeyError(f"No native builder is registered for {name!r}")


def _require_monai_nets():
    try:
        from monai.networks.nets import DynUNet, SegResNet, SwinUNETR, UNETR, UNet
    except ImportError as exc:
        raise ImportError(
            "MONAI is required to build this segmentation model. Install project "
            "dependencies with 'pip install -r requirements.txt'."
        ) from exc

    return {
        "DynUNet": DynUNet,
        "SegResNet": SegResNet,
        "SwinUNETR": SwinUNETR,
        "UNETR": UNETR,
        "UNet": UNet,
    }


def _build_monai_model(
    name: str,
    in_channels: int,
    out_channels: int,
    roi_size: Tuple[int, int, int],
    config: Mapping[str, Any],
) -> Any:
    nets = _require_monai_nets()

    if name == "monai_unet":
        return nets["UNet"](
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=tuple(config.get("channels", (16, 32, 64, 128, 256))),
            strides=tuple(config.get("strides", (2, 2, 2, 2))),
            num_res_units=int(config.get("num_res_units", 2)),
        )

    if name == "monai_segresnet":
        return nets["SegResNet"](
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            init_filters=int(config.get("init_filters", 32)),
            blocks_down=tuple(config.get("blocks_down", (1, 2, 2, 4))),
            blocks_up=tuple(config.get("blocks_up", (1, 1, 1))),
            dropout_prob=float(config.get("dropout_prob", 0.0)),
        )

    if name == "monai_dynunet":
        return nets["DynUNet"](
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=tuple(config.get("kernel_size", (3, 3, 3, 3, 3))),
            strides=tuple(config.get("strides", (1, 2, 2, 2, 2))),
            upsample_kernel_size=tuple(config.get("upsample_kernel_size", (2, 2, 2, 2))),
            deep_supervision=bool(config.get("deep_supervision", False)),
        )

    if name == "monai_unetr":
        kwargs = {
            "spatial_dims": 3,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "img_size": roi_size,
            "feature_size": int(config.get("feature_size", 16)),
            "hidden_size": int(config.get("hidden_size", 768)),
            "mlp_dim": int(config.get("mlp_dim", 3072)),
            "num_heads": int(config.get("num_heads", 12)),
            "norm_name": config.get("norm_name", "instance"),
            "res_block": bool(config.get("res_block", True)),
            "dropout_rate": float(config.get("dropout_rate", 0.0)),
        }
        try:
            return nets["UNETR"](**kwargs, pos_embed=config.get("pos_embed", "perceptron"))
        except TypeError:
            return nets["UNETR"](**kwargs, proj_type=config.get("proj_type", "conv"))

    if name == "monai_swinunetr":
        kwargs = {
            "spatial_dims": 3,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "feature_size": int(config.get("feature_size", 48)),
            "use_checkpoint": bool(config.get("use_checkpoint", True)),
        }
        try:
            return nets["SwinUNETR"](img_size=roi_size, **kwargs)
        except TypeError:
            return nets["SwinUNETR"](**kwargs)

    raise KeyError(f"No MONAI builder is registered for {name!r}")
