"""
Dependency-light tests for the classification model catalog.

These tests intentionally do not instantiate torch, torchvision, or MONAI.
"""

from src.classification.model_zoo import get_model_spec, list_model_specs


def test_catalog_contains_required_candidates():
    names = {spec.name for spec in list_model_specs()}

    required = {
        "project_stage3_temporal_evidential",
        "project_stage4_activity_evidential",
        "monai_densenet121_3d",
        "monai_resnet18_3d",
        "torchvision_r3d18",
        "external_medicalnet_resnet",
        "external_radiomics_ml",
    }

    assert required.issubset(names)


def test_stage3_native_spec_is_four_class():
    spec = get_model_spec("project-stage3-temporal-evidential")

    assert spec.implementation == "native"
    assert spec.config["num_classes"] == 4


def test_external_medicalnet_is_prediction_only():
    spec = get_model_spec("external_medicalnet_resnet")

    assert spec.implementation == "external"
    assert spec.status == "external_predictions_only"


def _run_without_pytest():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: PASS")


if __name__ == "__main__":
    _run_without_pytest()
