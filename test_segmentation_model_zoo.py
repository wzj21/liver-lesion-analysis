"""
Dependency-light tests for the segmentation model catalog.

These tests intentionally do not instantiate torch or MONAI models.
"""

from src.segmentation.model_zoo import get_model_spec, list_model_specs


def test_catalog_contains_required_candidates():
    names = {spec.name for spec in list_model_specs()}

    required = {
        "external_nnunet",
        "external_mednext",
        "external_nnformer",
        "monai_segresnet",
        "monai_swinunetr",
        "monai_unetr",
        "project_cascade_liver",
    }

    assert required.issubset(names)


def test_external_model_is_not_native_builder():
    spec = get_model_spec("external-nnunet")

    assert spec.implementation == "external"
    assert spec.status == "external_predictions_only"


def test_monai_segresnet_documents_default_config():
    spec = get_model_spec("monai_segresnet")

    assert spec.implementation == "monai"
    assert spec.config["init_filters"] == 32


def _run_without_pytest():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: PASS")


if __name__ == "__main__":
    _run_without_pytest()
