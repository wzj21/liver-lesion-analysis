# Contributing

Thank you for improving this project. Please keep the repository safe for medical imaging work.

## Ground rules

- Do not commit patient data, DICOM/NIfTI files, derived arrays, reports, or model weights.
- Use synthetic or de-identified examples only.
- Keep clinical wording cautious: AI output is decision support, not a standalone diagnosis.
- Add focused tests or validation notes for behavior changes.

## Local checks

Run lightweight checks before opening a pull request:

```powershell
python -m compileall src scripts test_software_wrapper.py
python test_software_wrapper.py
python scripts\run_clinical_inference.py --help
```

Full model tests require the deep learning dependencies and trained checkpoints.

## Development workflow

1. Create a branch:

```powershell
git checkout -b codex/your-change
```

2. Make focused changes.
3. Run checks.
4. Open a pull request and complete the safety checklist.

## Recommended priorities

- Robust data preparation and quality checks
- Complete stage-specific training scripts
- DICOM SEG / DICOM SR export
- Model and dataset versioning
- Multi-center validation
- Desktop packaging and deployment hardening
