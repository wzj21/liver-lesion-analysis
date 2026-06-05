# Roadmap

This roadmap turns the current research prototype into a more reliable medical imaging software project.

## Phase 1: Repository and software hygiene

- [x] GitHub-safe `.gitignore`
- [x] Lightweight CI
- [x] Desktop GUI launcher
- [x] Windows EXE build script
- [x] Clinical inference wrapper
- [x] Rule-based clinical review recommendations
- [x] Optional LLM advisor interface

## Phase 2: Training closure

- [ ] Complete Stage1 liver segmentation training script with DataLoader integration
- [ ] Complete Stage2 lesion detection/segmentation training script
- [ ] Complete Stage3 classification training script from lesion masks/key slices
- [ ] Complete Stage4 activity classifier training script
- [ ] Add deterministic train/val/test split manifests
- [ ] Log Dice, HD95, mAP, AUC, calibration error, and confusion matrices

## Phase 3: Medical imaging interoperability

- [ ] Export masks back to original image space
- [ ] Export DICOM SEG
- [ ] Export DICOM SR or FHIR-compatible structured report
- [ ] Add DICOM metadata quality checks
- [ ] Add multi-phase CT input support

## Phase 4: Model quality and robustness

- [x] Add segmentation benchmark catalog for nnU-Net, MONAI, and recent 3D candidates
- [x] Add shared mask-level segmentation evaluator and ranking script
- [ ] Train and compare nnU-Net-style baseline for liver/lesion segmentation
- [ ] Add MONAI transforms and cache dataset support
- [ ] Add test-time augmentation for uncertainty-sensitive cases
- [ ] Add active learning queue for low-confidence or high-uncertainty cases
- [ ] Add model cards and dataset cards

## Phase 5: Deployment and release

- [ ] GitHub Releases for source packages and desktop builds
- [ ] Optional Git LFS or release assets for model weights
- [ ] Docker/FastAPI deployment path
- [ ] CPU/GPU runtime profiles
- [ ] Signed Windows installer

## Phase 6: Clinical validation

- [ ] Single-center retrospective validation
- [ ] Multi-center external validation
- [ ] Reader study against radiologists
- [ ] Failure mode analysis
- [ ] Clinical safety and regulatory review
