# Generation Augmentation Guide

Generative augmentation is a separate research module for CE/AE CT and MRI data.
It is not a substitute for real labels and must not leak into validation or test
cohorts.

## Candidate Methods

Baseline methods:

```text
traditional augmentation
lesion copy-paste
MixUp / CutMix
cGAN
CycleGAN
StyleGAN
VQ-GAN
DDPM
latent diffusion
mask-conditioned diffusion
MAISI / MONAI generative pipeline
```

Proposed main method:

```text
Echi-MambaDiff = latent diffusion + Mamba denoiser + mask-conditioned control branch
```

## Allowed Conditions

Generation may be conditioned on curated training-set fields:

```text
modality
disease_type
liver_mask
lesion_mask
lesion_size
lesion_location
CE stage
AE morphology pattern
calcification
necrosis
subcyst pattern
infiltrative margin
```

## Prohibited Uses

Synthetic data must not be assigned to:

```text
validation
internal_test
external_test
prospective_test
```

Synthetic generation must not invent ground-truth labels for:

```text
PET/CT metabolic activity
AE P stage
postoperative complications
survival or recurrence outcomes
```

These labels require real clinical evidence.

## Quality Control

Generated samples should pass:

```text
intensity and geometry checks
mask-image consistency checks
lesion-in-liver checks
radiomics distribution checks
nearest-neighbor privacy checks
expert blind review
downstream task validation
```
