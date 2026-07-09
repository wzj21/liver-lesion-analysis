# Real-world Data Schema

This project now uses a lesion-level schema for real-world liver echinococcosis
and complex liver lesion studies. A patient can have multiple lesions and a
patient can have multiple concurrent labels.

## Tables

`patients.csv` stores one row per patient.

`studies.csv` stores one row per imaging study. The same patient can have
multiple studies over time.

`lesions.csv` stores one row per lesion. This is the core table for detection,
segmentation, classification, CE staging, AE risk, and reporting.

`patient_multilabels.csv` stores derived patient-level labels such as `has_ce`,
`has_ae`, `has_malignant`, and `has_complex_coexistence`.

`petct_activity.csv` stores PET/CT-derived AE activity labels. Cases without
PET/CT should not be treated as ground-truth AE activity labels.

`synthetic_manifest.csv` tracks generated samples. Synthetic samples are allowed
only in the training split.

## Main Lesion Classes

The lesion-level `main_class` field must be one of:

```text
CE
AE
BENIGN
MALIGNANT
UNCERTAIN
```

Complex coexistence is represented at the patient level, not by creating new
classes. For example, CE plus HCC should be represented as one CE lesion and one
MALIGNANT lesion, then aggregated to `has_ce=true` and `has_malignant=true`.

## Evidence Levels

Use `evidence_level` to record label strength:

```text
pathology
surgery
petct
serology_followup
expert_consensus
imaging_only
unknown
```

Higher-strength labels should be preferred for validation and test cohorts.
