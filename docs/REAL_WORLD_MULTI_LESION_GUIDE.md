# Real-world Multi-lesion Guide

The real-world workflow changes the unit of analysis from one patient with one
label to one patient with many lesion-level predictions and a patient-level
multi-label summary.

## Workflow

```text
DICOM / NIfTI input
  -> liver segmentation
  -> all suspicious lesion detection and instance segmentation
  -> lesion-level classification
  -> CE/AE-specific task heads
  -> patient-level aggregation
  -> safety gate
  -> structured report
```

## Patient-level Aggregation

The aggregation layer reads lesion predictions and derives flags:

```json
{
  "has_ce": true,
  "has_ae": false,
  "has_benign": false,
  "has_malignant": true,
  "has_complex_coexistence": true
}
```

This design supports CE plus malignancy, AE plus malignancy, CE plus AE, and
multiple lesions with different CE stages.

## Safety Gate

The safety gate triggers clinician review for:

```text
low confidence
high uncertainty
UNCERTAIN lesion class
CE and AE coexistence
echinococcosis and malignancy coexistence
suspected malignant lesion
model disagreement
segmentation failure
```

The safety gate should not be bypassed by report-generation logic or an LLM.
