# Security Policy

## Supported scope

This repository is an AI-assisted medical imaging research and engineering project. Security reports are welcome for:

- Patient data exposure risks
- Unsafe file handling
- Dependency or packaging vulnerabilities
- Inference/reporting behavior that could bypass clinician review
- Secret/API key leakage risks

## Do not upload sensitive data

Never create issues, pull requests, commits, or discussions containing:

- DICOM files or screenshots with patient identifiers
- NIfTI volumes derived from patient scans
- Names, IDs, dates of birth, accession numbers, study IDs, or hospital numbers
- Full clinical reports
- API keys, tokens, internal service URLs, or credentials

Use synthetic data or fully de-identified minimal examples.

## Reporting a vulnerability

If this repository is public and you need to report a sensitive vulnerability, open a minimal issue that says a private security report is needed, without disclosing details. If GitHub private vulnerability reporting is enabled later, use that route.

For non-sensitive bugs, use the bug report template.

## Medical safety

This project is not a certified medical device. Outputs must be reviewed by qualified clinicians. Any deployment intended for clinical use requires local validation, regulatory review, privacy review, and operational monitoring.
