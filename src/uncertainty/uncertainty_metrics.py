"""
Uncertainty Metrics Module
不确定性度量模块

Metrics for evaluating uncertainty estimation quality.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error (ECE).
    
    ECE measures the difference between predicted confidence and accuracy.
    
    Args:
        probs: Predicted probabilities (N,) or (N, C)
        labels: True labels (N,)
        num_bins: Number of bins for calibration
        
    Returns:
        ECE value (lower is better)
    """
    if probs.ndim == 2:
        # Multi-class: use max probability
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
    else:
        confidences = probs
        predictions = (probs > 0.5).astype(int)
        
    accuracies = (predictions == labels).astype(float)
    
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    
    for i in range(num_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += prop_in_bin * np.abs(avg_accuracy - avg_confidence)
            
    return ece


def maximum_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 15,
) -> float:
    """Compute Maximum Calibration Error (MCE).
    
    Args:
        probs: Predicted probabilities
        labels: True labels
        num_bins: Number of bins
        
    Returns:
        MCE value
    """
    if probs.ndim == 2:
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
    else:
        confidences = probs
        predictions = (probs > 0.5).astype(int)
        
    accuracies = (predictions == labels).astype(float)
    
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    mce = 0.0
    
    for i in range(num_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        
        if in_bin.sum() > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            mce = max(mce, np.abs(avg_accuracy - avg_confidence))
            
    return mce


def brier_score(
    probs: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute Brier Score.
    
    Brier score is the mean squared error of probabilistic predictions.
    
    Args:
        probs: Predicted probabilities
        labels: True labels (one-hot or indices)
        
    Returns:
        Brier score (lower is better)
    """
    if labels.ndim == 1:
        # Convert to one-hot
        num_classes = probs.shape[1] if probs.ndim == 2 else 2
        labels_onehot = np.eye(num_classes)[labels]
    else:
        labels_onehot = labels
        
    if probs.ndim == 1:
        probs = np.stack([1 - probs, probs], axis=1)
        
    return ((probs - labels_onehot) ** 2).mean()


def negative_log_likelihood(
    probs: np.ndarray,
    labels: np.ndarray,
    eps: float = 1e-10,
) -> float:
    """Compute Negative Log-Likelihood.
    
    Args:
        probs: Predicted probabilities
        labels: True labels
        eps: Small value for numerical stability
        
    Returns:
        NLL (lower is better)
    """
    if probs.ndim == 2:
        probs_true = probs[np.arange(len(labels)), labels]
    else:
        probs_true = probs * labels + (1 - probs) * (1 - labels)
        
    return -np.log(probs_true + eps).mean()


def uncertainty_auroc(
    uncertainties: np.ndarray,
    errors: np.ndarray,
) -> float:
    """Compute AUROC for uncertainty-error correlation.
    
    Measures how well uncertainty predicts errors.
    Higher is better (uncertainty should be high when model is wrong).
    
    Args:
        uncertainties: Predicted uncertainties
        errors: Binary error indicators (1 if wrong, 0 if correct)
        
    Returns:
        AUROC value
    """
    from sklearn.metrics import roc_auc_score
    
    if errors.sum() == 0 or errors.sum() == len(errors):
        return 0.5  # Cannot compute AUROC
        
    return roc_auc_score(errors, uncertainties)


def compute_uncertainty_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    uncertainties: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute all uncertainty metrics.
    
    Args:
        probs: Predicted probabilities
        labels: True labels
        uncertainties: Optional explicit uncertainty values
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Calibration metrics
    metrics['ece'] = expected_calibration_error(probs, labels)
    metrics['mce'] = maximum_calibration_error(probs, labels)
    
    # Scoring rules
    metrics['brier'] = brier_score(probs, labels)
    metrics['nll'] = negative_log_likelihood(probs, labels)
    
    # Uncertainty-error correlation
    if uncertainties is None:
        # Use 1 - max_prob as uncertainty
        if probs.ndim == 2:
            uncertainties = 1 - probs.max(axis=1)
        else:
            uncertainties = 1 - np.maximum(probs, 1 - probs)
            
    predictions = probs.argmax(axis=1) if probs.ndim == 2 else (probs > 0.5).astype(int)
    errors = (predictions != labels).astype(float)
    
    metrics['uncertainty_auroc'] = uncertainty_auroc(uncertainties, errors)
    
    # Uncertainty statistics
    metrics['uncertainty_mean'] = uncertainties.mean()
    metrics['uncertainty_std'] = uncertainties.std()
    
    return metrics


def reliability_diagram(
    probs: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute reliability diagram data.
    
    Args:
        probs: Predicted probabilities
        labels: True labels
        num_bins: Number of bins
        
    Returns:
        Tuple of (bin_centers, bin_accuracies, bin_counts)
    """
    if probs.ndim == 2:
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
    else:
        confidences = np.maximum(probs, 1 - probs)
        predictions = (probs > 0.5).astype(int)
        
    accuracies = (predictions == labels).astype(float)
    
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    bin_accuracies = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)
    
    for i in range(num_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        bin_counts[i] = in_bin.sum()
        
        if bin_counts[i] > 0:
            bin_accuracies[i] = accuracies[in_bin].mean()
            
    return bin_centers, bin_accuracies, bin_counts


if __name__ == "__main__":
    # Test metrics
    np.random.seed(42)
    
    # Simulated predictions
    probs = np.random.rand(1000, 4)
    probs = probs / probs.sum(axis=1, keepdims=True)  # Normalize
    labels = np.random.randint(0, 4, 1000)
    
    metrics = compute_uncertainty_metrics(probs, labels)
    
    print("Uncertainty Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
        
    # Reliability diagram
    centers, accs, counts = reliability_diagram(probs, labels)
    print("\nReliability Diagram:")
    for c, a, n in zip(centers, accs, counts):
        print(f"  Conf {c:.2f}: Acc {a:.3f}, Count {int(n)}")
