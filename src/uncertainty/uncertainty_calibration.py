"""
Uncertainty Calibration Module
不确定性校准模块

Post-hoc calibration methods for neural network predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class TemperatureScaling(nn.Module):
    """Temperature Scaling for model calibration.
    
    Scales logits by a learned temperature parameter.
    """
    
    def __init__(self, init_temperature: float = 1.0):
        """Initialize temperature scaling.
        
        Args:
            init_temperature: Initial temperature value
        """
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temperature)
        
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling.
        
        Args:
            logits: Input logits
            
        Returns:
            Scaled logits
        """
        return logits / self.temperature
        
    def calibrate(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> float:
        """Learn optimal temperature from validation data.
        
        Args:
            logits: Validation logits
            labels: Validation labels
            lr: Learning rate
            max_iter: Maximum iterations
            
        Returns:
            Final temperature value
        """
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def closure():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            loss = F.cross_entropy(scaled_logits, labels)
            loss.backward()
            return loss
            
        optimizer.step(closure)
        
        return self.temperature.item()


class PlattScaling(nn.Module):
    """Platt Scaling for binary classification calibration.
    
    Fits a logistic regression on the logits.
    """
    
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.zeros(1))
        
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply Platt scaling.
        
        Args:
            logits: Input logits (binary classification)
            
        Returns:
            Calibrated probabilities
        """
        return torch.sigmoid(self.a * logits + self.b)
        
    def calibrate(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 100,
    ):
        """Learn optimal parameters from validation data."""
        optimizer = torch.optim.LBFGS([self.a, self.b], lr=lr, max_iter=max_iter)
        
        def closure():
            optimizer.zero_grad()
            probs = self.forward(logits)
            loss = F.binary_cross_entropy(probs, labels.float())
            loss.backward()
            return loss
            
        optimizer.step(closure)


class IsotonicRegression:
    """Isotonic Regression for non-parametric calibration.
    
    Uses scikit-learn's isotonic regression.
    """
    
    def __init__(self):
        from sklearn.isotonic import IsotonicRegression as SKIsotonic
        self.isotonic = SKIsotonic(out_of_bounds='clip')
        self.fitted = False
        
    def fit(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
    ):
        """Fit isotonic regression.
        
        Args:
            probs: Predicted probabilities
            labels: True labels
        """
        self.isotonic.fit(probs, labels)
        self.fitted = True
        
    def transform(self, probs: np.ndarray) -> np.ndarray:
        """Transform probabilities.
        
        Args:
            probs: Predicted probabilities
            
        Returns:
            Calibrated probabilities
        """
        if not self.fitted:
            raise RuntimeError("Fit the calibrator first")
        return self.isotonic.predict(probs)


def calibrate_model(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    method: str = 'temperature',
    device: str = 'cuda',
) -> nn.Module:
    """Calibrate a model using validation data.
    
    Args:
        model: Model to calibrate
        val_loader: Validation data loader
        method: Calibration method ('temperature', 'platt')
        device: Device to use
        
    Returns:
        Calibrated model wrapper
    """
    model.eval()
    logits_list = []
    labels_list = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            logits_list.append(logits.cpu())
            labels_list.append(labels)
            
    all_logits = torch.cat(logits_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)
    
    if method == 'temperature':
        calibrator = TemperatureScaling()
        calibrator.calibrate(all_logits, all_labels)
    elif method == 'platt':
        calibrator = PlattScaling()
        calibrator.calibrate(all_logits, all_labels)
    else:
        raise ValueError(f"Unknown calibration method: {method}")
        
    return CalibratedModel(model, calibrator)


class CalibratedModel(nn.Module):
    """Wrapper for a calibrated model."""
    
    def __init__(self, model: nn.Module, calibrator: nn.Module):
        super().__init__()
        self.model = model
        self.calibrator = calibrator
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return self.calibrator(logits)


if __name__ == "__main__":
    # Test temperature scaling
    temp_scaling = TemperatureScaling()
    logits = torch.randn(100, 10)
    labels = torch.randint(0, 10, (100,))
    
    print(f"Initial temperature: {temp_scaling.temperature.item():.3f}")
    temp_scaling.calibrate(logits, labels)
    print(f"Calibrated temperature: {temp_scaling.temperature.item():.3f}")
