"""Loss functions for medical time series prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class HuberLoss(nn.Module):
    """Huber loss for robust regression."""
    
    def __init__(self, delta: float = 1.0):
        """Initialize Huber loss.
        
        Args:
            delta: Threshold for switching between MSE and MAE.
        """
        super().__init__()
        self.delta = delta
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Huber loss."""
        return F.huber_loss(pred, target, delta=self.delta)


class QuantileLoss(nn.Module):
    """Quantile loss for quantile regression."""
    
    def __init__(self, alpha: float = 0.5):
        """Initialize quantile loss.
        
        Args:
            alpha: Quantile level (0 < alpha < 1).
        """
        super().__init__()
        self.alpha = alpha
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute quantile loss."""
        error = target - pred
        loss = torch.max(
            (self.alpha - 1) * error,
            self.alpha * error
        )
        return torch.mean(loss)


class PinballLoss(nn.Module):
    """Pinball loss (same as quantile loss)."""
    
    def __init__(self, alpha: float = 0.5):
        """Initialize pinball loss.
        
        Args:
            alpha: Quantile level.
        """
        super().__init__()
        self.quantile_loss = QuantileLoss(alpha)
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute pinball loss."""
        return self.quantile_loss(pred, target)


class AsymmetricLoss(nn.Module):
    """Asymmetric loss for handling different costs for over/under prediction."""
    
    def __init__(self, alpha: float = 0.5):
        """Initialize asymmetric loss.
        
        Args:
            alpha: Asymmetry parameter (0 < alpha < 1).
                  alpha < 0.5 penalizes over-prediction more.
                  alpha > 0.5 penalizes under-prediction more.
        """
        super().__init__()
        self.alpha = alpha
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute asymmetric loss."""
        error = pred - target
        loss = torch.where(
            error >= 0,
            self.alpha * error,
            (self.alpha - 1) * error
        )
        return torch.mean(loss)


class ClinicalLoss(nn.Module):
    """Clinical loss that combines multiple objectives."""
    
    def __init__(
        self, 
        mse_weight: float = 1.0,
        mae_weight: float = 0.5,
        clinical_weight: float = 0.3,
        threshold: float = 10.0
    ):
        """Initialize clinical loss.
        
        Args:
            mse_weight: Weight for MSE component.
            mae_weight: Weight for MAE component.
            clinical_weight: Weight for clinical penalty.
            threshold: Threshold for clinical penalty (BPM).
        """
        super().__init__()
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self.clinical_weight = clinical_weight
        self.threshold = threshold
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute clinical loss."""
        # MSE component
        mse_loss = F.mse_loss(pred, target)
        
        # MAE component
        mae_loss = F.l1_loss(pred, target)
        
        # Clinical penalty for large errors
        error = torch.abs(pred - target)
        clinical_penalty = torch.mean(torch.relu(error - self.threshold))
        
        total_loss = (
            self.mse_weight * mse_loss +
            self.mae_weight * mae_loss +
            self.clinical_weight * clinical_penalty
        )
        
        return total_loss


def create_loss_function(config) -> nn.Module:
    """Create loss function based on configuration.
    
    Args:
        config: Configuration object containing loss parameters.
        
    Returns:
        Loss function.
    """
    loss_type = config.training.loss.type.lower()
    
    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "mae":
        return nn.L1Loss()
    elif loss_type == "huber":
        delta = config.training.loss.huber_delta
        return HuberLoss(delta=delta)
    elif loss_type == "quantile":
        alpha = config.training.loss.quantile_alpha
        return QuantileLoss(alpha=alpha)
    elif loss_type == "asymmetric":
        alpha = config.training.loss.quantile_alpha
        return AsymmetricLoss(alpha=alpha)
    elif loss_type == "clinical":
        return ClinicalLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
