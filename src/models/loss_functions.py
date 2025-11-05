"""
Custom loss functions for irrigation prediction with asymmetric penalties.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_sample_weight


class AsymmetricLoss:
    """Asymmetric loss functions for irrigation prediction."""
    
    def __init__(self, alpha: float = 2.0, beta: float = 1.0, gamma: float = 5.0):
        """
        Initialize asymmetric loss.
        
        Args:
            alpha: Penalty weight for under-irrigation (y_pred < y_true)
            beta: Penalty weight for over-irrigation (y_pred > y_true)
            gamma: Additional penalty for field capacity violations
        """
        self.alpha = alpha  # Under-irrigation penalty (higher)
        self.beta = beta    # Over-irrigation penalty (lower)
        self.gamma = gamma  # FC violation penalty
    
    def asymmetric_mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate asymmetric Mean Absolute Error.
        
        Args:
            y_true: True irrigation values
            y_pred: Predicted irrigation values
            
        Returns:
            Asymmetric MAE loss
        """
        error = y_pred - y_true
        
        # Under-irrigation penalty (y_pred < y_true, error < 0)
        under_penalty = self.alpha * np.abs(error[error < 0])
        
        # Over-irrigation penalty (y_pred > y_true, error > 0)
        over_penalty = self.beta * np.abs(error[error > 0])
        
        total_loss = np.sum(under_penalty) + np.sum(over_penalty)
        return total_loss / len(y_true)
    
    def asymmetric_mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate asymmetric Mean Squared Error.
        
        Args:
            y_true: True irrigation values
            y_pred: Predicted irrigation values
            
        Returns:
            Asymmetric MSE loss
        """
        error = y_pred - y_true
        
        # Under-irrigation penalty
        under_penalty = self.alpha * (error[error < 0] ** 2)
        
        # Over-irrigation penalty
        over_penalty = self.beta * (error[error > 0] ** 2)
        
        total_loss = np.sum(under_penalty) + np.sum(over_penalty)
        return total_loss / len(y_true)
    
    def field_capacity_violation_penalty(self, y_pred: np.ndarray, 
                                       current_theta: np.ndarray,
                                       field_capacity: np.ndarray,
                                       root_depth: np.ndarray,
                                       et_forecast: np.ndarray,
                                       rain_forecast: np.ndarray) -> np.ndarray:
        """
        Calculate penalty for field capacity violations.
        
        Args:
            y_pred: Predicted irrigation (mm)
            current_theta: Current soil moisture (VWC)
            field_capacity: Field capacity (VWC)
            root_depth: Root depth (m)
            et_forecast: Forecast ET (mm)
            rain_forecast: Forecast rainfall (mm)
            
        Returns:
            Array of FC violation penalties
        """
        # Convert to mm
        current_mm = current_theta * root_depth * 1000
        fc_mm = field_capacity * root_depth * 1000
        
        # Projected soil moisture after irrigation, rain, and ET
        projected_mm = current_mm + y_pred + rain_forecast - et_forecast
        
        # Violation amount (excess above FC)
        violations = np.maximum(0, projected_mm - fc_mm)
        
        # Penalty proportional to violation severity
        penalties = self.gamma * violations
        
        return penalties
    
    def total_loss_with_fc_penalty(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  current_theta: np.ndarray,
                                  field_capacity: np.ndarray,
                                  root_depth: np.ndarray,
                                  et_forecast: np.ndarray,
                                  rain_forecast: np.ndarray) -> float:
        """
        Calculate total loss including FC violation penalty.
        
        Returns:
            Total loss value
        """
        # Base asymmetric loss
        base_loss = self.asymmetric_mae(y_true, y_pred)
        
        # FC violation penalty
        fc_penalties = self.field_capacity_violation_penalty(
            y_pred, current_theta, field_capacity, root_depth,
            et_forecast, rain_forecast
        )
        fc_loss = np.mean(fc_penalties)
        
        return base_loss + fc_loss


def create_sample_weights(y_true: np.ndarray, 
                         et_demand: np.ndarray,
                         stress_risk: np.ndarray,
                         alpha: float = 2.0,
                         beta: float = 1.0) -> np.ndarray:
    """
    Create sample weights for XGBoost to approximate asymmetric loss.
    
    Args:
        y_true: True irrigation values
        et_demand: Crop ET demand (mm)
        stress_risk: Binary stress risk indicator
        alpha: Under-irrigation penalty weight
        beta: Over-irrigation penalty weight
        
    Returns:
        Sample weights array
    """
    base_weights = np.ones(len(y_true))
    
    # Higher weights for high-demand days (more critical)
    demand_weights = 1.0 + (et_demand / np.median(et_demand[et_demand > 0]))
    
    # Higher weights for stress-risk situations
    stress_weights = 1.0 + stress_risk * 0.5
    
    # Combine weights
    weights = base_weights * demand_weights * stress_weights
    
    # Normalize to prevent extreme values
    weights = weights / np.mean(weights)
    
    return weights


class PinballLoss:
    """Pinball (quantile) loss for biased predictions."""
    
    def __init__(self, quantile: float = 0.6):
        """
        Initialize pinball loss.
        
        Args:
            quantile: Target quantile (0.6 biases against under-irrigation)
        """
        self.quantile = quantile
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate pinball loss."""
        error = y_true - y_pred
        loss = np.maximum(self.quantile * error, (self.quantile - 1) * error)
        return np.mean(loss)


# PyTorch implementations for neural networks
class AsymmetricLossTorch(nn.Module):
    """PyTorch implementation of asymmetric loss."""
    
    def __init__(self, alpha: float = 2.0, beta: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Forward pass of asymmetric loss."""
        error = y_pred - y_true
        
        # Under-irrigation penalty
        under_mask = error < 0
        under_loss = self.alpha * torch.abs(error[under_mask])
        
        # Over-irrigation penalty
        over_mask = error >= 0
        over_loss = self.beta * torch.abs(error[over_mask])
        
        # Combine losses
        total_loss = torch.cat([under_loss, over_loss])
        return torch.mean(total_loss)


class PinballLossTorch(nn.Module):
    """PyTorch implementation of pinball loss."""
    
    def __init__(self, quantile: float = 0.6):
        super().__init__()
        self.quantile = quantile
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Forward pass of pinball loss."""
        error = y_true - y_pred
        loss = torch.maximum(
            self.quantile * error, 
            (self.quantile - 1) * error
        )
        return torch.mean(loss)


class CombinedLossTorch(nn.Module):
    """Combined asymmetric + field capacity violation loss."""
    
    def __init__(self, alpha: float = 2.0, beta: float = 1.0, gamma: float = 5.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.asymmetric_loss = AsymmetricLossTorch(alpha, beta)
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor,
                current_theta: torch.Tensor, field_capacity: torch.Tensor,
                root_depth: torch.Tensor, et_forecast: torch.Tensor,
                rain_forecast: torch.Tensor) -> torch.Tensor:
        """Forward pass with FC violation penalty."""
        
        # Base asymmetric loss
        base_loss = self.asymmetric_loss(y_pred, y_true)
        
        # FC violation penalty
        current_mm = current_theta * root_depth * 1000
        fc_mm = field_capacity * root_depth * 1000
        
        projected_mm = current_mm + y_pred + rain_forecast - et_forecast
        violations = torch.clamp(projected_mm - fc_mm, min=0)
        fc_penalty = self.gamma * torch.mean(violations)
        
        return base_loss + fc_penalty


def xgboost_asymmetric_objective(y_true: np.ndarray, y_pred: np.ndarray,
                                alpha: float = 2.0, beta: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Custom XGBoost objective function for asymmetric loss.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        alpha: Under-irrigation penalty
        beta: Over-irrigation penalty
        
    Returns:
        Tuple of (gradient, hessian)
    """
    error = y_pred - y_true
    
    # Gradient
    grad = np.where(error < 0, -alpha, beta)
    
    # Hessian (second derivative, approximated as constant)
    hess = np.ones_like(error) * 0.1  # Small positive value for numerical stability
    
    return grad, hess


def evaluate_asymmetric_performance(y_true: np.ndarray, y_pred: np.ndarray,
                                  threshold: float = 1.0) -> dict:
    """
    Evaluate model performance with asymmetric metrics.
    
    Args:
        y_true: True irrigation values
        y_pred: Predicted irrigation values
        threshold: Threshold for significant under/over irrigation (mm)
        
    Returns:
        Dictionary of performance metrics
    """
    error = y_pred - y_true
    
    # Basic metrics
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))
    
    # Asymmetric metrics
    under_irrigation = error < -threshold
    over_irrigation = error > threshold
    
    under_rate = np.mean(under_irrigation)
    over_rate = np.mean(over_irrigation)
    
    under_mae = np.mean(np.abs(error[under_irrigation])) if np.any(under_irrigation) else 0
    over_mae = np.mean(np.abs(error[over_irrigation])) if np.any(over_irrigation) else 0
    
    # Water use efficiency
    total_predicted = np.sum(y_pred)
    total_actual = np.sum(y_true)
    water_use_error = (total_predicted - total_actual) / total_actual * 100
    
    return {
        'mae': mae,
        'rmse': rmse,
        'under_irrigation_rate': under_rate,
        'over_irrigation_rate': over_rate,
        'under_irrigation_mae': under_mae,
        'over_irrigation_mae': over_mae,
        'water_use_error_pct': water_use_error,
        'asymmetric_score': under_rate * 2 + over_rate  # Weighted score
    }


class LossWeightScheduler:
    """Dynamic loss weight scheduling during training."""
    
    def __init__(self, initial_alpha: float = 2.0, initial_beta: float = 1.0,
                 schedule_type: str = 'constant'):
        """
        Initialize loss weight scheduler.
        
        Args:
            initial_alpha: Initial under-irrigation penalty
            initial_beta: Initial over-irrigation penalty
            schedule_type: 'constant', 'linear_decay', 'exponential_decay'
        """
        self.initial_alpha = initial_alpha
        self.initial_beta = initial_beta
        self.schedule_type = schedule_type
        self.current_epoch = 0
    
    def get_weights(self, epoch: int, total_epochs: int) -> Tuple[float, float]:
        """Get current loss weights based on epoch."""
        self.current_epoch = epoch
        
        if self.schedule_type == 'constant':
            return self.initial_alpha, self.initial_beta
        
        elif self.schedule_type == 'linear_decay':
            # Gradually reduce asymmetry as training progresses
            decay_factor = 1 - (epoch / total_epochs) * 0.5  # Reduce by 50% max
            alpha = self.initial_alpha * decay_factor
            beta = self.initial_beta
            return alpha, beta
        
        elif self.schedule_type == 'exponential_decay':
            # Exponential decay of asymmetry
            decay_rate = 0.95
            alpha = self.initial_alpha * (decay_rate ** epoch)
            beta = self.initial_beta
            return max(alpha, 1.0), beta  # Don't go below 1.0
        
        else:
            return self.initial_alpha, self.initial_beta
