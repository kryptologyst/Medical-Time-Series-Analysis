"""Evaluation metrics for medical time series prediction."""

import numpy as np
import torch
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score,
    mean_absolute_percentage_error
)
from typing import Dict, List, Optional, Tuple, Union


class RegressionMetrics:
    """Regression metrics for time series prediction."""
    
    @staticmethod
    def mse(y_true: Union[np.ndarray, torch.Tensor], 
            y_pred: Union[np.ndarray, torch.Tensor]) -> float:
        """Mean Squared Error."""
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        return mean_squared_error(y_true, y_pred)
    
    @staticmethod
    def mae(y_true: Union[np.ndarray, torch.Tensor], 
            y_pred: Union[np.ndarray, torch.Tensor]) -> float:
        """Mean Absolute Error."""
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def rmse(y_true: Union[np.ndarray, torch.Tensor], 
             y_pred: Union[np.ndarray, torch.Tensor]) -> float:
        """Root Mean Squared Error."""
        return np.sqrt(RegressionMetrics.mse(y_true, y_pred))
    
    @staticmethod
    def mape(y_true: Union[np.ndarray, torch.Tensor], 
             y_pred: Union[np.ndarray, torch.Tensor]) -> float:
        """Mean Absolute Percentage Error."""
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        return mean_absolute_percentage_error(y_true, y_pred)
    
    @staticmethod
    def r2(y_true: Union[np.ndarray, torch.Tensor], 
           y_pred: Union[np.ndarray, torch.Tensor]) -> float:
        """R-squared coefficient."""
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        return r2_score(y_true, y_pred)


class ClinicalMetrics:
    """Clinical metrics for medical time series prediction."""
    
    @staticmethod
    def sensitivity(y_true: Union[np.ndarray, torch.Tensor], 
                   y_pred: Union[np.ndarray, torch.Tensor], 
                   threshold: float = 10.0) -> float:
        """Sensitivity: proportion of true positives correctly identified.
        
        Args:
            y_true: True values.
            y_pred: Predicted values.
            threshold: Threshold for considering prediction as correct (BPM).
            
        Returns:
            Sensitivity score.
        """
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        
        errors = np.abs(y_true - y_pred)
        correct_predictions = errors <= threshold
        return np.mean(correct_predictions)
    
    @staticmethod
    def specificity(y_true: Union[np.ndarray, torch.Tensor], 
                    y_pred: Union[np.ndarray, torch.Tensor], 
                    threshold: float = 10.0) -> float:
        """Specificity: proportion of true negatives correctly identified."""
        return ClinicalMetrics.sensitivity(y_true, y_pred, threshold)
    
    @staticmethod
    def ppv(y_true: Union[np.ndarray, torch.Tensor], 
            y_pred: Union[np.ndarray, torch.Tensor], 
            threshold: float = 10.0) -> float:
        """Positive Predictive Value."""
        return ClinicalMetrics.sensitivity(y_true, y_pred, threshold)
    
    @staticmethod
    def npv(y_true: Union[np.ndarray, torch.Tensor], 
            y_pred: Union[np.ndarray, torch.Tensor], 
            threshold: float = 10.0) -> float:
        """Negative Predictive Value."""
        return ClinicalMetrics.sensitivity(y_true, y_pred, threshold)
    
    @staticmethod
    def clinical_accuracy(y_true: Union[np.ndarray, torch.Tensor], 
                          y_pred: Union[np.ndarray, torch.Tensor], 
                          threshold: float = 10.0) -> float:
        """Clinical accuracy within threshold."""
        return ClinicalMetrics.sensitivity(y_true, y_pred, threshold)


class CalibrationMetrics:
    """Calibration metrics for uncertainty quantification."""
    
    @staticmethod
    def brier_score(y_true: Union[np.ndarray, torch.Tensor], 
                    y_pred: Union[np.ndarray, torch.Tensor], 
                    y_var: Optional[Union[np.ndarray, torch.Tensor]] = None) -> float:
        """Brier score for calibration."""
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        if y_var is not None and isinstance(y_var, torch.Tensor):
            y_var = y_var.detach().cpu().numpy()
        
        if y_var is None:
            # Simple Brier score (MSE)
            return np.mean((y_true - y_pred) ** 2)
        else:
            # Proper Brier score with uncertainty
            return np.mean((y_true - y_pred) ** 2 + y_var)
    
    @staticmethod
    def expected_calibration_error(y_true: Union[np.ndarray, torch.Tensor], 
                                  y_pred: Union[np.ndarray, torch.Tensor], 
                                  y_var: Union[np.ndarray, torch.Tensor], 
                                  bins: int = 10) -> float:
        """Expected Calibration Error."""
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        if isinstance(y_var, torch.Tensor):
            y_var = y_var.detach().cpu().numpy()
        
        # Convert variance to confidence intervals
        y_std = np.sqrt(y_var)
        confidence_intervals = np.linspace(0.1, 0.9, bins)
        
        ece = 0.0
        total_samples = len(y_true)
        
        for ci in confidence_intervals:
            # Find samples within confidence interval
            z_score = 1.96  # 95% confidence
            lower_bound = y_pred - z_score * y_std
            upper_bound = y_pred + z_score * y_std
            
            in_interval = (y_true >= lower_bound) & (y_true <= upper_bound)
            n_in_interval = np.sum(in_interval)
            
            if n_in_interval > 0:
                empirical_coverage = n_in_interval / total_samples
                expected_coverage = ci
                ece += np.abs(empirical_coverage - expected_coverage) * n_in_interval
        
        return ece / total_samples


class MetricsCalculator:
    """Comprehensive metrics calculator."""
    
    def __init__(self, config):
        """Initialize metrics calculator.
        
        Args:
            config: Configuration object.
        """
        self.config = config
        self.regression_metrics = RegressionMetrics()
        self.clinical_metrics = ClinicalMetrics()
        self.calibration_metrics = CalibrationMetrics()
    
    def calculate_all_metrics(
        self, 
        y_true: Union[np.ndarray, torch.Tensor], 
        y_pred: Union[np.ndarray, torch.Tensor],
        y_var: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Dict[str, float]:
        """Calculate all metrics.
        
        Args:
            y_true: True values.
            y_pred: Predicted values.
            y_var: Predicted variance (for uncertainty quantification).
            
        Returns:
            Dictionary of metric names and values.
        """
        metrics = {}
        
        # Regression metrics
        metrics['mse'] = self.regression_metrics.mse(y_true, y_pred)
        metrics['mae'] = self.regression_metrics.mae(y_true, y_pred)
        metrics['rmse'] = self.regression_metrics.rmse(y_true, y_pred)
        metrics['mape'] = self.regression_metrics.mape(y_true, y_pred)
        metrics['r2'] = self.regression_metrics.r2(y_true, y_pred)
        
        # Clinical metrics
        clinical_threshold = 10.0  # BPM
        metrics['sensitivity'] = self.clinical_metrics.sensitivity(
            y_true, y_pred, clinical_threshold
        )
        metrics['specificity'] = self.clinical_metrics.specificity(
            y_true, y_pred, clinical_threshold
        )
        metrics['ppv'] = self.clinical_metrics.ppv(
            y_true, y_pred, clinical_threshold
        )
        metrics['npv'] = self.clinical_metrics.npv(
            y_true, y_pred, clinical_threshold
        )
        metrics['clinical_accuracy'] = self.clinical_metrics.clinical_accuracy(
            y_true, y_pred, clinical_threshold
        )
        
        # Calibration metrics
        if y_var is not None:
            metrics['brier_score'] = self.calibration_metrics.brier_score(
                y_true, y_pred, y_var
            )
            metrics['ece'] = self.calibration_metrics.expected_calibration_error(
                y_true, y_pred, y_var, bins=self.config.evaluation.calibration.bins
            )
        
        return metrics
