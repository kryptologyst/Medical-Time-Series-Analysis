"""Evaluation utilities for medical time series models."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import seaborn as sns

from ..utils.device import get_device
from ..metrics.metrics import MetricsCalculator


class ModelEvaluator:
    """Model evaluator for medical time series prediction."""
    
    def __init__(self, config, model: torch.nn.Module, device: torch.device):
        """Initialize evaluator.
        
        Args:
            config: Configuration object.
            model: Trained model.
            device: Device to run evaluation on.
        """
        self.config = config
        self.model = model
        self.device = device
        self.metrics_calculator = MetricsCalculator(config)
        
    def evaluate(
        self, 
        test_loader: DataLoader,
        uncertainty: bool = False
    ) -> Dict[str, float]:
        """Evaluate model on test set.
        
        Args:
            test_loader: Test data loader.
            uncertainty: Whether to compute uncertainty metrics.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        self.model.eval()
        all_preds = []
        all_targets = []
        all_uncertainties = []
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Evaluating"):
                data, target = data.to(self.device), target.to(self.device)
                
                if uncertainty and self.config.evaluation.uncertainty.enabled:
                    pred, var = self.predict_with_uncertainty(data)
                    all_uncertainties.append(var)
                else:
                    pred = self.model(data)
                
                all_preds.append(pred)
                all_targets.append(target)
        
        # Concatenate all predictions and targets
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        
        if uncertainty and all_uncertainties:
            all_uncertainties = torch.cat(all_uncertainties)
            metrics = self.metrics_calculator.calculate_all_metrics(
                all_targets, all_preds, all_uncertainties
            )
        else:
            metrics = self.metrics_calculator.calculate_all_metrics(
                all_targets, all_preds
            )
        
        return metrics
    
    def predict_with_uncertainty(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict with uncertainty quantification using MC Dropout.
        
        Args:
            data: Input data.
            
        Returns:
            Tuple of (mean_prediction, variance).
        """
        self.model.train()  # Enable dropout
        
        n_samples = self.config.evaluation.uncertainty.n_samples
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.model(data)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)  # (n_samples, batch_size)
        
        # Calculate mean and variance
        mean_pred = torch.mean(predictions, dim=0)
        var_pred = torch.var(predictions, dim=0)
        
        self.model.eval()  # Disable dropout
        
        return mean_pred, var_pred
    
    def create_evaluation_report(
        self, 
        test_loader: DataLoader,
        save_dir: str,
        model_name: str = "model"
    ) -> Dict[str, float]:
        """Create comprehensive evaluation report.
        
        Args:
            test_loader: Test data loader.
            save_dir: Directory to save results.
            model_name: Name of the model.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        print(f"Creating evaluation report for {model_name}...")
        
        # Evaluate model
        metrics = self.evaluate(test_loader, uncertainty=True)
        
        # Create visualizations
        self._create_prediction_plots(test_loader, save_dir, model_name)
        self._create_calibration_plot(test_loader, save_dir, model_name)
        self._create_error_analysis(test_loader, save_dir, model_name)
        
        # Save metrics
        self._save_metrics(metrics, save_dir, model_name)
        
        # Print summary
        self._print_evaluation_summary(metrics)
        
        return metrics
    
    def _create_prediction_plots(
        self, 
        test_loader: DataLoader, 
        save_dir: str, 
        model_name: str
    ) -> None:
        """Create prediction vs actual plots."""
        self.model.eval()
        
        # Get a batch of data for visualization
        data, target = next(iter(test_loader))
        data, target = data.to(self.device), target.to(self.device)
        
        with torch.no_grad():
            pred = self.model(data)
        
        # Convert to numpy
        pred_np = pred.cpu().numpy()
        target_np = target.cpu().numpy()
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Scatter plot
        axes[0, 0].scatter(target_np, pred_np, alpha=0.6)
        axes[0, 0].plot([target_np.min(), target_np.max()], 
                       [target_np.min(), target_np.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Heart Rate (BPM)')
        axes[0, 0].set_ylabel('Predicted Heart Rate (BPM)')
        axes[0, 0].set_title('Predictions vs Actual')
        
        # Residuals plot
        residuals = pred_np - target_np
        axes[0, 1].scatter(target_np, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Actual Heart Rate (BPM)')
        axes[0, 1].set_ylabel('Residuals (BPM)')
        axes[0, 1].set_title('Residuals Plot')
        
        # Time series plot (first sample)
        sample_idx = 0
        axes[1, 0].plot(data[sample_idx, :, 0].cpu().numpy(), label='Input Sequence')
        axes[1, 0].axhline(y=target_np[sample_idx], color='g', linestyle='--', 
                          label=f'Actual: {target_np[sample_idx]:.1f}')
        axes[1, 0].axhline(y=pred_np[sample_idx], color='r', linestyle='--', 
                          label=f'Predicted: {pred_np[sample_idx]:.1f}')
        axes[1, 0].set_xlabel('Time Steps')
        axes[1, 0].set_ylabel('Heart Rate (BPM)')
        axes[1, 0].set_title('Sample Prediction')
        axes[1, 0].legend()
        
        # Error distribution
        axes[1, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Prediction Error (BPM)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Error Distribution')
        
        plt.tight_layout()
        plt.savefig(Path(save_dir) / f"{model_name}_predictions.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_calibration_plot(
        self, 
        test_loader: DataLoader, 
        save_dir: str, 
        model_name: str
    ) -> None:
        """Create calibration plot for uncertainty quantification."""
        if not self.config.evaluation.calibration.enabled:
            return
        
        self.model.eval()
        
        # Collect predictions with uncertainty
        all_preds = []
        all_targets = []
        all_uncertainties = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                pred, var = self.predict_with_uncertainty(data)
                
                all_preds.append(pred)
                all_targets.append(target)
                all_uncertainties.append(var)
        
        all_preds = torch.cat(all_preds).cpu().numpy()
        all_targets = torch.cat(all_targets).cpu().numpy()
        all_uncertainties = torch.cat(all_uncertainties).cpu().numpy()
        
        # Create calibration plot
        plt.figure(figsize=(10, 6))
        
        # Calculate confidence intervals
        confidence_levels = np.linspace(0.1, 0.9, 9)
        empirical_coverage = []
        
        for conf_level in confidence_levels:
            z_score = 1.96  # 95% confidence
            lower_bound = all_preds - z_score * np.sqrt(all_uncertainties)
            upper_bound = all_preds + z_score * np.sqrt(all_uncertainties)
            
            in_interval = (all_targets >= lower_bound) & (all_targets <= upper_bound)
            coverage = np.mean(in_interval)
            empirical_coverage.append(coverage)
        
        plt.plot(confidence_levels, empirical_coverage, 'bo-', label='Empirical Coverage')
        plt.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
        plt.xlabel('Expected Coverage')
        plt.ylabel('Empirical Coverage')
        plt.title('Calibration Plot')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(Path(save_dir) / f"{model_name}_calibration.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_error_analysis(
        self, 
        test_loader: DataLoader, 
        save_dir: str, 
        model_name: str
    ) -> None:
        """Create error analysis plots."""
        self.model.eval()
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                pred = self.model(data)
                
                all_preds.append(pred)
                all_targets.append(target)
        
        all_preds = torch.cat(all_preds).cpu().numpy()
        all_targets = torch.cat(all_targets).cpu().numpy()
        
        errors = np.abs(all_preds - all_targets)
        
        # Create error analysis plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Error vs prediction
        axes[0, 0].scatter(all_preds, errors, alpha=0.6)
        axes[0, 0].set_xlabel('Predicted Heart Rate (BPM)')
        axes[0, 0].set_ylabel('Absolute Error (BPM)')
        axes[0, 0].set_title('Error vs Prediction')
        
        # Error vs actual
        axes[0, 1].scatter(all_targets, errors, alpha=0.6)
        axes[0, 1].set_xlabel('Actual Heart Rate (BPM)')
        axes[0, 1].set_ylabel('Absolute Error (BPM)')
        axes[0, 1].set_title('Error vs Actual')
        
        # Error distribution by ranges
        hr_ranges = [(40, 60), (60, 80), (80, 100), (100, 120), (120, 200)]
        range_errors = []
        range_labels = []
        
        for low, high in hr_ranges:
            mask = (all_targets >= low) & (all_targets < high)
            if np.sum(mask) > 0:
                range_errors.append(errors[mask])
                range_labels.append(f'{low}-{high}')
        
        axes[1, 0].boxplot(range_errors, labels=range_labels)
        axes[1, 0].set_xlabel('Heart Rate Range (BPM)')
        axes[1, 0].set_ylabel('Absolute Error (BPM)')
        axes[1, 0].set_title('Error Distribution by HR Range')
        
        # Cumulative error distribution
        sorted_errors = np.sort(errors)
        cumulative_prob = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        axes[1, 1].plot(sorted_errors, cumulative_prob)
        axes[1, 1].set_xlabel('Absolute Error (BPM)')
        axes[1, 1].set_ylabel('Cumulative Probability')
        axes[1, 1].set_title('Cumulative Error Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(Path(save_dir) / f"{model_name}_error_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_metrics(self, metrics: Dict[str, float], save_dir: str, model_name: str) -> None:
        """Save metrics to file."""
        metrics_df = pd.DataFrame([metrics])
        metrics_path = Path(save_dir) / f"{model_name}_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Metrics saved to {metrics_path}")
    
    def _print_evaluation_summary(self, metrics: Dict[str, float]) -> None:
        """Print evaluation summary."""
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        
        print(f"Regression Metrics:")
        print(f"  MSE:  {metrics['mse']:.4f}")
        print(f"  MAE:  {metrics['mae']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAPE: {metrics['mape']:.4f}%")
        print(f"  R²:   {metrics['r2']:.4f}")
        
        print(f"\nClinical Metrics:")
        print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  PPV:         {metrics['ppv']:.4f}")
        print(f"  NPV:         {metrics['npv']:.4f}")
        print(f"  Clinical Accuracy: {metrics['clinical_accuracy']:.4f}")
        
        if 'brier_score' in metrics:
            print(f"\nUncertainty Metrics:")
            print(f"  Brier Score: {metrics['brier_score']:.4f}")
            print(f"  ECE:         {metrics['ece']:.4f}")
        
        print("="*50)
