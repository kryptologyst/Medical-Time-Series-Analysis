"""Unit tests for medical time series analysis."""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils.device import set_seed, get_device
from src.models.heart_rate_models import create_model, HeartRateLSTM
from src.losses.losses import create_loss_function
from src.metrics.metrics import RegressionMetrics, ClinicalMetrics
from src.data.synthetic import SyntheticHeartRateGenerator
from src.data.preprocessing import TimeSeriesPreprocessor
from src.data.dataset import HeartRateDataset
from omegaconf import DictConfig


@pytest.fixture
def config():
    """Create a test configuration."""
    return DictConfig({
        'model': {
            'architecture': 'lstm',
            'hidden_size': 32,
            'num_layers': 1,
            'dropout': 0.1,
            'bidirectional': False,
            'transformer': {
                'd_model': 32,
                'nhead': 2,
                'num_layers': 1,
                'dim_feedforward': 64
            },
            'cnn1d': {
                'channels': [16, 32],
                'kernel_sizes': [3, 5],
                'pooling': 'max'
            }
        },
        'training': {
            'loss': {
                'type': 'mse',
                'huber_delta': 1.0,
                'quantile_alpha': 0.5
            }
        },
        'data': {
            'synthetic': {
                'n_samples': 100,
                'sequence_length': 10,
                'prediction_horizon': 1,
                'noise_level': 0.1,
                'trend_strength': 0.5,
                'seasonality_periods': [24, 168]
            },
            'preprocessing': {
                'normalize': True,
                'standardization_method': 'zscore',
                'handle_missing': 'interpolate'
            },
            'splits': {
                'train_ratio': 0.7,
                'val_ratio': 0.15,
                'test_ratio': 0.15,
                'random_seed': 42
            }
        },
        'evaluation': {
            'calibration': {
                'enabled': True,
                'bins': 10
            },
            'uncertainty': {
                'enabled': True,
                'method': 'mc_dropout',
                'n_samples': 10
            }
        }
    })


class TestDeviceUtils:
    """Test device utilities."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        assert True  # If no exception is raised
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
    
    def test_reproducibility(self):
        """Test reproducibility with seeds."""
        set_seed(42)
        a = torch.randn(10)
        
        set_seed(42)
        b = torch.randn(10)
        
        assert torch.allclose(a, b)


class TestModels:
    """Test model architectures."""
    
    def test_lstm_model(self, config):
        """Test LSTM model creation and forward pass."""
        model = HeartRateLSTM(config)
        
        # Test forward pass
        batch_size = 4
        seq_len = config.data.synthetic.sequence_length
        x = torch.randn(batch_size, seq_len, 1)
        
        output = model(x)
        
        assert output.shape == (batch_size,)
        assert not torch.isnan(output).any()
    
    def test_model_creation(self, config):
        """Test model creation for different architectures."""
        architectures = ['lstm', 'gru', 'transformer', 'cnn1d']
        
        for arch in architectures:
            config.model.architecture = arch
            model = create_model(config)
            assert model is not None
    
    def test_model_parameters(self, config):
        """Test that models have trainable parameters."""
        model = create_model(config)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params > 0


class TestLosses:
    """Test loss functions."""
    
    def test_mse_loss(self, config):
        """Test MSE loss."""
        loss_fn = create_loss_function(config)
        
        pred = torch.randn(10)
        target = torch.randn(10)
        
        loss = loss_fn(pred, target)
        assert loss.item() >= 0
    
    def test_huber_loss(self, config):
        """Test Huber loss."""
        config.training.loss.type = 'huber'
        loss_fn = create_loss_function(config)
        
        pred = torch.randn(10)
        target = torch.randn(10)
        
        loss = loss_fn(pred, target)
        assert loss.item() >= 0


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_regression_metrics(self):
        """Test regression metrics."""
        y_true = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = torch.tensor([1.1, 1.9, 3.1, 3.9, 5.1])
        
        mse = RegressionMetrics.mse(y_true, y_pred)
        mae = RegressionMetrics.mae(y_true, y_pred)
        rmse = RegressionMetrics.rmse(y_true, y_pred)
        r2 = RegressionMetrics.r2(y_true, y_pred)
        
        assert mse >= 0
        assert mae >= 0
        assert rmse >= 0
        assert r2 <= 1.0
    
    def test_clinical_metrics(self):
        """Test clinical metrics."""
        y_true = torch.tensor([70.0, 80.0, 90.0, 100.0])
        y_pred = torch.tensor([72.0, 78.0, 92.0, 98.0])
        
        sensitivity = ClinicalMetrics.sensitivity(y_true, y_pred, threshold=5.0)
        specificity = ClinicalMetrics.specificity(y_true, y_pred, threshold=5.0)
        
        assert 0 <= sensitivity <= 1
        assert 0 <= specificity <= 1


class TestDataGeneration:
    """Test data generation."""
    
    def test_synthetic_generator(self, config):
        """Test synthetic data generation."""
        generator = SyntheticHeartRateGenerator(config)
        df = generator.generate_dataset()
        
        assert len(df) > 0
        assert 'patient_id' in df.columns
        assert 'heart_rate' in df.columns
        assert 'timestamp' in df.columns
    
    def test_data_preprocessing(self, config):
        """Test data preprocessing."""
        generator = SyntheticHeartRateGenerator(config)
        df = generator.generate_dataset()
        
        preprocessor = TimeSeriesPreprocessor(config)
        processed_data = preprocessor.preprocess_pipeline(df)
        
        assert 'X_train' in processed_data
        assert 'y_train' in processed_data
        assert 'X_val' in processed_data
        assert 'y_val' in processed_data
        assert 'X_test' in processed_data
        assert 'y_test' in processed_data


class TestDataset:
    """Test PyTorch dataset."""
    
    def test_dataset_creation(self):
        """Test dataset creation."""
        X = np.random.randn(100, 10)
        y = np.random.randn(100, 1)
        
        dataset = HeartRateDataset(X, y)
        
        assert len(dataset) == 100
        
        # Test __getitem__
        x_item, y_item = dataset[0]
        assert x_item.shape == (10, 1)
        assert y_item.shape == (1,)


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_training(self, config):
        """Test end-to-end training pipeline."""
        # Generate data
        generator = SyntheticHeartRateGenerator(config)
        df = generator.generate_dataset()
        
        # Preprocess data
        preprocessor = TimeSeriesPreprocessor(config)
        processed_data = preprocessor.preprocess_pipeline(df)
        
        # Create dataset
        from src.data.dataset import TimeSeriesDataModule
        data_module = TimeSeriesDataModule(processed_data, batch_size=4)
        train_loader, val_loader, test_loader = data_module.get_data_loaders()
        
        # Create model
        model = create_model(config)
        
        # Test forward pass
        for data, target in train_loader:
            output = model(data)
            assert output.shape == target.shape
            break
        
        assert True  # If we get here, the pipeline works


if __name__ == "__main__":
    pytest.main([__file__])
