"""Medical Time Series Analysis Package."""

__version__ = "1.0.0"
__author__ = "Healthcare AI Research Team"
__email__ = "research@example.com"

# Import main components
from .models.heart_rate_models import create_model, HeartRateLSTM, HeartRateGRU, HeartRateTransformer, HeartRateCNN1D
from .data.synthetic import SyntheticHeartRateGenerator
from .data.preprocessing import TimeSeriesPreprocessor
from .data.dataset import HeartRateDataset, TimeSeriesDataModule
from .losses.losses import create_loss_function
from .metrics.metrics import MetricsCalculator, RegressionMetrics, ClinicalMetrics, CalibrationMetrics
from .utils.device import get_device, set_seed, get_device_info
from .utils.config import load_config, save_config, validate_config

__all__ = [
    # Models
    "create_model",
    "HeartRateLSTM", 
    "HeartRateGRU",
    "HeartRateTransformer",
    "HeartRateCNN1D",
    
    # Data
    "SyntheticHeartRateGenerator",
    "TimeSeriesPreprocessor", 
    "HeartRateDataset",
    "TimeSeriesDataModule",
    
    # Losses
    "create_loss_function",
    
    # Metrics
    "MetricsCalculator",
    "RegressionMetrics",
    "ClinicalMetrics", 
    "CalibrationMetrics",
    
    # Utils
    "get_device",
    "set_seed",
    "get_device_info",
    "load_config",
    "save_config", 
    "validate_config",
]
