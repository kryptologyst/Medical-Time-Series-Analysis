#!/usr/bin/env python3
"""Quick test script to verify the medical time series analysis setup."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.utils.device import set_seed, get_device
        from src.data.synthetic import SyntheticHeartRateGenerator
        from src.data.preprocessing import TimeSeriesPreprocessor
        from src.data.dataset import TimeSeriesDataModule
        from src.models.heart_rate_models import create_model
        from src.losses.losses import create_loss_function
        from src.metrics.metrics import MetricsCalculator
        from src.train.trainer import ModelTrainer
        from src.eval.evaluator import ModelEvaluator
        print("✅ All imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality."""
    print("\nTesting basic functionality...")
    
    try:
        from src.utils.device import set_seed, get_device
        from src.data.synthetic import SyntheticHeartRateGenerator
        from src.models.heart_rate_models import create_model
        from omegaconf import DictConfig
        
        # Set seed
        set_seed(42)
        device = get_device()
        print(f"✅ Device: {device}")
        
        # Create minimal config
        config = DictConfig({
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
        
        # Test data generation
        generator = SyntheticHeartRateGenerator(config)
        df = generator.generate_dataset()
        print(f"✅ Generated data: {len(df)} samples for {df['patient_id'].nunique()} patients")
        
        # Test model creation
        model = create_model(config)
        print(f"✅ Created {config.model.architecture} model with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test data preprocessing
        preprocessor = TimeSeriesPreprocessor(config)
        processed_data = preprocessor.preprocess_pipeline(df)
        print(f"✅ Preprocessed data: {len(processed_data['X_train'])} training samples")
        
        # Test data module
        data_module = TimeSeriesDataModule(processed_data, batch_size=16)
        train_loader, val_loader, test_loader = data_module.get_data_loaders()
        print(f"✅ Created data loaders: {len(train_loader)} batches")
        
        # Test model forward pass
        import torch
        model = model.to(device)
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            print(f"✅ Model forward pass: input {data.shape} -> output {output.shape}")
            break
        
        print("✅ All basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("="*60)
    print("MEDICAL TIME SERIES ANALYSIS - QUICK TEST")
    print("="*60)
    print("⚠️ DISCLAIMER: This is a research demonstration.")
    print("NOT FOR CLINICAL USE. NOT MEDICAL ADVICE.")
    print("="*60)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed!")
        return False
    
    # Test basic functionality
    if not test_basic_functionality():
        print("\n❌ Functionality test failed!")
        return False
    
    print("\n" + "="*60)
    print("🎉 ALL TESTS PASSED!")
    print("="*60)
    print("The medical time series analysis framework is ready to use.")
    print("\nNext steps:")
    print("1. Run the demo: streamlit run demo/app.py")
    print("2. Train a model: python scripts/train.py")
    print("3. Explore the notebook: jupyter notebook notebooks/demo.ipynb")
    print("\n⚠️ Remember: This is for research and education only!")
    print("="*60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
