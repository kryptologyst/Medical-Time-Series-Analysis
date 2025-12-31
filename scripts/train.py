#!/usr/bin/env python3
"""Main training script for medical time series analysis."""

import argparse
import os
import sys
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.device import set_seed, get_device, get_device_info
from src.utils.config import load_config, create_experiment_dir, validate_config
from src.data.synthetic import SyntheticHeartRateGenerator
from src.data.preprocessing import TimeSeriesPreprocessor
from src.data.dataset import TimeSeriesDataModule
from src.models.heart_rate_models import create_model
from src.train.trainer import ModelTrainer
from src.eval.evaluator import ModelEvaluator


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train medical time series model")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--overrides", nargs="*", default=[],
                       help="Configuration overrides")
    parser.add_argument("--experiment-dir", type=str, default="experiments",
                       help="Base directory for experiments")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config, args.overrides)
    
    # Validate configuration
    if not validate_config(config):
        print("Configuration validation failed!")
        sys.exit(1)
    
    # Set random seeds for reproducibility
    set_seed(config.seed, config.deterministic)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    print(f"Device info: {get_device_info()}")
    
    # Create experiment directory
    exp_dir = create_experiment_dir(config, args.experiment_dir)
    print(f"Experiment directory: {exp_dir}")
    
    # Save configuration
    config_path = exp_dir / "configs" / "config.yaml"
    OmegaConf.save(config, config_path)
    
    # Generate synthetic data
    print("Generating synthetic data...")
    generator = SyntheticHeartRateGenerator(config)
    df = generator.generate_dataset()
    print(f"Generated data for {df['patient_id'].nunique()} patients")
    print(f"Total samples: {len(df)}")
    
    # Preprocess data
    print("Preprocessing data...")
    preprocessor = TimeSeriesPreprocessor(config)
    processed_data = preprocessor.preprocess_pipeline(df)
    
    # Create data module
    data_module = TimeSeriesDataModule(
        processed_data,
        batch_size=config.training.batch_size,
        num_workers=4,
        pin_memory=True
    )
    
    train_loader, val_loader, test_loader = data_module.get_data_loaders()
    dataset_info = data_module.get_dataset_info()
    
    print(f"Dataset info: {dataset_info}")
    
    # Create model
    print(f"Creating {config.model.architecture} model...")
    model = create_model(config)
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = ModelTrainer(config, model, device)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train model
    print("Starting training...")
    history = trainer.train(train_loader, val_loader, str(exp_dir / "checkpoints"))
    
    # Save training history
    import json
    history_path = exp_dir / "results" / "training_history.json"
    with open(history_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_history = {}
        for key, value in history.items():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                json_history[key] = value
            else:
                json_history[key] = value
        json.dump(json_history, f, indent=2)
    
    # Evaluate model
    print("Evaluating model...")
    evaluator = ModelEvaluator(config, model, device)
    metrics = evaluator.create_evaluation_report(
        test_loader, 
        str(exp_dir / "results"),
        config.model.architecture
    )
    
    # Create leaderboard
    create_leaderboard(exp_dir, metrics, config)
    
    print(f"\nTraining completed! Results saved to: {exp_dir}")
    print("DISCLAIMER: This is a research demonstration. Not for clinical use.")


def create_leaderboard(exp_dir: Path, metrics: dict, config: DictConfig) -> None:
    """Create a simple leaderboard."""
    leaderboard_path = exp_dir / "results" / "leaderboard.csv"
    
    import pandas as pd
    
    # Create leaderboard entry
    entry = {
        'model': config.model.architecture,
        'experiment': config.experiment.name,
        'version': config.experiment.version,
        'mse': metrics['mse'],
        'mae': metrics['mae'],
        'rmse': metrics['rmse'],
        'r2': metrics['r2'],
        'clinical_accuracy': metrics['clinical_accuracy'],
        'sensitivity': metrics['sensitivity'],
        'specificity': metrics['specificity'],
    }
    
    # Load existing leaderboard or create new one
    if leaderboard_path.exists():
        df = pd.read_csv(leaderboard_path)
        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    else:
        df = pd.DataFrame([entry])
    
    # Sort by R² score
    df = df.sort_values('r2', ascending=False)
    df.to_csv(leaderboard_path, index=False)
    
    print(f"\nLeaderboard updated: {leaderboard_path}")
    print(df[['model', 'mse', 'mae', 'r2', 'clinical_accuracy']].to_string(index=False))


if __name__ == "__main__":
    main()
