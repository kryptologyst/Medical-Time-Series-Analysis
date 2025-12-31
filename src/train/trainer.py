"""Training utilities for medical time series models."""

import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from omegaconf import DictConfig

from ..utils.device import get_device, clear_gpu_memory
from ..losses.losses import create_loss_function
from ..metrics.metrics import MetricsCalculator


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping.
            min_delta: Minimum change to qualify as improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss: float) -> bool:
        """Check if training should stop.
        
        Args:
            val_loss: Current validation loss.
            
        Returns:
            True if training should stop.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience


class ModelTrainer:
    """Model trainer for medical time series prediction."""
    
    def __init__(self, config: DictConfig, model: nn.Module, device: torch.device):
        """Initialize trainer.
        
        Args:
            config: Configuration object.
            model: Model to train.
            device: Device to train on.
        """
        self.config = config
        self.model = model
        self.device = device
        self.metrics_calculator = MetricsCalculator(config)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Initialize loss function
        self.loss_fn = create_loss_function(config)
        
        # Initialize early stopping
        self.early_stopping = EarlyStopping(
            patience=config.training.early_stopping.patience,
            min_delta=config.training.early_stopping.min_delta
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on configuration."""
        optimizer_name = self.config.training.optimizer.lower()
        lr = self.config.training.learning_rate
        weight_decay = self.config.training.weight_decay
        
        if optimizer_name == "adam":
            return torch.optim.Adam(
                self.model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )
        elif optimizer_name == "sgd":
            return torch.optim.SGD(
                self.model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay,
                momentum=0.9
            )
        elif optimizer_name == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        scheduler_name = self.config.training.scheduler.lower()
        
        if scheduler_name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config.training.num_epochs
            )
        elif scheduler_name == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=30, 
                gamma=0.1
            )
        elif scheduler_name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                patience=5, 
                factor=0.5
            )
        else:
            return None
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader.
            
        Returns:
            Tuple of (average_loss, metrics_dict).
        """
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            all_preds.append(output.detach())
            all_targets.append(target.detach())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate metrics
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        metrics = self.metrics_calculator.calculate_all_metrics(all_targets, all_preds)
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss, metrics
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch.
        
        Args:
            val_loader: Validation data loader.
            
        Returns:
            Tuple of (average_loss, metrics_dict).
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.loss_fn(output, target)
                
                total_loss += loss.item()
                all_preds.append(output)
                all_targets.append(target)
        
        # Calculate metrics
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        metrics = self.metrics_calculator.calculate_all_metrics(all_targets, all_preds)
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss, metrics
    
    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader,
        save_dir: Optional[str] = None
    ) -> Dict[str, list]:
        """Train the model.
        
        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            save_dir: Directory to save checkpoints.
            
        Returns:
            Training history.
        """
        print(f"Starting training for {self.config.training.num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model: {self.config.model.architecture}")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config.training.num_epochs):
            start_time = time.time()
            
            # Training
            train_loss, train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_metrics = self.validate_epoch(val_loader)
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_metrics'].append(train_metrics)
            self.history['val_metrics'].append(val_metrics)
            
            # Print progress
            epoch_time = time.time() - start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch+1}/{self.config.training.num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"  Train R²: {train_metrics['r2']:.4f}, Val R²: {val_metrics['r2']:.4f}")
            print(f"  Train MAE: {train_metrics['mae']:.4f}, Val MAE: {val_metrics['mae']:.4f}")
            print(f"  LR: {current_lr:.6f}, Time: {epoch_time:.2f}s")
            print("-" * 50)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if save_dir:
                    self.save_checkpoint(save_dir, epoch, is_best=True)
            
            # Early stopping
            if self.early_stopping(val_loss):
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            # Clear GPU memory
            clear_gpu_memory()
        
        print("Training completed!")
        return self.history
    
    def save_checkpoint(self, save_dir: str, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint.
        
        Args:
            save_dir: Directory to save checkpoint.
            epoch: Current epoch.
            is_best: Whether this is the best model.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'history': self.history,
        }
        
        # Save regular checkpoint
        checkpoint_path = Path(save_dir) / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = Path(save_dir) / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Epoch: {checkpoint['epoch']}")
