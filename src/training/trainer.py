"""
Training pipeline for LSTM Sentiment Classifier.

This module provides comprehensive training functionality including
training loops, validation, checkpointing, and early stopping.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
from datetime import datetime

from models.lstm_model import LSTMClassifier
from data.text_preprocessor import TextPreprocessor


class Trainer:
    """
    Comprehensive trainer for LSTM sentiment classification model.
    
    Handles training loops, validation, checkpointing, early stopping,
    and training progress tracking.
    """
    
    def __init__(
        self,
        model: LSTMClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: str = None,
        scheduler: optim.lr_scheduler._LRScheduler = None,
        gradient_clip: float = 1.0,
        checkpoint_dir: str = "checkpoints",
        log_interval: int = 100
    ):
        """
        Initialize the trainer.
        
        Args:
            model: LSTM model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer for training
            criterion: Loss function
            device: Device to train on
            scheduler: Learning rate scheduler (optional)
            gradient_clip: Gradient clipping threshold
            checkpoint_dir: Directory to save checkpoints
            log_interval: Logging interval for training progress
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.scheduler = scheduler
        self.gradient_clip = gradient_clip
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.learning_rates = []
        
        # Early stopping
        self.early_stopping_patience = None
        self.early_stopping_counter = 0
        self.early_stopping_min_delta = 0.001
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Move model to device
        self.model.to(self.device)
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train the model for one epoch.
        
        Returns:
            Dictionary containing training metrics for the epoch
        """
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        batch_losses = []
        
        start_time = time.time()
        
        for batch_idx, (data, targets) in enumerate(self.train_loader):
            # Move data to device
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(data)
            
            # Calculate loss
            loss = self.criterion(outputs.squeeze(), targets.float())
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            
            # Optimizer step
            self.optimizer.step()
            
            # Calculate accuracy
            predictions = torch.sigmoid(outputs.squeeze()) >= 0.5
            correct = (predictions == targets).sum().item()
            
            # Update statistics
            total_loss += loss.item()
            total_correct += correct
            total_samples += targets.size(0)
            batch_losses.append(loss.item())
            
            # Log progress
            if batch_idx % self.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.info(
                    f'Epoch {self.current_epoch}, Batch {batch_idx}/{len(self.train_loader)}: '
                    f'Loss: {loss.item():.6f}, '
                    f'Accuracy: {100. * correct / targets.size(0):.2f}%, '
                    f'LR: {current_lr:.6f}'
                )
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * total_correct / total_samples
        epoch_time = time.time() - start_time
        
        # Store learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.learning_rates.append(current_lr)
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'time': epoch_time,
            'learning_rate': current_lr,
            'samples_processed': total_samples
        }
        
        self.train_losses.append(avg_loss)
        
        return metrics
    
    def evaluate_model(self, data_loader: DataLoader = None) -> Dict[str, float]:
        """
        Evaluate the model on validation or test data.
        
        Args:
            data_loader: Data loader to evaluate on (uses val_loader if None)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if data_loader is None:
            data_loader = self.val_loader
        
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_predictions = []
        all_targets = []
        
        start_time = time.time()
        
        with torch.no_grad():
            for data, targets in data_loader:
                # Move data to device
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(data)
                
                # Calculate loss
                loss = self.criterion(outputs.squeeze(), targets.float())
                
                # Calculate predictions
                probabilities = torch.sigmoid(outputs.squeeze())
                predictions = probabilities >= 0.5
                correct = (predictions == targets).sum().item()
                
                # Update statistics
                total_loss += loss.item()
                total_correct += correct
                total_samples += targets.size(0)
                
                # Store for detailed metrics
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(data_loader)
        accuracy = 100. * total_correct / total_samples
        eval_time = time.time() - start_time
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'time': eval_time,
            'samples_evaluated': total_samples
        }
        
        return metrics
    
    def train(
        self,
        epochs: int,
        early_stopping_patience: int = None,
        save_best: bool = True,
        save_every: int = None
    ) -> Dict[str, List[float]]:
        """
        Train the model for specified number of epochs.
        
        Args:
            epochs: Number of epochs to train
            early_stopping_patience: Patience for early stopping (None to disable)
            save_best: Whether to save the best model
            save_every: Save checkpoint every N epochs (None to disable)
            
        Returns:
            Dictionary containing training history
        """
        self.early_stopping_patience = early_stopping_patience
        
        self.logger.info(f"Starting training for {epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        training_start_time = time.time()
        
        for epoch in range(epochs):
            self.current_epoch = epoch + 1
            
            # Training phase
            train_metrics = self.train_epoch()
            
            # Validation phase
            val_metrics = self.evaluate_model()
            
            # Store validation metrics
            self.val_losses.append(val_metrics['loss'])
            self.val_accuracies.append(val_metrics['accuracy'])
            
            # Learning rate scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Log epoch results
            self.logger.info(
                f"Epoch {self.current_epoch}/{epochs} - "
                f"Train Loss: {train_metrics['loss']:.6f}, "
                f"Train Acc: {train_metrics['accuracy']:.2f}%, "
                f"Val Loss: {val_metrics['loss']:.6f}, "
                f"Val Acc: {val_metrics['accuracy']:.2f}%, "
                f"Time: {train_metrics['time']:.2f}s"
            )
            
            # Check for best model
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                self.best_val_accuracy = val_metrics['accuracy']
                
                if save_best:
                    self.save_checkpoint(
                        filename=f'best_model_epoch_{self.current_epoch}.pth',
                        is_best=True,
                        metrics={
                            'train_loss': train_metrics['loss'],
                            'val_loss': val_metrics['loss'],
                            'val_accuracy': val_metrics['accuracy']
                        }
                    )
            
            # Regular checkpoint saving
            if save_every and self.current_epoch % save_every == 0:
                self.save_checkpoint(
                    filename=f'checkpoint_epoch_{self.current_epoch}.pth',
                    metrics={
                        'train_loss': train_metrics['loss'],
                        'val_loss': val_metrics['loss'],
                        'val_accuracy': val_metrics['accuracy']
                    }
                )
            
            # Early stopping check
            if self._check_early_stopping(val_metrics['loss']):
                self.logger.info(f"Early stopping triggered at epoch {self.current_epoch}")
                break
        
        total_training_time = time.time() - training_start_time
        
        self.logger.info(f"Training completed in {total_training_time:.2f} seconds")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.6f}")
        self.logger.info(f"Best validation accuracy: {self.best_val_accuracy:.2f}%")
        
        # Return training history
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates,
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy,
            'total_epochs': self.current_epoch,
            'total_time': total_training_time
        }
    
    def _check_early_stopping(self, val_loss: float) -> bool:
        """
        Check if early stopping should be triggered.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if early stopping should be triggered
        """
        if self.early_stopping_patience is None:
            return False
        
        if val_loss < self.best_val_loss - self.early_stopping_min_delta:
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
        
        return self.early_stopping_counter >= self.early_stopping_patience    

    def save_checkpoint(
        self,
        filename: str = None,
        is_best: bool = False,
        metrics: Dict[str, float] = None
    ) -> str:
        """
        Save model checkpoint with training state.
        
        Args:
            filename: Checkpoint filename (auto-generated if None)
            is_best: Whether this is the best model so far
            metrics: Current metrics to save with checkpoint
            
        Returns:
            Path to saved checkpoint file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'checkpoint_epoch_{self.current_epoch}_{timestamp}.pth'
        
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        # Prepare checkpoint data
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': self.model.get_model_info(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates,
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy,
            'gradient_clip': self.gradient_clip,
            'is_best': is_best,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add scheduler state if available
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Add current metrics
        if metrics is not None:
            checkpoint['current_metrics'] = metrics
        
        # Save checkpoint
        torch.save(checkpoint, filepath)
        
        self.logger.info(f"Checkpoint saved: {filepath}")
        
        # Save best model copy
        if is_best:
            best_filepath = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_filepath)
            self.logger.info(f"Best model saved: {best_filepath}")
        
        return filepath
    
    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True) -> Dict[str, Any]:
        """
        Load model checkpoint and restore training state.
        
        Args:
            checkpoint_path: Path to checkpoint file
            load_optimizer: Whether to load optimizer state
            
        Returns:
            Dictionary containing checkpoint information
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        self.logger.info(f"Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore training state
        self.current_epoch = checkpoint.get('epoch', 0)
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
        self.learning_rates = checkpoint.get('learning_rates', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_accuracy = checkpoint.get('best_val_accuracy', 0.0)
        
        self.logger.info(f"Checkpoint loaded successfully")
        self.logger.info(f"Resumed from epoch {self.current_epoch}")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.6f}")
        
        return checkpoint
    
    def resume_training(
        self,
        checkpoint_path: str,
        additional_epochs: int,
        early_stopping_patience: int = None
    ) -> Dict[str, List[float]]:
        """
        Resume training from a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            additional_epochs: Number of additional epochs to train
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Dictionary containing complete training history
        """
        # Load checkpoint
        checkpoint_info = self.load_checkpoint(checkpoint_path)
        
        # Continue training
        start_epoch = self.current_epoch
        total_epochs = start_epoch + additional_epochs
        
        self.logger.info(f"Resuming training from epoch {start_epoch} to {total_epochs}")
        
        return self.train(
            epochs=additional_epochs,
            early_stopping_patience=early_stopping_patience
        )
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive training summary.
        
        Returns:
            Dictionary containing training statistics and information
        """
        if not self.train_losses:
            return {'status': 'No training completed yet'}
        
        summary = {
            'training_status': {
                'epochs_completed': self.current_epoch,
                'total_batches_processed': len(self.train_losses) * len(self.train_loader),
                'best_epoch': self.val_losses.index(min(self.val_losses)) + 1 if self.val_losses else None
            },
            'performance_metrics': {
                'best_val_loss': self.best_val_loss,
                'best_val_accuracy': self.best_val_accuracy,
                'final_train_loss': self.train_losses[-1] if self.train_losses else None,
                'final_val_loss': self.val_losses[-1] if self.val_losses else None,
                'final_val_accuracy': self.val_accuracies[-1] if self.val_accuracies else None
            },
            'training_progress': {
                'train_loss_improvement': (
                    self.train_losses[0] - self.train_losses[-1]
                    if len(self.train_losses) > 1 else 0
                ),
                'val_loss_improvement': (
                    self.val_losses[0] - self.best_val_loss
                    if self.val_losses else 0
                ),
                'accuracy_improvement': (
                    self.best_val_accuracy - (self.val_accuracies[0] if self.val_accuracies else 0)
                )
            },
            'model_info': self.model.get_model_info(),
            'training_config': {
                'device': str(self.device),
                'gradient_clip': self.gradient_clip,
                'optimizer': type(self.optimizer).__name__,
                'scheduler': type(self.scheduler).__name__ if self.scheduler else None,
                'criterion': type(self.criterion).__name__
            }
        }
        
        return summary


def create_trainer(
    model: LSTMClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    gradient_clip: float = 1.0,
    scheduler_type: str = 'plateau',
    device: str = None,
    checkpoint_dir: str = "checkpoints"
) -> Trainer:
    """
    Factory function to create a trainer with common configurations.
    
    Args:
        model: LSTM model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for regularization
        gradient_clip: Gradient clipping threshold
        scheduler_type: Type of learning rate scheduler ('plateau', 'step', 'cosine', None)
        device: Device to train on
        checkpoint_dir: Directory for checkpoints
        
    Returns:
        Configured Trainer instance
    """
    # Setup optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Setup loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Setup scheduler
    scheduler = None
    if scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
    elif scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=5, gamma=0.5
        )
    elif scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=20, eta_min=1e-6
        )
    
    return Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        gradient_clip=gradient_clip,
        device=device,
        checkpoint_dir=checkpoint_dir
    )