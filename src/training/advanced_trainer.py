"""
Advanced trainer with enhanced optimization strategies.

This module extends the base trainer with advanced optimization techniques,
convergence analysis, and comprehensive training monitoring.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
from datetime import datetime
import numpy as np

from models.lstm_model import LSTMClassifier
from training.trainer import Trainer
from training.advanced_optimizer import (
    OptimizerFactory,
    SchedulerFactory,
    GradientManager,
    ConvergenceAnalyzer,
    OptimizerComparison
)


class AdvancedTrainer(Trainer):
    """
    Enhanced trainer with advanced optimization strategies.
    
    Extends the base Trainer class with:
    - Multiple optimizer types and comparison
    - Advanced learning rate scheduling
    - Gradient accumulation and enhanced clipping
    - Convergence analysis and improved early stopping
    - Comprehensive training monitoring
    """
    
    def __init__(
        self,
        model: LSTMClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer_config: Dict[str, Any],
        scheduler_config: Dict[str, Any],
        gradient_config: Dict[str, Any] = None,
        convergence_config: Dict[str, Any] = None,
        device: str = None,
        checkpoint_dir: str = "checkpoints",
        log_interval: int = 100,
        comparison_mode: bool = False
    ):
        """
        Initialize advanced trainer.
        
        Args:
            model: LSTM model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer_config: Optimizer configuration
            scheduler_config: Scheduler configuration
            gradient_config: Gradient management configuration
            convergence_config: Convergence analysis configuration
            device: Device to train on
            checkpoint_dir: Directory for checkpoints
            log_interval: Logging interval
            comparison_mode: Whether to enable optimizer comparison
        """
        # Initialize gradient manager
        gradient_config = gradient_config or {}
        self.gradient_manager = GradientManager(
            clip_type=gradient_config.get('clip_type', 'norm'),
            clip_value=gradient_config.get('clip_value', 1.0),
            accumulation_steps=gradient_config.get('accumulation_steps', 1)
        )
        
        # Create optimizer
        optimizer = OptimizerFactory.create_optimizer(
            model.parameters(),
            **optimizer_config
        )
        
        # Create scheduler (will be set after determining total steps)
        self.scheduler_config = scheduler_config
        scheduler = None  # Will be created in train() method
        
        # Initialize convergence analyzer
        convergence_config = convergence_config or {}
        self.convergence_analyzer = ConvergenceAnalyzer(
            patience=convergence_config.get('patience', 10),
            min_delta=convergence_config.get('min_delta', 1e-4),
            restore_best_weights=convergence_config.get('restore_best_weights', True),
            monitor_metric=convergence_config.get('monitor_metric', 'val_loss'),
            mode=convergence_config.get('mode', 'min'),
            warmup_epochs=convergence_config.get('warmup_epochs', 5)
        )
        
        # Loss function
        criterion = nn.BCEWithLogitsLoss()
        
        # Initialize base trainer
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scheduler=scheduler,
            gradient_clip=0,  # We handle this in gradient_manager
            checkpoint_dir=checkpoint_dir,
            log_interval=log_interval
        )
        
        # Advanced training state
        self.optimizer_config = optimizer_config
        self.gradient_stats = []
        self.convergence_history = []
        self.comparison_mode = comparison_mode
        
        # Optimizer comparison framework
        if comparison_mode:
            self.optimizer_comparison = OptimizerComparison(
                save_dir=os.path.join(checkpoint_dir, "optimizer_comparison")
            )
        else:
            self.optimizer_comparison = None
        
        self.logger.info(f"Advanced trainer initialized with {optimizer_config['optimizer_type']} optimizer")
        self.logger.info(f"Gradient management: {gradient_config}")
        self.logger.info(f"Convergence analysis: {convergence_config}")
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train the model for one epoch with advanced optimization.
        
        Returns:
            Dictionary containing training metrics for the epoch
        """
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        batch_losses = []
        accumulated_loss = 0.0
        
        start_time = time.time()
        
        for batch_idx, (data, targets) in enumerate(self.train_loader):
            # Move data to device
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Forward pass
            outputs = self.model(data)
            
            # Calculate loss (normalize by accumulation steps)
            loss = self.criterion(outputs.squeeze(), targets.float())
            loss = loss / self.gradient_manager.accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Accumulate loss for logging
            accumulated_loss += loss.item()
            
            # Gradient management and optimizer step
            if self.gradient_manager.should_step():
                # Apply gradient clipping and get gradient norm
                grad_norm = self.gradient_manager.clip_gradients(self.model)
                
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Store gradient statistics
                self.gradient_stats.append({
                    'epoch': self.current_epoch,
                    'batch': batch_idx,
                    'grad_norm': grad_norm,
                    'loss': accumulated_loss
                })
                
                # Reset accumulated loss
                accumulated_loss = 0.0
            
            # Calculate accuracy
            predictions = torch.sigmoid(outputs.squeeze()) >= 0.5
            correct = (predictions == targets).sum().item()
            
            # Update statistics
            total_loss += loss.item() * self.gradient_manager.accumulation_steps
            total_correct += correct
            total_samples += targets.size(0)
            batch_losses.append(loss.item() * self.gradient_manager.accumulation_steps)
            
            # Log progress
            if batch_idx % self.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                recent_grad_norm = self.gradient_stats[-1]['grad_norm'] if self.gradient_stats else 0.0
                
                self.logger.info(
                    f'Epoch {self.current_epoch}, Batch {batch_idx}/{len(self.train_loader)}: '
                    f'Loss: {loss.item() * self.gradient_manager.accumulation_steps:.6f}, '
                    f'Accuracy: {100. * correct / targets.size(0):.2f}%, '
                    f'LR: {current_lr:.6f}, '
                    f'Grad Norm: {recent_grad_norm:.4f}'
                )
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * total_correct / total_samples
        epoch_time = time.time() - start_time
        
        # Store learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.learning_rates.append(current_lr)
        
        # Get gradient statistics for this epoch
        epoch_grad_stats = self.gradient_manager.get_statistics()
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'time': epoch_time,
            'learning_rate': current_lr,
            'samples_processed': total_samples,
            'gradient_stats': epoch_grad_stats
        }
        
        self.train_losses.append(avg_loss)
        
        return metrics
    
    def train(
        self,
        epochs: int,
        early_stopping_patience: int = None,
        save_best: bool = True,
        save_every: int = None,
        config_name: str = None
    ) -> Dict[str, Any]:
        """
        Train the model with advanced optimization strategies.
        
        Args:
            epochs: Number of epochs to train
            early_stopping_patience: Patience for early stopping (overrides convergence analyzer)
            save_best: Whether to save the best model
            save_every: Save checkpoint every N epochs
            config_name: Name for this configuration (for comparison mode)
            
        Returns:
            Dictionary containing comprehensive training results
        """
        # Override convergence analyzer patience if specified
        if early_stopping_patience is not None:
            self.convergence_analyzer.patience = early_stopping_patience
        
        # Create scheduler now that we know the total epochs and steps
        steps_per_epoch = len(self.train_loader)
        self.scheduler = SchedulerFactory.create_scheduler(
            self.optimizer,
            total_epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            **self.scheduler_config
        )
        
        self.logger.info(f"Starting advanced training for {epochs} epochs")
        self.logger.info(f"Optimizer: {self.optimizer_config['optimizer_type']}")
        self.logger.info(f"Scheduler: {self.scheduler_config.get('scheduler_type', 'none')}")
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
                if hasattr(self.scheduler, 'step'):
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['loss'])
                    else:
                        self.scheduler.step()
            
            # Convergence analysis
            should_stop = self.convergence_analyzer.update(
                current_value=val_metrics['loss'],
                epoch=self.current_epoch,
                model_state=self.model.state_dict()
            )
            
            # Store convergence history
            convergence_info = {
                'epoch': self.current_epoch,
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy'],
                'is_best': val_metrics['loss'] < self.best_val_loss,
                'wait': self.convergence_analyzer.wait,
                'should_stop': should_stop
            }
            self.convergence_history.append(convergence_info)
            
            # Log epoch results with enhanced information
            self.logger.info(
                f"Epoch {self.current_epoch}/{epochs} - "
                f"Train Loss: {train_metrics['loss']:.6f}, "
                f"Train Acc: {train_metrics['accuracy']:.2f}%, "
                f"Val Loss: {val_metrics['loss']:.6f}, "
                f"Val Acc: {val_metrics['accuracy']:.2f}%, "
                f"Time: {train_metrics['time']:.2f}s, "
                f"Wait: {self.convergence_analyzer.wait}/{self.convergence_analyzer.patience}"
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
                            'val_accuracy': val_metrics['accuracy'],
                            'gradient_stats': train_metrics['gradient_stats']
                        }
                    )
            
            # Regular checkpoint saving
            if save_every and self.current_epoch % save_every == 0:
                self.save_checkpoint(
                    filename=f'checkpoint_epoch_{self.current_epoch}.pth',
                    metrics={
                        'train_loss': train_metrics['loss'],
                        'val_loss': val_metrics['loss'],
                        'val_accuracy': val_metrics['accuracy'],
                        'gradient_stats': train_metrics['gradient_stats']
                    }
                )
            
            # Early stopping check
            if should_stop:
                self.logger.info(f"Early stopping triggered at epoch {self.current_epoch}")
                
                # Restore best weights if configured
                if (self.convergence_analyzer.restore_best_weights and 
                    self.convergence_analyzer.best_weights is not None):
                    self.model.load_state_dict(self.convergence_analyzer.best_weights)
                    self.logger.info("Restored best model weights")
                
                break
        
        total_training_time = time.time() - training_start_time
        
        # Get comprehensive analysis
        convergence_analysis = self.convergence_analyzer.get_convergence_analysis()
        gradient_analysis = self.gradient_manager.get_statistics()
        
        self.logger.info(f"Training completed in {total_training_time:.2f} seconds")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.6f}")
        self.logger.info(f"Best validation accuracy: {self.best_val_accuracy:.2f}%")
        self.logger.info(f"Convergence analysis: {convergence_analysis}")
        
        # Prepare comprehensive results
        results = {
            'training_history': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'val_accuracies': self.val_accuracies,
                'learning_rates': self.learning_rates,
                'convergence_history': self.convergence_history
            },
            'final_metrics': {
                'best_val_loss': self.best_val_loss,
                'best_val_accuracy': self.best_val_accuracy,
                'final_train_loss': self.train_losses[-1] if self.train_losses else None,
                'final_val_loss': self.val_losses[-1] if self.val_losses else None,
                'final_val_accuracy': self.val_accuracies[-1] if self.val_accuracies else None,
                'total_epochs': self.current_epoch,
                'total_time': total_training_time
            },
            'convergence_analysis': convergence_analysis,
            'gradient_analysis': gradient_analysis,
            'configuration': {
                'optimizer_config': self.optimizer_config,
                'scheduler_config': self.scheduler_config,
                'gradient_config': {
                    'clip_type': self.gradient_manager.clip_type,
                    'clip_value': self.gradient_manager.clip_value,
                    'accumulation_steps': self.gradient_manager.accumulation_steps
                },
                'convergence_config': {
                    'patience': self.convergence_analyzer.patience,
                    'min_delta': self.convergence_analyzer.min_delta,
                    'monitor_metric': self.convergence_analyzer.monitor_metric,
                    'mode': self.convergence_analyzer.mode
                }
            }
        }
        
        # Add to optimizer comparison if in comparison mode
        if self.comparison_mode and config_name and self.optimizer_comparison:
            self.optimizer_comparison.add_result(
                config_name=config_name,
                optimizer_type=self.optimizer_config['optimizer_type'],
                scheduler_type=self.scheduler_config.get('scheduler_type', 'none'),
                final_metrics=results['final_metrics'],
                training_history=results['training_history'],
                convergence_analysis=convergence_analysis,
                hyperparameters=results['configuration']
            )
        
        return results
    
    def save_checkpoint(
        self,
        filename: str = None,
        is_best: bool = False,
        metrics: Dict[str, Any] = None
    ) -> str:
        """
        Save enhanced checkpoint with advanced training information.
        
        Args:
            filename: Checkpoint filename
            is_best: Whether this is the best model
            metrics: Current metrics to save
            
        Returns:
            Path to saved checkpoint
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'advanced_checkpoint_epoch_{self.current_epoch}_{timestamp}.pth'
        
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        # Prepare enhanced checkpoint data
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': self.model.get_model_info(),
            'training_history': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'val_accuracies': self.val_accuracies,
                'learning_rates': self.learning_rates,
                'convergence_history': self.convergence_history
            },
            'best_metrics': {
                'best_val_loss': self.best_val_loss,
                'best_val_accuracy': self.best_val_accuracy
            },
            'advanced_config': {
                'optimizer_config': self.optimizer_config,
                'scheduler_config': self.scheduler_config,
                'gradient_config': {
                    'clip_type': self.gradient_manager.clip_type,
                    'clip_value': self.gradient_manager.clip_value,
                    'accumulation_steps': self.gradient_manager.accumulation_steps
                }
            },
            'convergence_analysis': self.convergence_analyzer.get_convergence_analysis(),
            'gradient_analysis': self.gradient_manager.get_statistics(),
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
        
        self.logger.info(f"Advanced checkpoint saved: {filepath}")
        
        # Save best model copy
        if is_best:
            best_filepath = os.path.join(self.checkpoint_dir, 'best_advanced_model.pth')
            torch.save(checkpoint, best_filepath)
            self.logger.info(f"Best advanced model saved: {best_filepath}")
        
        return filepath
    
    def get_optimizer_comparison_report(self) -> Optional[Dict[str, Any]]:
        """
        Get optimizer comparison report if in comparison mode.
        
        Returns:
            Comparison report or None if not in comparison mode
        """
        if not self.comparison_mode or not self.optimizer_comparison:
            return None
        
        return self.optimizer_comparison.generate_comparison_report()
    
    def plot_advanced_training_analysis(self, save_plots: bool = True) -> Dict[str, str]:
        """
        Generate advanced training analysis plots.
        
        Args:
            save_plots: Whether to save plots to files
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        import matplotlib.pyplot as plt
        
        plot_paths = {}
        
        if not self.train_losses:
            return plot_paths
        
        # Create comprehensive training analysis plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Training and validation loss
        axes[0, 0].plot(self.train_losses, label='Training Loss', color='blue')
        axes[0, 0].plot(self.val_losses, label='Validation Loss', color='red')
        axes[0, 0].set_title('Training Progress')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Validation accuracy
        axes[0, 1].plot(self.val_accuracies, label='Validation Accuracy', color='green')
        axes[0, 1].set_title('Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate schedule
        axes[0, 2].plot(self.learning_rates, label='Learning Rate', color='orange')
        axes[0, 2].set_title('Learning Rate Schedule')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Learning Rate')
        axes[0, 2].set_yscale('log')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # Gradient norms
        if self.gradient_stats:
            grad_norms = [stat['grad_norm'] for stat in self.gradient_stats]
            axes[1, 0].plot(grad_norms, alpha=0.7, color='purple')
            axes[1, 0].set_title('Gradient Norms')
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].set_ylabel('Gradient Norm')
            axes[1, 0].grid(True)
        
        # Convergence analysis
        if self.convergence_history:
            wait_times = [conv['wait'] for conv in self.convergence_history]
            axes[1, 1].plot(wait_times, label='Early Stopping Wait', color='brown')
            axes[1, 1].axhline(y=self.convergence_analyzer.patience, 
                              color='red', linestyle='--', label='Patience Limit')
            axes[1, 1].set_title('Early Stopping Analysis')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Wait Count')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        # Training efficiency
        if len(self.train_losses) > 1:
            loss_improvements = np.diff(self.train_losses)
            axes[1, 2].plot(loss_improvements, label='Loss Improvement', color='teal')
            axes[1, 2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[1, 2].set_title('Training Efficiency')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Loss Change')
            axes[1, 2].legend()
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = os.path.join(self.checkpoint_dir, 'advanced_training_analysis.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plot_paths['training_analysis'] = plot_path
            self.logger.info(f"Advanced training analysis plot saved to: {plot_path}")
        
        plt.show()
        
        return plot_paths


def create_advanced_trainer(
    model: LSTMClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer_type: str = 'adamw',
    scheduler_type: str = 'cosine_warm_restarts',
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    gradient_clip_type: str = 'norm',
    gradient_clip_value: float = 1.0,
    accumulation_steps: int = 1,
    early_stopping_patience: int = 10,
    device: str = None,
    checkpoint_dir: str = "checkpoints",
    comparison_mode: bool = False
) -> AdvancedTrainer:
    """
    Factory function to create an advanced trainer with optimized configurations.
    
    Args:
        model: LSTM model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer_type: Type of optimizer to use
        scheduler_type: Type of learning rate scheduler
        learning_rate: Initial learning rate
        weight_decay: Weight decay for regularization
        gradient_clip_type: Type of gradient clipping
        gradient_clip_value: Gradient clipping threshold
        accumulation_steps: Number of gradient accumulation steps
        early_stopping_patience: Patience for early stopping
        device: Device to train on
        checkpoint_dir: Directory for checkpoints
        comparison_mode: Whether to enable optimizer comparison
        
    Returns:
        Configured AdvancedTrainer instance
    """
    optimizer_config = {
        'optimizer_type': optimizer_type,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay
    }
    
    scheduler_config = {
        'scheduler_type': scheduler_type
    }
    
    gradient_config = {
        'clip_type': gradient_clip_type,
        'clip_value': gradient_clip_value,
        'accumulation_steps': accumulation_steps
    }
    
    convergence_config = {
        'patience': early_stopping_patience,
        'min_delta': 1e-4,
        'restore_best_weights': True,
        'monitor_metric': 'val_loss',
        'mode': 'min',
        'warmup_epochs': 3
    }
    
    return AdvancedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
        gradient_config=gradient_config,
        convergence_config=convergence_config,
        device=device,
        checkpoint_dir=checkpoint_dir,
        comparison_mode=comparison_mode
    )