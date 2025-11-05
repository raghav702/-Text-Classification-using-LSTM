"""
Advanced optimization strategies for LSTM Sentiment Classifier.

This module provides enhanced optimization techniques including:
- Multiple optimizer types with comparison framework
- Advanced learning rate schedulers
- Gradient accumulation and clipping strategies
- Training convergence analysis
- Improved early stopping mechanisms
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau, 
    CosineAnnealingLR, 
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    StepLR,
    ExponentialLR
)
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import json
import os


class OptimizerFactory:
    """Factory class for creating different types of optimizers."""
    
    @staticmethod
    def create_optimizer(
        model_parameters,
        optimizer_type: str = 'adam',
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        **kwargs
    ) -> optim.Optimizer:
        """
        Create optimizer based on type specification.
        
        Args:
            model_parameters: Model parameters to optimize
            optimizer_type: Type of optimizer ('adam', 'adamw', 'sgd', 'rmsprop')
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            **kwargs: Additional optimizer-specific parameters
            
        Returns:
            Configured optimizer instance
        """
        optimizer_type = optimizer_type.lower()
        
        if optimizer_type == 'adam':
            return optim.Adam(
                model_parameters,
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=kwargs.get('betas', (0.9, 0.999)),
                eps=kwargs.get('eps', 1e-8)
            )
        
        elif optimizer_type == 'adamw':
            return optim.AdamW(
                model_parameters,
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=kwargs.get('betas', (0.9, 0.999)),
                eps=kwargs.get('eps', 1e-8)
            )
        
        elif optimizer_type == 'sgd':
            return optim.SGD(
                model_parameters,
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=kwargs.get('momentum', 0.9),
                nesterov=kwargs.get('nesterov', True)
            )
        
        elif optimizer_type == 'rmsprop':
            return optim.RMSprop(
                model_parameters,
                lr=learning_rate,
                weight_decay=weight_decay,
                alpha=kwargs.get('alpha', 0.99),
                eps=kwargs.get('eps', 1e-8),
                momentum=kwargs.get('momentum', 0.0)
            )
        
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


class SchedulerFactory:
    """Factory class for creating different types of learning rate schedulers."""
    
    @staticmethod
    def create_scheduler(
        optimizer: optim.Optimizer,
        scheduler_type: str = 'plateau',
        total_epochs: int = None,
        steps_per_epoch: int = None,
        **kwargs
    ) -> Optional[optim.lr_scheduler._LRScheduler]:
        """
        Create learning rate scheduler based on type specification.
        
        Args:
            optimizer: Optimizer to schedule
            scheduler_type: Type of scheduler
            total_epochs: Total number of training epochs
            steps_per_epoch: Number of steps per epoch (for OneCycleLR)
            **kwargs: Additional scheduler-specific parameters
            
        Returns:
            Configured scheduler instance or None
        """
        scheduler_type = scheduler_type.lower()
        
        if scheduler_type == 'none' or scheduler_type is None:
            return None
        
        elif scheduler_type == 'plateau':
            return ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=kwargs.get('factor', 0.5),
                patience=kwargs.get('patience', 3),
                threshold=kwargs.get('threshold', 1e-4),
                min_lr=kwargs.get('min_lr', 1e-7),
                verbose=kwargs.get('verbose', True)
            )
        
        elif scheduler_type == 'cosine':
            if total_epochs is None:
                raise ValueError("total_epochs required for CosineAnnealingLR")
            return CosineAnnealingLR(
                optimizer,
                T_max=total_epochs,
                eta_min=kwargs.get('eta_min', 1e-6)
            )
        
        elif scheduler_type == 'cosine_warm_restarts':
            return CosineAnnealingWarmRestarts(
                optimizer,
                T_0=kwargs.get('T_0', 10),
                T_mult=kwargs.get('T_mult', 2),
                eta_min=kwargs.get('eta_min', 1e-6)
            )
        
        elif scheduler_type == 'onecycle':
            if total_epochs is None or steps_per_epoch is None:
                raise ValueError("total_epochs and steps_per_epoch required for OneCycleLR")
            return OneCycleLR(
                optimizer,
                max_lr=kwargs.get('max_lr', optimizer.param_groups[0]['lr'] * 10),
                total_steps=total_epochs * steps_per_epoch,
                pct_start=kwargs.get('pct_start', 0.3),
                anneal_strategy=kwargs.get('anneal_strategy', 'cos')
            )
        
        elif scheduler_type == 'step':
            return StepLR(
                optimizer,
                step_size=kwargs.get('step_size', 10),
                gamma=kwargs.get('gamma', 0.1)
            )
        
        elif scheduler_type == 'exponential':
            return ExponentialLR(
                optimizer,
                gamma=kwargs.get('gamma', 0.95)
            )
        
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


class GradientManager:
    """Manages gradient clipping and accumulation strategies."""
    
    def __init__(
        self,
        clip_type: str = 'norm',
        clip_value: float = 1.0,
        accumulation_steps: int = 1
    ):
        """
        Initialize gradient manager.
        
        Args:
            clip_type: Type of gradient clipping ('norm', 'value', 'none')
            clip_value: Clipping threshold
            accumulation_steps: Number of steps to accumulate gradients
        """
        self.clip_type = clip_type.lower()
        self.clip_value = clip_value
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
        
        # Gradient statistics
        self.grad_norms = []
        self.clipped_steps = 0
        
    def clip_gradients(self, model: nn.Module) -> float:
        """
        Apply gradient clipping to model parameters.
        
        Args:
            model: Model to clip gradients for
            
        Returns:
            Gradient norm before clipping
        """
        if self.clip_type == 'none':
            return 0.0
        
        # Calculate gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        self.grad_norms.append(total_norm)
        
        # Apply clipping
        if self.clip_type == 'norm' and total_norm > self.clip_value:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_value)
            self.clipped_steps += 1
        elif self.clip_type == 'value':
            torch.nn.utils.clip_grad_value_(model.parameters(), self.clip_value)
            if total_norm > self.clip_value:
                self.clipped_steps += 1
        
        return total_norm
    
    def should_step(self) -> bool:
        """
        Check if optimizer should step based on accumulation strategy.
        
        Returns:
            True if optimizer should step
        """
        self.current_step += 1
        return self.current_step % self.accumulation_steps == 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get gradient statistics.
        
        Returns:
            Dictionary containing gradient statistics
        """
        if not self.grad_norms:
            return {}
        
        return {
            'mean_grad_norm': np.mean(self.grad_norms),
            'max_grad_norm': np.max(self.grad_norms),
            'min_grad_norm': np.min(self.grad_norms),
            'std_grad_norm': np.std(self.grad_norms),
            'clipped_ratio': self.clipped_steps / len(self.grad_norms),
            'total_steps': len(self.grad_norms)
        }


class ConvergenceAnalyzer:
    """Analyzes training convergence and provides early stopping improvements."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        restore_best_weights: bool = True,
        monitor_metric: str = 'val_loss',
        mode: str = 'min',
        warmup_epochs: int = 5
    ):
        """
        Initialize convergence analyzer.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights on early stop
            monitor_metric: Metric to monitor for early stopping
            mode: 'min' or 'max' for the monitored metric
            warmup_epochs: Number of epochs before early stopping can trigger
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.monitor_metric = monitor_metric
        self.mode = mode
        self.warmup_epochs = warmup_epochs
        
        # State tracking
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = 0
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        
        # Convergence analysis
        self.metric_history = []
        self.improvement_history = []
        self.plateau_lengths = []
        
    def update(self, current_value: float, epoch: int, model_state: Dict = None) -> bool:
        """
        Update convergence analysis with new metric value.
        
        Args:
            current_value: Current value of monitored metric
            epoch: Current epoch number
            model_state: Model state dict to save if best
            
        Returns:
            True if early stopping should be triggered
        """
        self.metric_history.append(current_value)
        
        # Check for improvement
        if self.mode == 'min':
            improved = current_value < (self.best_value - self.min_delta)
        else:
            improved = current_value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = current_value
            self.best_epoch = epoch
            self.wait = 0
            
            if self.restore_best_weights and model_state is not None:
                self.best_weights = model_state.copy()
            
            self.improvement_history.append(epoch)
            
            # Record plateau length if we had one
            if len(self.improvement_history) > 1:
                plateau_length = epoch - self.improvement_history[-2]
                self.plateau_lengths.append(plateau_length)
        else:
            self.wait += 1
        
        # Check early stopping conditions
        should_stop = (
            epoch >= self.warmup_epochs and 
            self.wait >= self.patience
        )
        
        if should_stop:
            self.stopped_epoch = epoch
        
        return should_stop
    
    def get_convergence_analysis(self) -> Dict[str, Any]:
        """
        Get detailed convergence analysis.
        
        Returns:
            Dictionary containing convergence statistics
        """
        if len(self.metric_history) < 2:
            return {}
        
        # Calculate convergence metrics
        recent_window = min(10, len(self.metric_history) // 4)
        recent_values = self.metric_history[-recent_window:]
        
        analysis = {
            'best_value': self.best_value,
            'best_epoch': self.best_epoch,
            'total_epochs': len(self.metric_history),
            'improvement_epochs': len(self.improvement_history),
            'improvement_frequency': len(self.improvement_history) / len(self.metric_history),
            'average_plateau_length': np.mean(self.plateau_lengths) if self.plateau_lengths else 0,
            'recent_trend': self._calculate_trend(recent_values),
            'convergence_rate': self._calculate_convergence_rate(),
            'stability_score': self._calculate_stability_score(recent_values),
            'early_stopped': self.stopped_epoch > 0,
            'stopped_epoch': self.stopped_epoch
        }
        
        return analysis
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for recent values."""
        if len(values) < 3:
            return 'insufficient_data'
        
        # Linear regression to find trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if abs(slope) < self.min_delta / 10:
            return 'stable'
        elif slope < 0:
            return 'improving' if self.mode == 'min' else 'degrading'
        else:
            return 'degrading' if self.mode == 'min' else 'improving'
    
    def _calculate_convergence_rate(self) -> float:
        """Calculate rate of convergence."""
        if len(self.metric_history) < 2:
            return 0.0
        
        initial_value = self.metric_history[0]
        current_value = self.metric_history[-1]
        
        if initial_value == current_value:
            return 0.0
        
        improvement = abs(current_value - initial_value)
        epochs = len(self.metric_history)
        
        return improvement / epochs
    
    def _calculate_stability_score(self, values: List[float]) -> float:
        """Calculate stability score based on variance."""
        if len(values) < 2:
            return 1.0
        
        variance = np.var(values)
        mean_value = np.mean(values)
        
        if mean_value == 0:
            return 1.0 if variance == 0 else 0.0
        
        # Coefficient of variation (lower is more stable)
        cv = np.sqrt(variance) / abs(mean_value)
        
        # Convert to stability score (0-1, higher is more stable)
        return 1.0 / (1.0 + cv)


class OptimizerComparison:
    """Framework for comparing different optimizer configurations."""
    
    def __init__(self, save_dir: str = "optimizer_comparison"):
        """
        Initialize optimizer comparison framework.
        
        Args:
            save_dir: Directory to save comparison results
        """
        self.save_dir = save_dir
        self.results = {}
        self.logger = logging.getLogger(__name__)
        
        os.makedirs(save_dir, exist_ok=True)
    
    def add_result(
        self,
        config_name: str,
        optimizer_type: str,
        scheduler_type: str,
        final_metrics: Dict[str, float],
        training_history: Dict[str, List[float]],
        convergence_analysis: Dict[str, Any],
        hyperparameters: Dict[str, Any]
    ):
        """
        Add optimization result for comparison.
        
        Args:
            config_name: Name for this configuration
            optimizer_type: Type of optimizer used
            scheduler_type: Type of scheduler used
            final_metrics: Final training metrics
            training_history: Complete training history
            convergence_analysis: Convergence analysis results
            hyperparameters: Hyperparameters used
        """
        self.results[config_name] = {
            'optimizer_type': optimizer_type,
            'scheduler_type': scheduler_type,
            'final_metrics': final_metrics,
            'training_history': training_history,
            'convergence_analysis': convergence_analysis,
            'hyperparameters': hyperparameters,
            'timestamp': torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        }
        
        self.logger.info(f"Added optimization result: {config_name}")
    
    def generate_comparison_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive comparison report.
        
        Returns:
            Dictionary containing comparison analysis
        """
        if not self.results:
            return {'error': 'No results to compare'}
        
        # Extract metrics for comparison
        comparison_data = {}
        
        for config_name, result in self.results.items():
            comparison_data[config_name] = {
                'best_val_loss': result['final_metrics'].get('best_val_loss', float('inf')),
                'best_val_accuracy': result['final_metrics'].get('best_val_accuracy', 0.0),
                'convergence_rate': result['convergence_analysis'].get('convergence_rate', 0.0),
                'stability_score': result['convergence_analysis'].get('stability_score', 0.0),
                'total_epochs': result['convergence_analysis'].get('total_epochs', 0),
                'improvement_frequency': result['convergence_analysis'].get('improvement_frequency', 0.0),
                'optimizer_type': result['optimizer_type'],
                'scheduler_type': result['scheduler_type']
            }
        
        # Find best configurations
        best_loss_config = min(comparison_data.keys(), 
                              key=lambda x: comparison_data[x]['best_val_loss'])
        best_accuracy_config = max(comparison_data.keys(), 
                                  key=lambda x: comparison_data[x]['best_val_accuracy'])
        best_convergence_config = max(comparison_data.keys(), 
                                     key=lambda x: comparison_data[x]['convergence_rate'])
        
        # Generate rankings
        rankings = self._generate_rankings(comparison_data)
        
        report = {
            'summary': {
                'total_configurations': len(self.results),
                'best_loss_config': best_loss_config,
                'best_accuracy_config': best_accuracy_config,
                'best_convergence_config': best_convergence_config
            },
            'detailed_comparison': comparison_data,
            'rankings': rankings,
            'recommendations': self._generate_recommendations(comparison_data, rankings)
        }
        
        # Save report
        report_path = os.path.join(self.save_dir, 'comparison_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Comparison report saved to: {report_path}")
        
        return report
    
    def _generate_rankings(self, comparison_data: Dict) -> Dict[str, List[str]]:
        """Generate rankings for different metrics."""
        rankings = {}
        
        # Rank by validation loss (lower is better)
        rankings['by_val_loss'] = sorted(
            comparison_data.keys(),
            key=lambda x: comparison_data[x]['best_val_loss']
        )
        
        # Rank by validation accuracy (higher is better)
        rankings['by_val_accuracy'] = sorted(
            comparison_data.keys(),
            key=lambda x: comparison_data[x]['best_val_accuracy'],
            reverse=True
        )
        
        # Rank by convergence rate (higher is better)
        rankings['by_convergence_rate'] = sorted(
            comparison_data.keys(),
            key=lambda x: comparison_data[x]['convergence_rate'],
            reverse=True
        )
        
        # Rank by stability (higher is better)
        rankings['by_stability'] = sorted(
            comparison_data.keys(),
            key=lambda x: comparison_data[x]['stability_score'],
            reverse=True
        )
        
        return rankings
    
    def _generate_recommendations(
        self, 
        comparison_data: Dict, 
        rankings: Dict
    ) -> Dict[str, str]:
        """Generate recommendations based on comparison results."""
        recommendations = {}
        
        # Best overall performance
        best_overall = rankings['by_val_accuracy'][0]
        recommendations['best_overall'] = (
            f"Use {best_overall} for best overall performance "
            f"(accuracy: {comparison_data[best_overall]['best_val_accuracy']:.2f}%)"
        )
        
        # Fastest convergence
        fastest_convergence = rankings['by_convergence_rate'][0]
        recommendations['fastest_convergence'] = (
            f"Use {fastest_convergence} for fastest convergence "
            f"(rate: {comparison_data[fastest_convergence]['convergence_rate']:.6f})"
        )
        
        # Most stable training
        most_stable = rankings['by_stability'][0]
        recommendations['most_stable'] = (
            f"Use {most_stable} for most stable training "
            f"(stability: {comparison_data[most_stable]['stability_score']:.3f})"
        )
        
        # Optimizer type analysis
        optimizer_performance = {}
        for config, data in comparison_data.items():
            opt_type = data['optimizer_type']
            if opt_type not in optimizer_performance:
                optimizer_performance[opt_type] = []
            optimizer_performance[opt_type].append(data['best_val_accuracy'])
        
        best_optimizer = max(optimizer_performance.keys(),
                           key=lambda x: np.mean(optimizer_performance[x]))
        recommendations['best_optimizer_type'] = (
            f"Use {best_optimizer} optimizer type "
            f"(avg accuracy: {np.mean(optimizer_performance[best_optimizer]):.2f}%)"
        )
        
        return recommendations
    
    def plot_comparison(self, save_plots: bool = True) -> Dict[str, str]:
        """
        Generate comparison plots.
        
        Args:
            save_plots: Whether to save plots to files
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        if not self.results:
            return {}
        
        plot_paths = {}
        
        # Training curves comparison
        plt.figure(figsize=(15, 10))
        
        # Validation loss comparison
        plt.subplot(2, 2, 1)
        for config_name, result in self.results.items():
            val_losses = result['training_history'].get('val_losses', [])
            if val_losses:
                plt.plot(val_losses, label=f"{config_name} ({result['optimizer_type']})")
        plt.title('Validation Loss Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Validation accuracy comparison
        plt.subplot(2, 2, 2)
        for config_name, result in self.results.items():
            val_accuracies = result['training_history'].get('val_accuracies', [])
            if val_accuracies:
                plt.plot(val_accuracies, label=f"{config_name} ({result['optimizer_type']})")
        plt.title('Validation Accuracy Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        # Learning rate comparison
        plt.subplot(2, 2, 3)
        for config_name, result in self.results.items():
            learning_rates = result['training_history'].get('learning_rates', [])
            if learning_rates:
                plt.plot(learning_rates, label=f"{config_name} ({result['scheduler_type']})")
        plt.title('Learning Rate Schedules')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')
        
        # Final metrics comparison
        plt.subplot(2, 2, 4)
        configs = list(self.results.keys())
        accuracies = [self.results[config]['final_metrics'].get('best_val_accuracy', 0) 
                     for config in configs]
        
        bars = plt.bar(range(len(configs)), accuracies)
        plt.title('Final Validation Accuracy Comparison')
        plt.xlabel('Configuration')
        plt.ylabel('Best Validation Accuracy (%)')
        plt.xticks(range(len(configs)), [c[:10] + '...' if len(c) > 10 else c 
                                        for c in configs], rotation=45)
        
        # Color bars by optimizer type
        optimizer_colors = {'adam': 'blue', 'adamw': 'green', 'sgd': 'red', 'rmsprop': 'orange'}
        for i, config in enumerate(configs):
            opt_type = self.results[config]['optimizer_type']
            bars[i].set_color(optimizer_colors.get(opt_type, 'gray'))
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = os.path.join(self.save_dir, 'optimizer_comparison.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plot_paths['comparison'] = plot_path
            self.logger.info(f"Comparison plot saved to: {plot_path}")
        
        plt.show()
        
        return plot_paths