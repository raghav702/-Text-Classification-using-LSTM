"""
Hyperparameter optimization framework for LSTM Sentiment Classifier.

This module provides comprehensive hyperparameter tuning capabilities including:
- Grid search and random search
- Bayesian optimization with Gaussian Process
- Cross-validation pipeline for robust evaluation
- Automated hyperparameter logging and comparison
- Parallel execution support
"""

import os
import json
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold, StratifiedKFold

# Optional Bayesian optimization dependencies
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    from skopt.acquisition import gaussian_ei
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    print("Warning: scikit-optimize not available. Bayesian optimization disabled.")

from models.lstm_model import LSTMClassifier
from training.advanced_trainer import create_advanced_trainer


class HyperparameterSpace:
    """Defines the hyperparameter search space."""
    
    def __init__(self):
        """Initialize hyperparameter space definitions."""
        self.spaces = {}
        self.parameter_types = {}
        
    def add_parameter(
        self, 
        name: str, 
        param_type: str, 
        values: Union[List, Tuple], 
        log_scale: bool = False
    ):
        """
        Add a parameter to the search space.
        
        Args:
            name: Parameter name
            param_type: Type of parameter ('categorical', 'integer', 'real')
            values: Values or range for the parameter
            log_scale: Whether to use log scale for continuous parameters
        """
        self.parameter_types[name] = param_type
        
        if param_type == 'categorical':
            self.spaces[name] = values
        elif param_type == 'integer':
            self.spaces[name] = (int(values[0]), int(values[1]))
        elif param_type == 'real':
            self.spaces[name] = (float(values[0]), float(values[1]))
        else:
            raise ValueError(f"Unsupported parameter type: {param_type}")
        
        # Store log scale information
        if log_scale and param_type in ['integer', 'real']:
            self.parameter_types[f"{name}_log_scale"] = True
    
    def get_skopt_space(self):
        """Get scikit-optimize compatible space definition."""
        if not BAYESIAN_AVAILABLE:
            raise ImportError("scikit-optimize required for Bayesian optimization")
        
        skopt_space = []
        for name, space in self.spaces.items():
            param_type = self.parameter_types[name]
            log_scale = self.parameter_types.get(f"{name}_log_scale", False)
            
            if param_type == 'categorical':
                skopt_space.append(Categorical(space, name=name))
            elif param_type == 'integer':
                skopt_space.append(Integer(space[0], space[1], name=name, prior='log-uniform' if log_scale else 'uniform'))
            elif param_type == 'real':
                skopt_space.append(Real(space[0], space[1], name=name, prior='log-uniform' if log_scale else 'uniform'))
        
        return skopt_space
    
    def sample_random(self, n_samples: int = 1) -> List[Dict[str, Any]]:
        """
        Sample random hyperparameter configurations.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            List of hyperparameter dictionaries
        """
        samples = []
        
        for _ in range(n_samples):
            sample = {}
            for name, space in self.spaces.items():
                param_type = self.parameter_types[name]
                log_scale = self.parameter_types.get(f"{name}_log_scale", False)
                
                if param_type == 'categorical':
                    sample[name] = np.random.choice(space)
                elif param_type == 'integer':
                    if log_scale:
                        sample[name] = int(np.random.lognormal(
                            np.log(space[0]), 
                            np.log(space[1] / space[0]) / 3
                        ))
                        sample[name] = max(space[0], min(space[1], sample[name]))
                    else:
                        sample[name] = np.random.randint(space[0], space[1] + 1)
                elif param_type == 'real':
                    if log_scale:
                        sample[name] = np.random.lognormal(
                            np.log(space[0]), 
                            np.log(space[1] / space[0]) / 3
                        )
                        sample[name] = max(space[0], min(space[1], sample[name]))
                    else:
                        sample[name] = np.random.uniform(space[0], space[1])
            
            samples.append(sample)
        
        return samples
    
    def get_grid_combinations(self) -> List[Dict[str, Any]]:
        """
        Get all combinations for grid search.
        
        Returns:
            List of all hyperparameter combinations
        """
        # Convert continuous spaces to discrete for grid search
        grid_spaces = {}
        
        for name, space in self.spaces.items():
            param_type = self.parameter_types[name]
            
            if param_type == 'categorical':
                grid_spaces[name] = space
            elif param_type == 'integer':
                # Create reasonable grid for integer parameters
                if space[1] - space[0] <= 10:
                    grid_spaces[name] = list(range(space[0], space[1] + 1))
                else:
                    grid_spaces[name] = [space[0], (space[0] + space[1]) // 2, space[1]]
            elif param_type == 'real':
                # Create reasonable grid for real parameters
                log_scale = self.parameter_types.get(f"{name}_log_scale", False)
                if log_scale:
                    grid_spaces[name] = np.logspace(
                        np.log10(space[0]), np.log10(space[1]), 3
                    ).tolist()
                else:
                    grid_spaces[name] = np.linspace(space[0], space[1], 3).tolist()
        
        # Generate all combinations
        keys = list(grid_spaces.keys())
        values = list(grid_spaces.values())
        combinations = []
        
        for combination in product(*values):
            combinations.append(dict(zip(keys, combination)))
        
        return combinations


class CrossValidator:
    """Cross-validation framework for robust hyperparameter evaluation."""
    
    def __init__(
        self,
        n_folds: int = 5,
        stratified: bool = True,
        shuffle: bool = True,
        random_state: int = 42
    ):
        """
        Initialize cross-validator.
        
        Args:
            n_folds: Number of cross-validation folds
            stratified: Whether to use stratified k-fold
            shuffle: Whether to shuffle data before splitting
            random_state: Random state for reproducibility
        """
        self.n_folds = n_folds
        self.stratified = stratified
        self.shuffle = shuffle
        self.random_state = random_state
        
        if stratified:
            self.cv = StratifiedKFold(
                n_splits=n_folds,
                shuffle=shuffle,
                random_state=random_state
            )
        else:
            self.cv = KFold(
                n_splits=n_folds,
                shuffle=shuffle,
                random_state=random_state
            )
        
        self.logger = logging.getLogger(__name__)
    
    def split_data(
        self, 
        dataset: torch.utils.data.Dataset, 
        labels: torch.Tensor
    ) -> List[Tuple[DataLoader, DataLoader]]:
        """
        Split dataset into cross-validation folds.
        
        Args:
            dataset: Full dataset
            labels: Labels for stratification
            
        Returns:
            List of (train_loader, val_loader) tuples for each fold
        """
        # Convert labels to numpy for sklearn compatibility
        labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else labels
        
        folds = []
        for fold_idx, (train_indices, val_indices) in enumerate(self.cv.split(range(len(dataset)), labels_np)):
            # Create subset datasets
            train_subset = Subset(dataset, train_indices)
            val_subset = Subset(dataset, val_indices)
            
            # Create data loaders
            train_loader = DataLoader(
                train_subset,
                batch_size=64,  # Default batch size, can be overridden
                shuffle=True,
                num_workers=2,
                pin_memory=torch.cuda.is_available()
            )
            
            val_loader = DataLoader(
                val_subset,
                batch_size=64,
                shuffle=False,
                num_workers=2,
                pin_memory=torch.cuda.is_available()
            )
            
            folds.append((train_loader, val_loader))
            
            self.logger.debug(f"Fold {fold_idx + 1}: {len(train_indices)} train, {len(val_indices)} val samples")
        
        return folds
    
    def evaluate_hyperparameters(
        self,
        hyperparams: Dict[str, Any],
        model_factory: Callable,
        dataset: torch.utils.data.Dataset,
        labels: torch.Tensor,
        training_config: Dict[str, Any],
        device: str = 'cpu'
    ) -> Dict[str, Any]:
        """
        Evaluate hyperparameters using cross-validation.
        
        Args:
            hyperparams: Hyperparameter configuration
            model_factory: Function to create model with hyperparams
            dataset: Full dataset
            labels: Labels for the dataset
            training_config: Training configuration
            device: Device to use for training
            
        Returns:
            Cross-validation results
        """
        self.logger.info(f"Evaluating hyperparameters: {hyperparams}")
        
        # Split data into folds
        folds = self.split_data(dataset, labels)
        
        fold_results = []
        fold_times = []
        
        for fold_idx, (train_loader, val_loader) in enumerate(folds):
            self.logger.info(f"Training fold {fold_idx + 1}/{self.n_folds}")
            
            fold_start_time = time.time()
            
            try:
                # Create model with hyperparameters
                model = model_factory(hyperparams)
                
                # Update batch size in data loaders if specified
                if 'batch_size' in hyperparams:
                    train_loader = DataLoader(
                        train_loader.dataset,
                        batch_size=hyperparams['batch_size'],
                        shuffle=True,
                        num_workers=train_loader.num_workers,
                        pin_memory=train_loader.pin_memory
                    )
                    val_loader = DataLoader(
                        val_loader.dataset,
                        batch_size=hyperparams['batch_size'],
                        shuffle=False,
                        num_workers=val_loader.num_workers,
                        pin_memory=val_loader.pin_memory
                    )
                
                # Create trainer with hyperparameters
                trainer_config = training_config.copy()
                trainer_config.update(hyperparams)
                
                trainer = create_advanced_trainer(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    **trainer_config
                )
                
                # Train model
                results = trainer.train(
                    epochs=training_config.get('epochs', 10),
                    save_best=False,  # Don't save during CV
                    save_every=None
                )
                
                fold_time = time.time() - fold_start_time
                fold_times.append(fold_time)
                
                # Extract key metrics
                fold_result = {
                    'fold': fold_idx + 1,
                    'best_val_loss': results['final_metrics']['best_val_loss'],
                    'best_val_accuracy': results['final_metrics']['best_val_accuracy'],
                    'final_val_loss': results['final_metrics']['final_val_loss'],
                    'final_val_accuracy': results['final_metrics']['final_val_accuracy'],
                    'total_epochs': results['final_metrics']['total_epochs'],
                    'training_time': fold_time,
                    'convergence_rate': results['convergence_analysis'].get('convergence_rate', 0.0),
                    'stability_score': results['convergence_analysis'].get('stability_score', 0.0)
                }
                
                fold_results.append(fold_result)
                
                self.logger.info(f"Fold {fold_idx + 1} completed: "
                               f"Val Acc: {fold_result['best_val_accuracy']:.2f}%, "
                               f"Time: {fold_time:.2f}s")
                
            except Exception as e:
                self.logger.error(f"Fold {fold_idx + 1} failed: {e}")
                # Add failed fold result
                fold_result = {
                    'fold': fold_idx + 1,
                    'error': str(e),
                    'best_val_loss': float('inf'),
                    'best_val_accuracy': 0.0,
                    'final_val_loss': float('inf'),
                    'final_val_accuracy': 0.0,
                    'total_epochs': 0,
                    'training_time': 0.0,
                    'convergence_rate': 0.0,
                    'stability_score': 0.0
                }
                fold_results.append(fold_result)
        
        # Calculate cross-validation statistics
        valid_results = [r for r in fold_results if 'error' not in r]
        
        if not valid_results:
            return {
                'hyperparams': hyperparams,
                'cv_score': 0.0,
                'cv_std': float('inf'),
                'fold_results': fold_results,
                'status': 'failed',
                'error': 'All folds failed'
            }
        
        cv_scores = [r['best_val_accuracy'] for r in valid_results]
        cv_losses = [r['best_val_loss'] for r in valid_results]
        
        cv_result = {
            'hyperparams': hyperparams,
            'cv_score': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'cv_loss': np.mean(cv_losses),
            'cv_loss_std': np.std(cv_losses),
            'fold_results': fold_results,
            'n_successful_folds': len(valid_results),
            'total_training_time': sum(fold_times),
            'avg_training_time': np.mean(fold_times) if fold_times else 0.0,
            'status': 'success' if len(valid_results) == self.n_folds else 'partial_success'
        }
        
        self.logger.info(f"CV completed: Score: {cv_result['cv_score']:.2f}% ± {cv_result['cv_std']:.2f}%")
        
        return cv_result


class HyperparameterOptimizer:
    """Main hyperparameter optimization framework."""
    
    def __init__(
        self,
        hyperparameter_space: HyperparameterSpace,
        cross_validator: CrossValidator,
        save_dir: str = "hyperparameter_optimization",
        n_jobs: int = 1
    ):
        """
        Initialize hyperparameter optimizer.
        
        Args:
            hyperparameter_space: Hyperparameter search space
            cross_validator: Cross-validation framework
            save_dir: Directory to save optimization results
            n_jobs: Number of parallel jobs (1 for sequential)
        """
        self.hyperparameter_space = hyperparameter_space
        self.cross_validator = cross_validator
        self.save_dir = save_dir
        self.n_jobs = n_jobs
        
        self.results = []
        self.best_result = None
        
        self.logger = logging.getLogger(__name__)
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
    
    def grid_search(
        self,
        model_factory: Callable,
        dataset: torch.utils.data.Dataset,
        labels: torch.Tensor,
        training_config: Dict[str, Any],
        device: str = 'cpu'
    ) -> Dict[str, Any]:
        """
        Perform grid search optimization.
        
        Args:
            model_factory: Function to create model with hyperparams
            dataset: Full dataset
            labels: Labels for the dataset
            training_config: Training configuration
            device: Device to use for training
            
        Returns:
            Grid search results
        """
        self.logger.info("Starting grid search optimization")
        
        # Get all hyperparameter combinations
        combinations = self.hyperparameter_space.get_grid_combinations()
        
        self.logger.info(f"Grid search will evaluate {len(combinations)} combinations")
        
        return self._evaluate_combinations(
            combinations, model_factory, dataset, labels, training_config, device
        )
    
    def random_search(
        self,
        model_factory: Callable,
        dataset: torch.utils.data.Dataset,
        labels: torch.Tensor,
        training_config: Dict[str, Any],
        n_iter: int = 50,
        device: str = 'cpu'
    ) -> Dict[str, Any]:
        """
        Perform random search optimization.
        
        Args:
            model_factory: Function to create model with hyperparams
            dataset: Full dataset
            labels: Labels for the dataset
            training_config: Training configuration
            n_iter: Number of random samples to evaluate
            device: Device to use for training
            
        Returns:
            Random search results
        """
        self.logger.info(f"Starting random search optimization with {n_iter} iterations")
        
        # Sample random hyperparameter combinations
        combinations = self.hyperparameter_space.sample_random(n_iter)
        
        return self._evaluate_combinations(
            combinations, model_factory, dataset, labels, training_config, device
        )
    
    def bayesian_optimization(
        self,
        model_factory: Callable,
        dataset: torch.utils.data.Dataset,
        labels: torch.Tensor,
        training_config: Dict[str, Any],
        n_calls: int = 50,
        n_initial_points: int = 10,
        device: str = 'cpu'
    ) -> Dict[str, Any]:
        """
        Perform Bayesian optimization.
        
        Args:
            model_factory: Function to create model with hyperparams
            dataset: Full dataset
            labels: Labels for the dataset
            training_config: Training configuration
            n_calls: Number of optimization calls
            n_initial_points: Number of initial random points
            device: Device to use for training
            
        Returns:
            Bayesian optimization results
        """
        if not BAYESIAN_AVAILABLE:
            raise ImportError("scikit-optimize required for Bayesian optimization")
        
        self.logger.info(f"Starting Bayesian optimization with {n_calls} calls")
        
        # Get scikit-optimize compatible space
        space = self.hyperparameter_space.get_skopt_space()
        
        # Define objective function
        @use_named_args(space)
        def objective(**hyperparams):
            """Objective function for Bayesian optimization."""
            try:
                # Evaluate hyperparameters using cross-validation
                cv_result = self.cross_validator.evaluate_hyperparameters(
                    hyperparams, model_factory, dataset, labels, training_config, device
                )
                
                # Store result
                self.results.append(cv_result)
                
                # Update best result
                if self.best_result is None or cv_result['cv_score'] > self.best_result['cv_score']:
                    self.best_result = cv_result
                
                # Return negative score for minimization
                return -cv_result['cv_score']
                
            except Exception as e:
                self.logger.error(f"Objective function failed: {e}")
                return float('inf')  # Return worst possible score
        
        # Run Bayesian optimization
        start_time = time.time()
        
        result = gp_minimize(
            func=objective,
            dimensions=space,
            n_calls=n_calls,
            n_initial_points=n_initial_points,
            acquisition_func=gaussian_ei,
            random_state=42
        )
        
        optimization_time = time.time() - start_time
        
        # Prepare results
        bayesian_result = {
            'method': 'bayesian_optimization',
            'best_score': -result.fun,
            'best_hyperparams': dict(zip([dim.name for dim in space], result.x)),
            'n_evaluations': len(result.func_vals),
            'optimization_time': optimization_time,
            'convergence_trace': [-val for val in result.func_vals],
            'all_results': self.results,
            'best_result': self.best_result
        }
        
        self.logger.info(f"Bayesian optimization completed: Best score: {bayesian_result['best_score']:.2f}%")
        
        # Save results
        self._save_results(bayesian_result)
        
        return bayesian_result
    
    def _evaluate_combinations(
        self,
        combinations: List[Dict[str, Any]],
        model_factory: Callable,
        dataset: torch.utils.data.Dataset,
        labels: torch.Tensor,
        training_config: Dict[str, Any],
        device: str
    ) -> Dict[str, Any]:
        """Evaluate list of hyperparameter combinations."""
        start_time = time.time()
        
        if self.n_jobs == 1:
            # Sequential evaluation
            for i, hyperparams in enumerate(combinations):
                self.logger.info(f"Evaluating combination {i + 1}/{len(combinations)}")
                
                cv_result = self.cross_validator.evaluate_hyperparameters(
                    hyperparams, model_factory, dataset, labels, training_config, device
                )
                
                self.results.append(cv_result)
                
                # Update best result
                if self.best_result is None or cv_result['cv_score'] > self.best_result['cv_score']:
                    self.best_result = cv_result
        else:
            # Parallel evaluation (simplified - would need more complex implementation for full support)
            self.logger.warning("Parallel evaluation not fully implemented, falling back to sequential")
            return self._evaluate_combinations(
                combinations, model_factory, dataset, labels, training_config, device
            )
        
        optimization_time = time.time() - start_time
        
        # Prepare results
        search_result = {
            'method': 'grid_search' if len(combinations) == len(self.hyperparameter_space.get_grid_combinations()) else 'random_search',
            'best_score': self.best_result['cv_score'] if self.best_result else 0.0,
            'best_hyperparams': self.best_result['hyperparams'] if self.best_result else {},
            'n_evaluations': len(self.results),
            'optimization_time': optimization_time,
            'all_results': self.results,
            'best_result': self.best_result
        }
        
        self.logger.info(f"Optimization completed: Best score: {search_result['best_score']:.2f}%")
        
        # Save results
        self._save_results(search_result)
        
        return search_result
    
    def _save_results(self, results: Dict[str, Any]):
        """Save optimization results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_path = os.path.join(self.save_dir, f"optimization_results_{timestamp}.json")
        
        # Prepare serializable results
        serializable_results = results.copy()
        if 'all_results' in serializable_results:
            # Remove non-serializable parts
            for result in serializable_results['all_results']:
                if 'fold_results' in result:
                    for fold_result in result['fold_results']:
                        if 'error' in fold_result and isinstance(fold_result['error'], Exception):
                            fold_result['error'] = str(fold_result['error'])
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        # Save results summary as CSV
        if results['all_results']:
            df_data = []
            for result in results['all_results']:
                row = result['hyperparams'].copy()
                row.update({
                    'cv_score': result['cv_score'],
                    'cv_std': result['cv_std'],
                    'cv_loss': result.get('cv_loss', float('inf')),
                    'n_successful_folds': result.get('n_successful_folds', 0),
                    'total_training_time': result.get('total_training_time', 0.0),
                    'status': result.get('status', 'unknown')
                })
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            csv_path = os.path.join(self.save_dir, f"optimization_summary_{timestamp}.csv")
            df.to_csv(csv_path, index=False)
            
            self.logger.info(f"Results saved to: {results_path}")
            self.logger.info(f"Summary saved to: {csv_path}")
    
    def plot_optimization_results(self, results: Dict[str, Any], save_plots: bool = True) -> Dict[str, str]:
        """
        Generate optimization analysis plots.
        
        Args:
            results: Optimization results
            save_plots: Whether to save plots to files
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        plot_paths = {}
        
        if not results['all_results']:
            return plot_paths
        
        # Create optimization analysis plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract data for plotting
        scores = [r['cv_score'] for r in results['all_results']]
        stds = [r['cv_std'] for r in results['all_results']]
        times = [r.get('total_training_time', 0) for r in results['all_results']]
        
        # Score progression
        axes[0, 0].plot(scores, 'b-', alpha=0.7, label='CV Score')
        axes[0, 0].fill_between(range(len(scores)), 
                               [s - std for s, std in zip(scores, stds)],
                               [s + std for s, std in zip(scores, stds)],
                               alpha=0.3, label='±1 std')
        axes[0, 0].set_title('Cross-Validation Score Progression')
        axes[0, 0].set_xlabel('Evaluation')
        axes[0, 0].set_ylabel('CV Score (%)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Score distribution
        axes[0, 1].hist(scores, bins=20, alpha=0.7, color='green')
        axes[0, 1].axvline(np.mean(scores), color='red', linestyle='--', label=f'Mean: {np.mean(scores):.2f}%')
        axes[0, 1].axvline(np.max(scores), color='orange', linestyle='--', label=f'Best: {np.max(scores):.2f}%')
        axes[0, 1].set_title('CV Score Distribution')
        axes[0, 1].set_xlabel('CV Score (%)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Training time vs performance
        axes[1, 0].scatter(times, scores, alpha=0.6, c=range(len(scores)), cmap='viridis')
        axes[1, 0].set_title('Training Time vs Performance')
        axes[1, 0].set_xlabel('Training Time (seconds)')
        axes[1, 0].set_ylabel('CV Score (%)')
        axes[1, 0].grid(True)
        
        # Hyperparameter importance (if applicable)
        if len(results['all_results']) > 1:
            # Create a simple correlation analysis
            df_data = []
            for result in results['all_results']:
                row = result['hyperparams'].copy()
                row['cv_score'] = result['cv_score']
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            
            # Select numeric columns for correlation
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()['cv_score'].drop('cv_score')
                
                axes[1, 1].barh(range(len(corr_matrix)), corr_matrix.values)
                axes[1, 1].set_yticks(range(len(corr_matrix)))
                axes[1, 1].set_yticklabels(corr_matrix.index)
                axes[1, 1].set_title('Hyperparameter Correlation with CV Score')
                axes[1, 1].set_xlabel('Correlation Coefficient')
                axes[1, 1].grid(True)
            else:
                axes[1, 1].text(0.5, 0.5, 'Insufficient numeric\nhyperparameters\nfor correlation analysis',
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Hyperparameter Analysis')
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(self.save_dir, f"optimization_analysis_{timestamp}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plot_paths['optimization_analysis'] = plot_path
            self.logger.info(f"Optimization analysis plot saved to: {plot_path}")
        
        plt.show()
        
        return plot_paths


def create_default_hyperparameter_space() -> HyperparameterSpace:
    """
    Create a default hyperparameter space for LSTM sentiment classifier.
    
    Returns:
        Configured HyperparameterSpace instance
    """
    space = HyperparameterSpace()
    
    # Model architecture parameters
    space.add_parameter('hidden_dim', 'categorical', [64, 128, 256])
    space.add_parameter('n_layers', 'categorical', [1, 2, 3])
    space.add_parameter('dropout', 'real', (0.1, 0.5))
    space.add_parameter('embedding_dim', 'categorical', [100, 200, 300])
    
    # Training parameters
    space.add_parameter('learning_rate', 'real', (1e-4, 1e-2), log_scale=True)
    space.add_parameter('weight_decay', 'real', (1e-6, 1e-3), log_scale=True)
    space.add_parameter('batch_size', 'categorical', [32, 64, 128])
    
    # Optimization parameters
    space.add_parameter('optimizer_type', 'categorical', ['adam', 'adamw', 'sgd'])
    space.add_parameter('scheduler_type', 'categorical', ['plateau', 'cosine', 'cosine_warm_restarts'])
    
    # Gradient management
    space.add_parameter('gradient_clip_value', 'real', (0.5, 2.0))
    
    return space