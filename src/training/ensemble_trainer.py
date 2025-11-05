"""
Ensemble Training Pipeline
Utilities for training diverse models for ensemble creation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import random
from pathlib import Path
import json
import time
from sklearn.model_selection import KFold

from src.models.lstm_model import LSTMClassifier
from src.models.attention_lstm import AttentionLSTMClassifier
from src.models.ensemble import ModelEnsemble, MajorityVoting, WeightedVoting, ConfidenceBasedVoting
from src.training.trainer import train_epoch, evaluate_model


class EnsembleTrainer:
    """
    Trainer for creating diverse model ensembles with different configurations and training strategies.
    """
    
    def __init__(
        self,
        base_config: Dict[str, Any],
        device: torch.device = None,
        random_seed: int = 42
    ):
        """
        Initialize ensemble trainer.
        
        Args:
            base_config: Base configuration for model training
            device: Device to train on
            random_seed: Random seed for reproducibility
        """
        self.base_config = base_config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.random_seed = random_seed
        
        # Set random seeds for reproducibility
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
        
        self.trained_models = []
        self.model_configs = []
        self.training_histories = []
        
    def create_diverse_configs(
        self,
        n_models: int = 5,
        diversity_strategies: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Create diverse model configurations for ensemble training.
        
        Args:
            n_models: Number of diverse models to create
            diversity_strategies: List of diversity strategies to apply
            
        Returns:
            List of diverse model configurations
        """
        if diversity_strategies is None:
            diversity_strategies = ['architecture', 'hyperparameters', 'regularization']
        
        diverse_configs = []
        base_config = self.base_config.copy()
        
        for i in range(n_models):
            config = base_config.copy()
            
            # Apply diversity strategies
            if 'architecture' in diversity_strategies:
                config = self._vary_architecture(config, i)
            
            if 'hyperparameters' in diversity_strategies:
                config = self._vary_hyperparameters(config, i)
            
            if 'regularization' in diversity_strategies:
                config = self._vary_regularization(config, i)
            
            # Add model identifier
            config['model_id'] = f'model_{i}'
            config['diversity_seed'] = self.random_seed + i
            
            diverse_configs.append(config)
        
        return diverse_configs
    
    def _vary_architecture(self, config: Dict[str, Any], model_idx: int) -> Dict[str, Any]:
        """Vary model architecture parameters."""
        # Vary hidden dimensions
        hidden_dims = [64, 96, 128, 160, 192]
        config['hidden_dim'] = hidden_dims[model_idx % len(hidden_dims)]
        
        # Vary number of layers
        n_layers_options = [1, 2, 3]
        config['n_layers'] = n_layers_options[model_idx % len(n_layers_options)]
        
        # Vary model type (LSTM vs Attention LSTM)
        if model_idx % 2 == 0:
            config['model_class'] = LSTMClassifier
            config['model_type'] = 'LSTM'
        else:
            config['model_class'] = AttentionLSTMClassifier
            config['model_type'] = 'AttentionLSTM'
            config['attention_dim'] = 64
            config['use_attention_pooling'] = True
        
        return config
    
    def _vary_hyperparameters(self, config: Dict[str, Any], model_idx: int) -> Dict[str, Any]:
        """Vary training hyperparameters."""
        # Vary learning rates
        learning_rates = [0.0005, 0.001, 0.002, 0.003, 0.005]
        config['learning_rate'] = learning_rates[model_idx % len(learning_rates)]
        
        # Vary batch sizes
        batch_sizes = [32, 48, 64, 80, 96]
        config['batch_size'] = batch_sizes[model_idx % len(batch_sizes)]
        
        # Vary optimizers
        optimizers = ['Adam', 'AdamW', 'RMSprop']
        config['optimizer'] = optimizers[model_idx % len(optimizers)]
        
        return config
    
    def _vary_regularization(self, config: Dict[str, Any], model_idx: int) -> Dict[str, Any]:
        """Vary regularization parameters."""
        # Vary dropout rates
        dropout_rates = [0.2, 0.3, 0.4, 0.5]
        config['dropout'] = dropout_rates[model_idx % len(dropout_rates)]
        
        # Vary weight decay
        weight_decays = [0.0, 1e-5, 1e-4, 1e-3]
        config['weight_decay'] = weight_decays[model_idx % len(weight_decays)]
        
        return config
    
    def train_diverse_models(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        diverse_configs: List[Dict[str, Any]],
        epochs: int = 10,
        save_dir: Optional[str] = None
    ) -> List[nn.Module]:
        """
        Train diverse models with different configurations.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            diverse_configs: List of diverse model configurations
            epochs: Number of training epochs
            save_dir: Optional directory to save trained models
            
        Returns:
            List of trained models
        """
        trained_models = []
        
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
        
        for i, config in enumerate(diverse_configs):
            print(f"\nTraining Model {i+1}/{len(diverse_configs)}: {config['model_id']}")
            print(f"Config: {config['model_type']}, Hidden: {config['hidden_dim']}, "
                  f"Layers: {config['n_layers']}, LR: {config['learning_rate']}")
            
            # Create model
            model_class = config.pop('model_class')
            model_config = {k: v for k, v in config.items() 
                          if k in ['vocab_size', 'embedding_dim', 'hidden_dim', 'n_layers', 
                                  'dropout', 'bidirectional', 'attention_dim', 'use_attention_pooling']}
            
            model = model_class(**model_config)
            model.to(self.device)
            
            # Setup optimizer
            optimizer_name = config.get('optimizer', 'Adam')
            learning_rate = config.get('learning_rate', 0.001)
            weight_decay = config.get('weight_decay', 0.0)
            
            if optimizer_name == 'Adam':
                optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            elif optimizer_name == 'AdamW':
                optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            elif optimizer_name == 'RMSprop':
                optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            else:
                optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            
            # Setup loss function
            criterion = nn.BCEWithLogitsLoss()
            
            # Training loop
            train_losses = []
            val_losses = []
            val_accuracies = []
            
            best_val_loss = float('inf')
            best_model_state = None
            
            for epoch in range(epochs):
                # Training
                model.train()
                train_loss = train_epoch(model, train_dataloader, optimizer, criterion, self.device)
                train_losses.append(train_loss)
                
                # Validation
                model.eval()
                val_loss, val_accuracy = evaluate_model(model, val_dataloader, criterion, self.device)
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict().copy()
                
                if (epoch + 1) % 5 == 0:
                    print(f"  Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            
            # Load best model state
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            
            # Store training history
            history = {
                'model_id': config['model_id'],
                'config': config,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_accuracies': val_accuracies,
                'best_val_loss': best_val_loss,
                'final_val_accuracy': val_accuracies[-1]
            }
            self.training_histories.append(history)
            
            # Save model if directory provided
            if save_dir:
                model_path = save_path / f"{config['model_id']}.pth"
                torch.save(model.state_dict(), model_path)
                
                # Save config
                config_path = save_path / f"{config['model_id']}_config.json"
                with open(config_path, 'w') as f:
                    # Convert non-serializable items
                    serializable_config = config.copy()
                    serializable_config.pop('model_class', None)
                    json.dump(serializable_config, f, indent=2)
            
            trained_models.append(model)
            self.trained_models.append(model)
            self.model_configs.append(config)
            
            print(f"  Final validation accuracy: {val_accuracies[-1]:.4f}")
        
        return trained_models
    
    def train_with_cross_validation(
        self,
        dataset,
        diverse_configs: List[Dict[str, Any]],
        n_folds: int = 5,
        epochs: int = 10,
        save_dir: Optional[str] = None
    ) -> List[List[nn.Module]]:
        """
        Train diverse models using cross-validation for robust ensemble creation.
        
        Args:
            dataset: Full dataset for cross-validation
            diverse_configs: List of diverse model configurations
            n_folds: Number of cross-validation folds
            epochs: Number of training epochs per fold
            save_dir: Optional directory to save models
            
        Returns:
            List of lists containing trained models for each fold
        """
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_seed)
        fold_models = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            print(f"\n{'='*50}")
            print(f"Cross-Validation Fold {fold+1}/{n_folds}")
            print(f"{'='*50}")
            
            # Create data loaders for this fold
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)
            
            train_dataloader = DataLoader(
                dataset, 
                batch_size=self.base_config.get('batch_size', 64),
                sampler=train_sampler
            )
            val_dataloader = DataLoader(
                dataset,
                batch_size=self.base_config.get('batch_size', 64),
                sampler=val_sampler
            )
            
            # Train models for this fold
            fold_save_dir = None
            if save_dir:
                fold_save_dir = Path(save_dir) / f"fold_{fold}"
            
            fold_trained_models = self.train_diverse_models(
                train_dataloader, val_dataloader, diverse_configs, epochs, fold_save_dir
            )
            
            fold_models.append(fold_trained_models)
        
        return fold_models
    
    def create_ensemble_from_trained_models(
        self,
        trained_models: List[nn.Module],
        strategy_type: str = 'majority',
        strategy_params: Optional[Dict] = None
    ) -> ModelEnsemble:
        """
        Create ensemble from trained models.
        
        Args:
            trained_models: List of trained models
            strategy_type: Type of ensemble strategy ('majority', 'weighted', 'confidence')
            strategy_params: Optional parameters for the strategy
            
        Returns:
            ModelEnsemble instance
        """
        # Create ensemble strategy
        if strategy_type == 'majority':
            strategy = MajorityVoting()
        elif strategy_type == 'weighted':
            weights = strategy_params.get('weights') if strategy_params else None
            if weights is None:
                # Equal weights if not provided
                weights = [1.0 / len(trained_models)] * len(trained_models)
            strategy = WeightedVoting(weights)
        elif strategy_type == 'confidence':
            threshold = strategy_params.get('confidence_threshold', 0.1) if strategy_params else 0.1
            strategy = ConfidenceBasedVoting(threshold)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        # Create model names
        model_names = [f"Model_{i}" for i in range(len(trained_models))]
        if hasattr(self, 'model_configs') and self.model_configs:
            model_names = [config.get('model_id', f'Model_{i}') for i, config in enumerate(self.model_configs)]
        
        ensemble = ModelEnsemble(
            models=trained_models,
            strategy=strategy,
            model_names=model_names,
            device=self.device
        )
        
        return ensemble
    
    def optimize_ensemble_weights(
        self,
        trained_models: List[nn.Module],
        val_dataloader: DataLoader,
        method: str = 'accuracy'
    ) -> List[float]:
        """
        Optimize ensemble weights based on individual model performance.
        
        Args:
            trained_models: List of trained models
            val_dataloader: Validation data loader
            method: Optimization method ('accuracy', 'loss', 'f1')
            
        Returns:
            List of optimized weights
        """
        model_scores = []
        
        # Evaluate individual models
        for model in trained_models:
            model.eval()
            if method == 'accuracy':
                _, accuracy = evaluate_model(model, val_dataloader, nn.BCEWithLogitsLoss(), self.device)
                model_scores.append(accuracy)
            elif method == 'loss':
                loss, _ = evaluate_model(model, val_dataloader, nn.BCEWithLogitsLoss(), self.device)
                model_scores.append(1.0 / (1.0 + loss))  # Convert loss to score (higher is better)
            # Add more methods as needed
        
        # Normalize scores to weights
        total_score = sum(model_scores)
        weights = [score / total_score for score in model_scores]
        
        print(f"Optimized weights based on {method}: {weights}")
        return weights
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get summary of training results.
        
        Returns:
            Dictionary with training summary statistics
        """
        if not self.training_histories:
            return {"message": "No training history available"}
        
        summary = {
            'n_models_trained': len(self.training_histories),
            'best_model': None,
            'worst_model': None,
            'average_performance': {},
            'model_details': []
        }
        
        # Find best and worst models
        best_accuracy = 0
        worst_accuracy = 1
        
        accuracies = []
        val_losses = []
        
        for history in self.training_histories:
            final_accuracy = history['final_val_accuracy']
            final_loss = history['best_val_loss']
            
            accuracies.append(final_accuracy)
            val_losses.append(final_loss)
            
            if final_accuracy > best_accuracy:
                best_accuracy = final_accuracy
                summary['best_model'] = {
                    'model_id': history['model_id'],
                    'accuracy': final_accuracy,
                    'config': history['config']
                }
            
            if final_accuracy < worst_accuracy:
                worst_accuracy = final_accuracy
                summary['worst_model'] = {
                    'model_id': history['model_id'],
                    'accuracy': final_accuracy,
                    'config': history['config']
                }
            
            summary['model_details'].append({
                'model_id': history['model_id'],
                'model_type': history['config'].get('model_type', 'Unknown'),
                'final_accuracy': final_accuracy,
                'best_val_loss': final_loss,
                'hidden_dim': history['config'].get('hidden_dim'),
                'n_layers': history['config'].get('n_layers'),
                'learning_rate': history['config'].get('learning_rate')
            })
        
        # Calculate average performance
        summary['average_performance'] = {
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'mean_val_loss': np.mean(val_losses),
            'std_val_loss': np.std(val_losses)
        }
        
        return summary