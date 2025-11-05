"""
Advanced Model Factory
Factory for creating and configuring advanced model architectures with attention and ensemble capabilities
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Union, Type
from pathlib import Path
import json

from src.models.lstm_model import LSTMClassifier
from src.models.attention_lstm import AttentionLSTMClassifier
from src.models.ensemble import (
    ModelEnsemble, MajorityVoting, WeightedVoting, 
    ConfidenceBasedVoting, create_diverse_ensemble
)


class AdvancedModelFactory:
    """
    Factory class for creating advanced model architectures with easy configuration and experimentation.
    """
    
    # Predefined model configurations
    PRESET_CONFIGS = {
        'small_lstm': {
            'model_type': 'lstm',
            'vocab_size': 10000,
            'embedding_dim': 128,
            'hidden_dim': 64,
            'n_layers': 1,
            'dropout': 0.3,
            'bidirectional': True
        },
        'medium_lstm': {
            'model_type': 'lstm',
            'vocab_size': 10000,
            'embedding_dim': 200,
            'hidden_dim': 128,
            'n_layers': 2,
            'dropout': 0.4,
            'bidirectional': True
        },
        'large_lstm': {
            'model_type': 'lstm',
            'vocab_size': 10000,
            'embedding_dim': 300,
            'hidden_dim': 256,
            'n_layers': 3,
            'dropout': 0.5,
            'bidirectional': True
        },
        'small_attention_lstm': {
            'model_type': 'attention_lstm',
            'vocab_size': 10000,
            'embedding_dim': 128,
            'hidden_dim': 64,
            'attention_dim': 32,
            'n_layers': 1,
            'dropout': 0.3,
            'bidirectional': True,
            'use_attention_pooling': True
        },
        'medium_attention_lstm': {
            'model_type': 'attention_lstm',
            'vocab_size': 10000,
            'embedding_dim': 200,
            'hidden_dim': 128,
            'attention_dim': 64,
            'n_layers': 2,
            'dropout': 0.4,
            'bidirectional': True,
            'use_attention_pooling': True
        },
        'large_attention_lstm': {
            'model_type': 'attention_lstm',
            'vocab_size': 10000,
            'embedding_dim': 300,
            'hidden_dim': 256,
            'attention_dim': 128,
            'n_layers': 3,
            'dropout': 0.5,
            'bidirectional': True,
            'use_attention_pooling': True
        }
    }
    
    # Predefined ensemble configurations
    ENSEMBLE_CONFIGS = {
        'diverse_small': {
            'models': ['small_lstm', 'small_attention_lstm'],
            'strategy': 'majority',
            'strategy_params': {}
        },
        'diverse_medium': {
            'models': ['medium_lstm', 'medium_attention_lstm'],
            'strategy': 'weighted',
            'strategy_params': {'weights': [0.4, 0.6]}
        },
        'diverse_large': {
            'models': ['large_lstm', 'large_attention_lstm'],
            'strategy': 'confidence',
            'strategy_params': {'confidence_threshold': 0.1}
        },
        'multi_scale': {
            'models': ['small_lstm', 'medium_lstm', 'large_lstm'],
            'strategy': 'majority',
            'strategy_params': {}
        },
        'attention_ensemble': {
            'models': ['small_attention_lstm', 'medium_attention_lstm', 'large_attention_lstm'],
            'strategy': 'weighted',
            'strategy_params': {'weights': [0.2, 0.3, 0.5]}
        }
    }
    
    @classmethod
    def create_model(
        cls,
        config: Union[str, Dict[str, Any]],
        device: torch.device = None
    ) -> nn.Module:
        """
        Create a single model from configuration.
        
        Args:
            config: Model configuration (preset name or config dict)
            device: Device to place the model on
            
        Returns:
            Configured model instance
        """
        if isinstance(config, str):
            if config not in cls.PRESET_CONFIGS:
                raise ValueError(f"Unknown preset config: {config}")
            config = cls.PRESET_CONFIGS[config].copy()
        
        model_type = config.pop('model_type', 'lstm')
        
        if model_type == 'lstm':
            model = LSTMClassifier(**config)
        elif model_type == 'attention_lstm':
            model = AttentionLSTMClassifier(**config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        if device is not None:
            model = model.to(device)
        
        return model
    
    @classmethod
    def create_ensemble(
        cls,
        config: Union[str, Dict[str, Any]],
        device: torch.device = None
    ) -> ModelEnsemble:
        """
        Create an ensemble from configuration.
        
        Args:
            config: Ensemble configuration (preset name or config dict)
            device: Device to place the ensemble on
            
        Returns:
            Configured ensemble instance
        """
        if isinstance(config, str):
            if config not in cls.ENSEMBLE_CONFIGS:
                raise ValueError(f"Unknown ensemble preset: {config}")
            config = cls.ENSEMBLE_CONFIGS[config].copy()
        
        # Create individual models
        models = []
        model_names = []
        
        for model_config in config['models']:
            model = cls.create_model(model_config, device)
            models.append(model)
            
            # Generate model name
            if isinstance(model_config, str):
                model_names.append(model_config)
            else:
                model_type = model_config.get('model_type', 'unknown')
                hidden_dim = model_config.get('hidden_dim', 'unknown')
                model_names.append(f"{model_type}_{hidden_dim}")
        
        # Create ensemble strategy
        strategy_type = config.get('strategy', 'majority')
        strategy_params = config.get('strategy_params', {})
        
        if strategy_type == 'majority':
            strategy = MajorityVoting()
        elif strategy_type == 'weighted':
            weights = strategy_params.get('weights')
            if weights is None:
                weights = [1.0 / len(models)] * len(models)
            strategy = WeightedVoting(weights)
        elif strategy_type == 'confidence':
            threshold = strategy_params.get('confidence_threshold', 0.1)
            strategy = ConfidenceBasedVoting(threshold)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        ensemble = ModelEnsemble(
            models=models,
            strategy=strategy,
            model_names=model_names,
            device=device
        )
        
        return ensemble
    
    @classmethod
    def create_custom_diverse_ensemble(
        cls,
        base_config: Dict[str, Any],
        diversity_params: Dict[str, Any],
        n_models: int = 5,
        device: torch.device = None
    ) -> ModelEnsemble:
        """
        Create a custom diverse ensemble with specified diversity parameters.
        
        Args:
            base_config: Base model configuration
            diversity_params: Parameters for creating diversity
            n_models: Number of models in the ensemble
            device: Device to place the ensemble on
            
        Returns:
            Diverse ensemble instance
        """
        diverse_configs = []
        
        for i in range(n_models):
            config = base_config.copy()
            
            # Apply diversity transformations
            if 'hidden_dims' in diversity_params:
                hidden_dims = diversity_params['hidden_dims']
                config['hidden_dim'] = hidden_dims[i % len(hidden_dims)]
            
            if 'n_layers_options' in diversity_params:
                n_layers_options = diversity_params['n_layers_options']
                config['n_layers'] = n_layers_options[i % len(n_layers_options)]
            
            if 'dropout_rates' in diversity_params:
                dropout_rates = diversity_params['dropout_rates']
                config['dropout'] = dropout_rates[i % len(dropout_rates)]
            
            if 'model_types' in diversity_params:
                model_types = diversity_params['model_types']
                config['model_type'] = model_types[i % len(model_types)]
                
                # Add attention-specific parameters if needed
                if config['model_type'] == 'attention_lstm':
                    config['attention_dim'] = config.get('attention_dim', 64)
                    config['use_attention_pooling'] = config.get('use_attention_pooling', True)
            
            diverse_configs.append(config)
        
        # Create models
        models = []
        model_names = []
        
        for i, config in enumerate(diverse_configs):
            model = cls.create_model(config, device)
            models.append(model)
            
            # Generate descriptive name
            model_type = config.get('model_type', 'lstm')
            hidden_dim = config.get('hidden_dim', 'unknown')
            n_layers = config.get('n_layers', 'unknown')
            model_names.append(f"{model_type}_{hidden_dim}h_{n_layers}l")
        
        # Create ensemble with majority voting by default
        ensemble = ModelEnsemble(
            models=models,
            strategy=MajorityVoting(),
            model_names=model_names,
            device=device
        )
        
        return ensemble
    
    @classmethod
    def get_available_presets(cls) -> Dict[str, List[str]]:
        """
        Get available preset configurations.
        
        Returns:
            Dictionary with available model and ensemble presets
        """
        return {
            'models': list(cls.PRESET_CONFIGS.keys()),
            'ensembles': list(cls.ENSEMBLE_CONFIGS.keys())
        }
    
    @classmethod
    def save_config(cls, config: Dict[str, Any], save_path: str):
        """
        Save configuration to file.
        
        Args:
            config: Configuration dictionary
            save_path: Path to save the configuration
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Configuration saved to: {save_path}")
    
    @classmethod
    def load_config(cls, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Loaded configuration dictionary
        """
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return config
    
    @classmethod
    def create_experimental_setup(
        cls,
        experiment_config: Dict[str, Any],
        device: torch.device = None
    ) -> Dict[str, Any]:
        """
        Create a complete experimental setup with multiple models and ensembles.
        
        Args:
            experiment_config: Configuration for the experiment
            device: Device to place models on
            
        Returns:
            Dictionary with created models and ensembles
        """
        setup = {
            'individual_models': {},
            'ensembles': {},
            'config': experiment_config
        }
        
        # Create individual models
        if 'individual_models' in experiment_config:
            for model_name, model_config in experiment_config['individual_models'].items():
                model = cls.create_model(model_config, device)
                setup['individual_models'][model_name] = model
        
        # Create ensembles
        if 'ensembles' in experiment_config:
            for ensemble_name, ensemble_config in experiment_config['ensembles'].items():
                ensemble = cls.create_ensemble(ensemble_config, device)
                setup['ensembles'][ensemble_name] = ensemble
        
        # Create diverse ensembles
        if 'diverse_ensembles' in experiment_config:
            for ensemble_name, diverse_config in experiment_config['diverse_ensembles'].items():
                base_config = diverse_config['base_config']
                diversity_params = diverse_config['diversity_params']
                n_models = diverse_config.get('n_models', 5)
                
                ensemble = cls.create_custom_diverse_ensemble(
                    base_config, diversity_params, n_models, device
                )
                setup['ensembles'][ensemble_name] = ensemble
        
        return setup
    
    @classmethod
    def get_model_summary(cls, model: nn.Module) -> Dict[str, Any]:
        """
        Get summary information about a model.
        
        Args:
            model: Model to summarize
            
        Returns:
            Dictionary with model summary information
        """
        summary = {
            'model_class': model.__class__.__name__,
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
        
        # Add model-specific information
        if hasattr(model, 'get_model_info'):
            summary.update(model.get_model_info())
        
        return summary
    
    @classmethod
    def compare_model_complexities(cls, models: Dict[str, nn.Module]):
        """
        Compare computational complexity of multiple models.
        
        Args:
            models: Dictionary of models to compare
            
        Returns:
            DataFrame with complexity comparison
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for model complexity comparison. Install with: pip install pandas")
        
        comparison_data = []
        
        for model_name, model in models.items():
            summary = cls.get_model_summary(model)
            comparison_data.append({
                'Model': model_name,
                'Type': summary['model_class'],
                'Total Parameters': summary['total_parameters'],
                'Trainable Parameters': summary['trainable_parameters'],
                'Hidden Dim': summary.get('hidden_dim', 'N/A'),
                'Layers': summary.get('n_layers', 'N/A'),
                'Attention': 'Yes' if 'Attention' in summary['model_class'] else 'No'
            })
        
        return pd.DataFrame(comparison_data)


# Convenience functions for quick model creation
def create_quick_lstm(size: str = 'medium', vocab_size: int = 10000) -> LSTMClassifier:
    """
    Quickly create an LSTM model with predefined size.
    
    Args:
        size: Model size ('small', 'medium', 'large')
        vocab_size: Vocabulary size
        
    Returns:
        LSTM model instance
    """
    config = AdvancedModelFactory.PRESET_CONFIGS[f'{size}_lstm'].copy()
    config['vocab_size'] = vocab_size
    return AdvancedModelFactory.create_model(config)


def create_quick_attention_lstm(size: str = 'medium', vocab_size: int = 10000) -> AttentionLSTMClassifier:
    """
    Quickly create an attention LSTM model with predefined size.
    
    Args:
        size: Model size ('small', 'medium', 'large')
        vocab_size: Vocabulary size
        
    Returns:
        Attention LSTM model instance
    """
    config = AdvancedModelFactory.PRESET_CONFIGS[f'{size}_attention_lstm'].copy()
    config['vocab_size'] = vocab_size
    return AdvancedModelFactory.create_model(config)


def create_quick_ensemble(ensemble_type: str = 'diverse_medium', vocab_size: int = 10000) -> ModelEnsemble:
    """
    Quickly create an ensemble with predefined configuration.
    
    Args:
        ensemble_type: Type of ensemble to create
        vocab_size: Vocabulary size
        
    Returns:
        Ensemble instance
    """
    config = AdvancedModelFactory.ENSEMBLE_CONFIGS[ensemble_type].copy()
    
    # Update vocab_size in model configs
    for i, model_config in enumerate(config['models']):
        if isinstance(model_config, str):
            model_preset = AdvancedModelFactory.PRESET_CONFIGS[model_config].copy()
            model_preset['vocab_size'] = vocab_size
            config['models'][i] = model_preset
    
    return AdvancedModelFactory.create_ensemble(config)