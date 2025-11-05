"""
Model factory for creating LSTM sentiment classifiers with pre-trained embeddings.

This module provides factory functions to create and configure LSTM models
with various embedding strategies and configurations.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any
from pathlib import Path

from src.models.lstm_model import LSTMClassifier
from src.utils.embedding_utils import EmbeddingManager, load_embeddings_for_model


class ModelFactory:
    """
    Factory class for creating LSTM sentiment classifiers with various configurations.
    """
    
    def __init__(self, cache_dir: str = "data/embeddings"):
        """
        Initialize the model factory.
        
        Args:
            cache_dir: Directory for caching embeddings
        """
        self.cache_dir = cache_dir
        self.embedding_manager = EmbeddingManager(cache_dir)
    
    def create_lstm_classifier(
        self,
        vocabulary: Dict[str, int],
        embedding_dim: int = 300,
        hidden_dim: int = 128,
        output_dim: int = 1,
        n_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        use_pretrained_embeddings: bool = True,
        embedding_strategy: str = 'fine_tune',
        oov_init_strategy: str = 'random'
    ) -> Tuple[LSTMClassifier, Dict[str, Any]]:
        """
        Create an LSTM classifier with optional pre-trained embeddings.
        
        Args:
            vocabulary: Model vocabulary (word_to_idx mapping)
            embedding_dim: Dimension of word embeddings
            hidden_dim: Hidden dimension of LSTM layers
            output_dim: Output dimension (1 for binary classification)
            n_layers: Number of LSTM layers
            dropout: Dropout rate for regularization
            bidirectional: Whether to use bidirectional LSTM
            use_pretrained_embeddings: Whether to use GloVe embeddings
            embedding_strategy: 'freeze', 'fine_tune', or 'random'
            oov_init_strategy: Strategy for OOV words ('random', 'zero', 'mean')
            
        Returns:
            Tuple of (model, model_info)
        """
        vocab_size = len(vocabulary)
        pad_idx = vocabulary.get('<PAD>', 0)
        
        # Create the model
        model = LSTMClassifier(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            pad_idx=pad_idx
        )
        
        model_info = {
            'vocab_size': vocab_size,
            'embedding_dim': embedding_dim,
            'hidden_dim': hidden_dim,
            'output_dim': output_dim,
            'n_layers': n_layers,
            'dropout': dropout,
            'bidirectional': bidirectional,
            'use_pretrained_embeddings': use_pretrained_embeddings,
            'embedding_strategy': embedding_strategy,
            'oov_init_strategy': oov_init_strategy
        }
        
        # Load pre-trained embeddings if requested
        if use_pretrained_embeddings:
            print("Loading pre-trained GloVe embeddings...")
            
            embedding_matrix, alignment_stats = load_embeddings_for_model(
                vocabulary=vocabulary,
                embedding_dim=embedding_dim,
                init_strategy=oov_init_strategy,
                cache_dir=self.cache_dir
            )
            
            # Determine freezing strategy
            freeze_embeddings = (embedding_strategy == 'freeze')
            fine_tune_embeddings = (embedding_strategy == 'fine_tune')
            
            # Load embeddings into model
            model.load_pretrained_embeddings(
                embedding_matrix,
                freeze_embeddings=freeze_embeddings,
                fine_tune_embeddings=fine_tune_embeddings
            )
            
            # Add alignment stats to model info
            model_info['alignment_stats'] = alignment_stats
            model_info['embedding_coverage'] = alignment_stats['coverage']
            
            print(f"Embedding strategy: {embedding_strategy}")
            print(f"Embedding coverage: {alignment_stats['coverage']:.2%}")
        
        else:
            print("Using randomly initialized embeddings")
            model_info['alignment_stats'] = None
            model_info['embedding_coverage'] = 0.0
        
        # Get model statistics
        model_stats = model.get_model_info()
        model_info.update(model_stats)
        
        return model, model_info
    
    def create_model_with_analysis(
        self,
        vocabulary: Dict[str, int],
        embedding_dim: int = 300,
        **kwargs
    ) -> Tuple[LSTMClassifier, Dict[str, Any]]:
        """
        Create model with comprehensive embedding analysis.
        
        Args:
            vocabulary: Model vocabulary
            embedding_dim: Dimension of embeddings
            **kwargs: Additional arguments for create_lstm_classifier
            
        Returns:
            Tuple of (model, comprehensive_info)
        """
        # Create the model
        model, model_info = self.create_lstm_classifier(
            vocabulary=vocabulary,
            embedding_dim=embedding_dim,
            **kwargs
        )
        
        # Perform embedding analysis
        if kwargs.get('use_pretrained_embeddings', True):
            print("Performing embedding coverage analysis...")
            
            coverage_stats = self.embedding_manager.analyze_vocabulary_coverage(
                vocabulary, embedding_dim, save_analysis=True
            )
            
            quality_metrics = self.embedding_manager.get_embedding_quality_metrics(
                vocabulary, embedding_dim
            )
            
            # Add analysis results to model info
            model_info['coverage_analysis'] = coverage_stats
            model_info['quality_metrics'] = quality_metrics
            
            # Create visualization
            try:
                self.embedding_manager.create_embedding_visualization(
                    vocabulary, embedding_dim
                )
                model_info['visualization_created'] = True
            except Exception as e:
                print(f"Warning: Could not create visualization: {e}")
                model_info['visualization_created'] = False
        
        return model, model_info
    
    def compare_embedding_strategies(
        self,
        vocabulary: Dict[str, int],
        embedding_dim: int = 300,
        strategies: Optional[list] = None,
        **kwargs
    ) -> Dict[str, Tuple[LSTMClassifier, Dict[str, Any]]]:
        """
        Create models with different embedding strategies for comparison.
        
        Args:
            vocabulary: Model vocabulary
            embedding_dim: Dimension of embeddings
            strategies: List of strategies to compare
            **kwargs: Additional arguments for model creation
            
        Returns:
            Dictionary mapping strategy names to (model, info) tuples
        """
        if strategies is None:
            strategies = ['random', 'freeze', 'fine_tune']
        
        models = {}
        
        for strategy in strategies:
            print(f"\nCreating model with {strategy} embedding strategy...")
            
            if strategy == 'random':
                use_pretrained = False
                embedding_strategy = 'fine_tune'
            else:
                use_pretrained = True
                embedding_strategy = strategy
            
            model, info = self.create_lstm_classifier(
                vocabulary=vocabulary,
                embedding_dim=embedding_dim,
                use_pretrained_embeddings=use_pretrained,
                embedding_strategy=embedding_strategy,
                **kwargs
            )
            
            models[strategy] = (model, info)
        
        return models
    
    def save_model_config(self, model_info: Dict[str, Any], save_path: str):
        """
        Save model configuration to file.
        
        Args:
            model_info: Model information dictionary
            save_path: Path to save configuration
        """
        import json
        
        # Make config JSON serializable
        config = {}
        for key, value in model_info.items():
            if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
                config[key] = value
            else:
                config[key] = str(value)
        
        with open(save_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Model configuration saved to {save_path}")


def create_sentiment_classifier(
    vocabulary: Dict[str, int],
    embedding_dim: int = 300,
    hidden_dim: int = 128,
    use_pretrained_embeddings: bool = True,
    embedding_strategy: str = 'fine_tune',
    cache_dir: str = "data/embeddings"
) -> Tuple[LSTMClassifier, Dict[str, Any]]:
    """
    Convenience function to create a sentiment classifier.
    
    Args:
        vocabulary: Model vocabulary (word_to_idx mapping)
        embedding_dim: Dimension of word embeddings
        hidden_dim: Hidden dimension of LSTM layers
        use_pretrained_embeddings: Whether to use GloVe embeddings
        embedding_strategy: 'freeze', 'fine_tune', or 'random'
        cache_dir: Directory for caching embeddings
        
    Returns:
        Tuple of (model, model_info)
    """
    factory = ModelFactory(cache_dir)
    return factory.create_lstm_classifier(
        vocabulary=vocabulary,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        use_pretrained_embeddings=use_pretrained_embeddings,
        embedding_strategy=embedding_strategy
    )


def analyze_embedding_quality(
    vocabulary: Dict[str, int],
    embedding_dim: int = 300,
    cache_dir: str = "data/embeddings"
) -> Dict[str, Any]:
    """
    Convenience function to analyze embedding quality for vocabulary.
    
    Args:
        vocabulary: Model vocabulary
        embedding_dim: Dimension of embeddings
        cache_dir: Directory for caching
        
    Returns:
        Dictionary with quality analysis results
    """
    factory = ModelFactory(cache_dir)
    return factory.embedding_manager.get_embedding_quality_metrics(
        vocabulary, embedding_dim
    )