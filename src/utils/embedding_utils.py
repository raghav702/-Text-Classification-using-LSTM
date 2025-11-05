"""
Embedding utilities for LSTM sentiment classifier.

This module provides utilities for loading, processing, and managing
pre-trained word embeddings (GloVe) for the sentiment classification model.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import pickle
import json
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from download_glove import GloVeEmbeddingProcessor


class EmbeddingManager:
    """
    Manager class for handling pre-trained embeddings in the LSTM model.
    
    Provides functionality to load, align, and manage embeddings with
    model vocabulary, including caching and analysis capabilities.
    """
    
    def __init__(self, cache_dir: str = "data/embeddings"):
        """
        Initialize the embedding manager.
        
        Args:
            cache_dir: Directory to cache processed embeddings
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.glove_processor = GloVeEmbeddingProcessor()
        self.embedding_matrix = None
        self.alignment_stats = None
        
    def prepare_embeddings_for_model(
        self,
        vocabulary: Dict[str, int],
        embedding_dim: int = 300,
        init_strategy: str = 'random',
        use_cache: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, any]]:
        """
        Prepare embedding matrix for model initialization.
        
        Args:
            vocabulary: Model vocabulary (word_to_idx mapping)
            embedding_dim: Dimension of embeddings
            init_strategy: Strategy for OOV words ('random', 'zero', 'mean')
            use_cache: Whether to use cached embeddings if available
            
        Returns:
            Tuple of (embedding_matrix, alignment_stats)
        """
        cache_key = self._get_cache_key(vocabulary, embedding_dim, init_strategy)
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        
        # Try to load from cache
        if use_cache and cache_path.exists():
            print(f"Loading cached embeddings from {cache_path}")
            return self._load_cached_embeddings(cache_path)
        
        # Download GloVe embeddings if needed
        if not self.glove_processor._check_embeddings_exist():
            print("GloVe embeddings not found. Downloading...")
            if not self.glove_processor.download_glove_embeddings():
                raise RuntimeError("Failed to download GloVe embeddings")
        
        # Align embeddings with vocabulary
        embedding_matrix, alignment_stats = self.glove_processor.align_embeddings_with_vocabulary(
            vocabulary, embedding_dim, init_strategy
        )
        
        # Cache the results
        if use_cache:
            self._cache_embeddings(cache_path, embedding_matrix, alignment_stats)
        
        self.embedding_matrix = embedding_matrix
        self.alignment_stats = alignment_stats
        
        return embedding_matrix, alignment_stats
    
    def analyze_vocabulary_coverage(
        self,
        vocabulary: Dict[str, int],
        embedding_dim: int = 300,
        save_analysis: bool = True
    ) -> Dict[str, any]:
        """
        Analyze embedding coverage for the given vocabulary.
        
        Args:
            vocabulary: Model vocabulary
            embedding_dim: Dimension of embeddings to analyze
            save_analysis: Whether to save analysis results
            
        Returns:
            Coverage analysis results
        """
        coverage_stats = self.glove_processor.analyze_embedding_coverage(
            vocabulary, embedding_dim
        )
        
        if save_analysis:
            analysis_path = self.cache_dir / f"coverage_analysis_{embedding_dim}d.json"
            with open(analysis_path, 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                serializable_stats = self._make_json_serializable(coverage_stats)
                json.dump(serializable_stats, f, indent=2)
            print(f"Coverage analysis saved to {analysis_path}")
        
        return coverage_stats
    
    def create_embedding_visualization(
        self,
        vocabulary: Dict[str, int],
        embedding_dim: int = 300,
        save_path: Optional[str] = None
    ):
        """
        Create and save embedding coverage visualization.
        
        Args:
            vocabulary: Model vocabulary
            embedding_dim: Dimension of embeddings
            save_path: Path to save visualization (auto-generated if None)
        """
        coverage_stats = self.analyze_vocabulary_coverage(vocabulary, embedding_dim, save_analysis=False)
        
        if save_path is None:
            save_path = self.cache_dir / f"embedding_coverage_{embedding_dim}d.png"
        
        self.glove_processor.visualize_embedding_coverage(coverage_stats, str(save_path))
    
    def get_embedding_quality_metrics(
        self,
        vocabulary: Dict[str, int],
        embedding_dim: int = 300,
        sample_words: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Calculate embedding quality metrics.
        
        Args:
            vocabulary: Model vocabulary
            embedding_dim: Dimension of embeddings
            sample_words: Sample words for similarity analysis
            
        Returns:
            Dictionary with quality metrics
        """
        if sample_words is None:
            # Use common sentiment words for analysis
            sample_words = ['good', 'bad', 'great', 'terrible', 'amazing', 'awful', 
                          'excellent', 'horrible', 'fantastic', 'disappointing']
        
        # Filter words that exist in vocabulary
        vocab_sample_words = [word for word in sample_words if word in vocabulary]
        
        if not vocab_sample_words:
            return {'error': 'No sample words found in vocabulary'}
        
        # Analyze word similarities
        similarity_results = self.glove_processor.create_embedding_similarity_analysis(
            vocab_sample_words, embedding_dim, top_k=5
        )
        
        # Calculate coverage for sample words
        glove_embeddings = self.glove_processor.load_glove_embeddings(embedding_dim)
        sample_coverage = sum(1 for word in vocab_sample_words if word in glove_embeddings)
        sample_coverage_rate = sample_coverage / len(vocab_sample_words)
        
        quality_metrics = {
            'sample_words_analyzed': len(vocab_sample_words),
            'sample_coverage_rate': sample_coverage_rate,
            'similarity_analysis': similarity_results,
            'embedding_dimension': embedding_dim
        }
        
        return quality_metrics
    
    def _get_cache_key(self, vocabulary: Dict[str, int], embedding_dim: int, init_strategy: str) -> str:
        """Generate cache key for embedding matrix."""
        vocab_hash = hash(frozenset(vocabulary.items()))
        return f"embeddings_{abs(vocab_hash)}_{embedding_dim}d_{init_strategy}"
    
    def _cache_embeddings(self, cache_path: Path, embedding_matrix: torch.Tensor, alignment_stats: Dict):
        """Cache embedding matrix and alignment statistics."""
        cache_data = {
            'embedding_matrix': embedding_matrix,
            'alignment_stats': alignment_stats
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"Embeddings cached to {cache_path}")
    
    def _load_cached_embeddings(self, cache_path: Path) -> Tuple[torch.Tensor, Dict]:
        """Load cached embedding matrix and alignment statistics."""
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        return cache_data['embedding_matrix'], cache_data['alignment_stats']
    
    def _make_json_serializable(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj


def load_embeddings_for_model(
    vocabulary: Dict[str, int],
    embedding_dim: int = 300,
    init_strategy: str = 'random',
    cache_dir: str = "data/embeddings"
) -> Tuple[torch.Tensor, Dict[str, any]]:
    """
    Convenience function to load embeddings for model initialization.
    
    Args:
        vocabulary: Model vocabulary (word_to_idx mapping)
        embedding_dim: Dimension of embeddings
        init_strategy: Strategy for OOV words
        cache_dir: Directory for caching embeddings
        
    Returns:
        Tuple of (embedding_matrix, alignment_stats)
    """
    manager = EmbeddingManager(cache_dir)
    return manager.prepare_embeddings_for_model(
        vocabulary, embedding_dim, init_strategy, use_cache=True
    )


def analyze_embedding_coverage(
    vocabulary: Dict[str, int],
    embedding_dim: int = 300,
    cache_dir: str = "data/embeddings"
) -> Dict[str, any]:
    """
    Convenience function to analyze embedding coverage.
    
    Args:
        vocabulary: Model vocabulary
        embedding_dim: Dimension of embeddings
        cache_dir: Directory for caching results
        
    Returns:
        Coverage analysis results
    """
    manager = EmbeddingManager(cache_dir)
    return manager.analyze_vocabulary_coverage(vocabulary, embedding_dim)