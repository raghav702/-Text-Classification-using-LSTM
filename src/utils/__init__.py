"""
Utility modules for LSTM sentiment classifier.
"""

from src.utils.embedding_utils import EmbeddingManager, load_embeddings_for_model, analyze_embedding_coverage
from src.utils.augmentation_utils import (
    AugmentationManager,
    load_augmentation_manager,
    apply_augmentation_pipeline
)

__all__ = [
    'EmbeddingManager',
    'load_embeddings_for_model', 
    'analyze_embedding_coverage',
    'AugmentationManager',
    'load_augmentation_manager',
    'apply_augmentation_pipeline'
]