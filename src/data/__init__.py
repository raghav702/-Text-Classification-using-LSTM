"""
Data processing module for LSTM sentiment classifier.

This module provides text preprocessing, dataset loading, PyTorch Dataset/DataLoader
functionality, text augmentation, and advanced preprocessing capabilities for the 
IMDB movie review sentiment classification task.
"""

from .text_preprocessor import TextPreprocessor
from .imdb_loader import IMDBLoader, load_imdb_data
from .dataset import (
    IMDBDataset, 
    IMDBDatasetFromDataFrame, 
    IMDBDataLoaderManager, 
    create_imdb_dataloaders,
    collate_fn
)
from .text_augmentation import (
    TextAugmenter,
    BackTranslationAugmenter,
    create_balanced_dataset
)
from .advanced_preprocessor import (
    AdvancedTextPreprocessor,
    create_preprocessing_pipeline,
    evaluate_preprocessing_pipeline
)

__all__ = [
    'TextPreprocessor',
    'IMDBLoader',
    'load_imdb_data',
    'IMDBDataset',
    'IMDBDatasetFromDataFrame',
    'IMDBDataLoaderManager',
    'create_imdb_dataloaders',
    'collate_fn',
    'TextAugmenter',
    'BackTranslationAugmenter',
    'create_balanced_dataset',
    'AdvancedTextPreprocessor',
    'create_preprocessing_pipeline',
    'evaluate_preprocessing_pipeline'
]