"""
Augmentation utilities for integrating text augmentation and advanced preprocessing
into the training pipeline.
"""

import yaml
import os
from typing import List, Dict, Tuple, Any, Optional
import pandas as pd
import numpy as np

from ..data import (
    TextAugmenter,
    BackTranslationAugmenter,
    AdvancedTextPreprocessor,
    create_balanced_dataset,
    create_preprocessing_pipeline,
    evaluate_preprocessing_pipeline
)


class AugmentationManager:
    """
    Manager class for handling text augmentation and advanced preprocessing
    in the training pipeline.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the AugmentationManager.
        
        Args:
            config_path: Path to augmentation configuration file
        """
        self.config = self._load_config(config_path)
        self.text_augmenter = None
        self.back_translator = None
        self.preprocessor = None
        
        self._initialize_components()
    
    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), '..', '..', 'configs', 'augmentation_config.yaml'
            )
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            print(f"Warning: Config file {config_path} not found. Using default settings.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'text_augmentation': {
                'synonym_prob': 0.15,
                'insert_prob': 0.05,
                'delete_prob': 0.05,
                'swap_prob': 0.10,
                'max_augmentations': 1,
                'preserve_length': True,
                'min_similarity': 0.7,
                'augmentation_factor': 0.3,
                'noise_prob': 0.02
            },
            'back_translation': {
                'enabled': False,
                'intermediate_languages': ['es', 'fr', 'de'],
                'num_augmentations': 1
            },
            'advanced_preprocessing': {
                'tokenizer_type': 'advanced',
                'use_lemmatization': True,
                'use_stemming': False,
                'remove_stopwords': False,
                'handle_negation': True,
                'preserve_entities': True,
                'max_vocab_size': 10000,
                'min_freq': 2,
                'max_length': 500
            },
            'data_balancing': {
                'enabled': True,
                'target_balance': 0.5,
                'method': 'augmentation'
            }
        }
    
    def _initialize_components(self):
        """Initialize augmentation and preprocessing components."""
        # Initialize text augmenter
        aug_config = self.config.get('text_augmentation', {})
        self.text_augmenter = TextAugmenter(
            synonym_prob=aug_config.get('synonym_prob', 0.15),
            insert_prob=aug_config.get('insert_prob', 0.05),
            delete_prob=aug_config.get('delete_prob', 0.05),
            swap_prob=aug_config.get('swap_prob', 0.10),
            max_augmentations=aug_config.get('max_augmentations', 1),
            preserve_length=aug_config.get('preserve_length', True)
        )
        
        # Initialize back-translator if enabled
        bt_config = self.config.get('back_translation', {})
        if bt_config.get('enabled', False):
            self.back_translator = BackTranslationAugmenter(
                intermediate_languages=bt_config.get('intermediate_languages', ['es', 'fr'])
            )
        
        # Initialize advanced preprocessor
        prep_config = self.config.get('advanced_preprocessing', {})
        self.preprocessor = create_preprocessing_pipeline(prep_config)
    
    def augment_dataset(self, texts: List[str], labels: List[int]) -> Tuple[List[str], List[int]]:
        """
        Apply augmentation to the dataset.
        
        Args:
            texts: List of text strings
            labels: List of corresponding labels
            
        Returns:
            Tuple of (augmented_texts, augmented_labels)
        """
        augmented_texts = texts.copy()
        augmented_labels = labels.copy()
        
        # Apply text augmentation
        aug_config = self.config.get('text_augmentation', {})
        augmentation_factor = aug_config.get('augmentation_factor', 0.3)
        
        if augmentation_factor > 0:
            aug_texts, aug_labels = self.text_augmenter.augment_dataset(
                texts, labels, augmentation_factor
            )
            augmented_texts = aug_texts
            augmented_labels = aug_labels
        
        # Apply back-translation if enabled
        bt_config = self.config.get('back_translation', {})
        if bt_config.get('enabled', False) and self.back_translator:
            bt_augmented_texts = []
            bt_augmented_labels = []
            
            num_to_augment = int(len(texts) * 0.1)  # Augment 10% with back-translation
            indices = np.random.choice(len(texts), num_to_augment, replace=False)
            
            for idx in indices:
                bt_versions = self.back_translator.augment_text(
                    texts[idx], bt_config.get('num_augmentations', 1)
                )
                bt_augmented_texts.extend(bt_versions)
                bt_augmented_labels.extend([labels[idx]] * len(bt_versions))
            
            augmented_texts.extend(bt_augmented_texts)
            augmented_labels.extend(bt_augmented_labels)
        
        # Apply data balancing if enabled
        balance_config = self.config.get('data_balancing', {})
        if balance_config.get('enabled', True):
            target_balance = balance_config.get('target_balance', 0.5)
            augmented_texts, augmented_labels = create_balanced_dataset(
                augmented_texts, augmented_labels, target_balance
            )
        
        return augmented_texts, augmented_labels
    
    def preprocess_texts(self, texts: List[str], fit_vocabulary: bool = False) -> Any:
        """
        Apply advanced preprocessing to texts.
        
        Args:
            texts: List of text strings
            fit_vocabulary: Whether to build vocabulary from these texts
            
        Returns:
            Preprocessed tensor sequences
        """
        return self.preprocessor.preprocess_texts(texts, fit_vocabulary)
    
    def get_preprocessor(self) -> AdvancedTextPreprocessor:
        """Get the configured preprocessor."""
        return self.preprocessor
    
    def evaluate_preprocessing(self, texts: List[str], labels: List[int]) -> Dict[str, Any]:
        """
        Evaluate preprocessing pipeline effectiveness.
        
        Args:
            texts: List of texts
            labels: List of corresponding labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        return evaluate_preprocessing_pipeline(self.preprocessor, texts, labels)
    
    def compare_preprocessing_methods(self, texts: List[str]) -> Dict[str, Any]:
        """
        Compare different preprocessing methods.
        
        Args:
            texts: List of texts for comparison
            
        Returns:
            Dictionary comparing different methods
        """
        return self.preprocessor.compare_preprocessing_methods(texts)
    
    def get_augmentation_stats(self, original_texts: List[str], 
                             augmented_texts: List[str]) -> Dict[str, Any]:
        """
        Get statistics about the augmentation process.
        
        Args:
            original_texts: Original text list
            augmented_texts: Augmented text list
            
        Returns:
            Dictionary of augmentation statistics
        """
        stats = {
            'original_count': len(original_texts),
            'augmented_count': len(augmented_texts),
            'augmentation_ratio': len(augmented_texts) / len(original_texts) if original_texts else 0,
            'new_samples': len(augmented_texts) - len(original_texts)
        }
        
        # Analyze text length changes
        original_lengths = [len(text.split()) for text in original_texts]
        augmented_lengths = [len(text.split()) for text in augmented_texts]
        
        stats['original_avg_length'] = np.mean(original_lengths) if original_lengths else 0
        stats['augmented_avg_length'] = np.mean(augmented_lengths) if augmented_lengths else 0
        
        # Quality assessment on a sample
        if len(original_texts) > 0 and len(augmented_texts) > len(original_texts):
            sample_size = min(100, len(original_texts))
            quality_scores = []
            
            for i in range(sample_size):
                if i < len(original_texts):
                    # Find corresponding augmented version (simplified)
                    aug_idx = len(original_texts) + i
                    if aug_idx < len(augmented_texts):
                        quality = self.text_augmenter.validate_augmentation_quality(
                            original_texts[i], augmented_texts[aug_idx]
                        )
                        quality_scores.append(1.0 if quality else 0.0)
            
            stats['avg_quality_score'] = np.mean(quality_scores) if quality_scores else 0
        
        return stats
    
    def save_augmented_dataset(self, texts: List[str], labels: List[int], 
                             output_path: str, format: str = 'csv'):
        """
        Save augmented dataset to file.
        
        Args:
            texts: Augmented texts
            labels: Corresponding labels
            output_path: Output file path
            format: Output format ('csv', 'json', 'txt')
        """
        if format == 'csv':
            df = pd.DataFrame({
                'text': texts,
                'label': labels
            })
            df.to_csv(output_path, index=False)
        
        elif format == 'json':
            import json
            data = [{'text': text, 'label': label} for text, label in zip(texts, labels)]
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        elif format == 'txt':
            with open(output_path, 'w') as f:
                for text, label in zip(texts, labels):
                    f.write(f"{label}\t{text}\n")
        
        print(f"Augmented dataset saved to {output_path}")
    
    def create_preprocessing_report(self, texts: List[str], labels: List[int], 
                                  output_path: str = None) -> Dict[str, Any]:
        """
        Create a comprehensive preprocessing report.
        
        Args:
            texts: List of texts
            labels: List of labels
            output_path: Optional path to save report
            
        Returns:
            Dictionary containing the report
        """
        report = {
            'dataset_info': {
                'total_samples': len(texts),
                'positive_samples': sum(labels),
                'negative_samples': len(labels) - sum(labels),
                'class_balance': sum(labels) / len(labels) if labels else 0
            },
            'preprocessing_evaluation': self.evaluate_preprocessing(texts, labels),
            'method_comparison': self.compare_preprocessing_methods(texts[:1000]),  # Sample for speed
            'preprocessing_stats': self.preprocessor.get_preprocessing_stats(texts[:1000]),
            'configuration': self.config
        }
        
        if output_path:
            import json
            with open(output_path, 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                def convert_numpy(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return obj
                
                def clean_for_json(data):
                    if isinstance(data, dict):
                        return {k: clean_for_json(v) for k, v in data.items()}
                    elif isinstance(data, list):
                        return [clean_for_json(item) for item in data]
                    else:
                        return convert_numpy(data)
                
                json.dump(clean_for_json(report), f, indent=2)
            
            print(f"Preprocessing report saved to {output_path}")
        
        return report


def load_augmentation_manager(config_path: str = None) -> AugmentationManager:
    """
    Load and initialize an AugmentationManager.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Initialized AugmentationManager instance
    """
    return AugmentationManager(config_path)


def apply_augmentation_pipeline(texts: List[str], labels: List[int],
                              config_path: str = None) -> Tuple[List[str], List[int], Dict[str, Any]]:
    """
    Apply the complete augmentation pipeline to a dataset.
    
    Args:
        texts: List of text strings
        labels: List of corresponding labels
        config_path: Path to configuration file
        
    Returns:
        Tuple of (augmented_texts, augmented_labels, stats)
    """
    manager = load_augmentation_manager(config_path)
    
    # Apply augmentation
    augmented_texts, augmented_labels = manager.augment_dataset(texts, labels)
    
    # Get statistics
    stats = manager.get_augmentation_stats(texts, augmented_texts)
    
    return augmented_texts, augmented_labels, stats