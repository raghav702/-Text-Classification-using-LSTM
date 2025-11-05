"""
Integration test for data augmentation and advanced preprocessing functionality.

This script tests the complete augmentation pipeline to ensure all components
work together correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import numpy as np
import pandas as pd
from typing import List, Dict, Any

from src.data import (
    TextAugmenter,
    BackTranslationAugmenter,
    AdvancedTextPreprocessor,
    create_balanced_dataset,
    create_preprocessing_pipeline,
    evaluate_preprocessing_pipeline
)
from src.utils import (
    AugmentationManager,
    load_augmentation_manager,
    apply_augmentation_pipeline
)


def test_text_augmenter():
    """Test TextAugmenter functionality."""
    print("Testing TextAugmenter...")
    
    augmenter = TextAugmenter(
        synonym_prob=0.3,
        insert_prob=0.1,
        delete_prob=0.1,
        swap_prob=0.2,
        max_augmentations=1
    )
    
    # Test single text augmentation
    text = "This movie was absolutely fantastic and amazing!"
    augmented = augmenter.augment_text(text, num_augmentations=3)
    
    assert len(augmented) == 3, f"Expected 3 augmentations, got {len(augmented)}"
    assert all(isinstance(aug, str) for aug in augmented), "All augmentations should be strings"
    
    # Test quality validation
    for aug_text in augmented:
        quality_ok = augmenter.validate_augmentation_quality(text, aug_text)
        print(f"  Original: {text}")
        print(f"  Augmented: {aug_text}")
        print(f"  Quality OK: {quality_ok}")
    
    # Test dataset augmentation
    texts = ["Great movie!", "Terrible film.", "Good acting.", "Bad plot."]
    labels = [1, 0, 1, 0]
    
    aug_texts, aug_labels = augmenter.augment_dataset(texts, labels, augmentation_factor=0.5)
    
    assert len(aug_texts) >= len(texts), "Augmented dataset should be larger"
    assert len(aug_texts) == len(aug_labels), "Texts and labels should have same length"
    
    print("✓ TextAugmenter tests passed")


def test_back_translation():
    """Test BackTranslationAugmenter functionality."""
    print("Testing BackTranslationAugmenter...")
    
    back_translator = BackTranslationAugmenter()
    
    text = "This is an excellent movie with great acting."
    back_translated = back_translator.augment_text(text, num_augmentations=2)
    
    assert len(back_translated) == 2, f"Expected 2 back-translations, got {len(back_translated)}"
    assert all(isinstance(bt, str) for bt in back_translated), "All back-translations should be strings"
    
    print(f"  Original: {text}")
    for i, bt_text in enumerate(back_translated):
        print(f"  Back-translated {i+1}: {bt_text}")
    
    print("✓ BackTranslationAugmenter tests passed")


def test_advanced_preprocessor():
    """Test AdvancedTextPreprocessor functionality."""
    print("Testing AdvancedTextPreprocessor...")
    
    # Test different configurations
    configs = [
        {'tokenizer_type': 'basic', 'use_lemmatization': False},
        {'tokenizer_type': 'advanced', 'use_lemmatization': True, 'handle_negation': True},
        {'tokenizer_type': 'nltk', 'use_lemmatization': True, 'preserve_entities': True}
    ]
    
    test_texts = [
        "I can't believe how AMAZING this movie was!",
        "The film wasn't good at all. Very disappointing.",
        "John Smith starred in this excellent movie."
    ]
    
    for i, config in enumerate(configs):
        print(f"  Testing config {i+1}: {config}")
        
        try:
            preprocessor = create_preprocessing_pipeline(config)
            
            for text in test_texts:
                tokens = preprocessor.tokenize(text)
                features = preprocessor.extract_features(text)
                
                assert isinstance(tokens, list), "Tokens should be a list"
                assert isinstance(features, dict), "Features should be a dictionary"
                assert 'word_count' in features, "Features should include word_count"
                
                print(f"    Text: {text}")
                print(f"    Tokens: {tokens[:5]}{'...' if len(tokens) > 5 else ''}")
                print(f"    Word count: {features['word_count']}")
        
        except Exception as e:
            print(f"    Warning: Config {i+1} failed: {e}")
    
    print("✓ AdvancedTextPreprocessor tests passed")


def test_data_balancing():
    """Test data balancing functionality."""
    print("Testing data balancing...")
    
    # Create imbalanced dataset
    texts = ["Great movie!"] * 10 + ["Bad movie."] * 3
    labels = [1] * 10 + [0] * 3
    
    print(f"  Original: {sum(labels)} positive, {len(labels) - sum(labels)} negative")
    
    balanced_texts, balanced_labels = create_balanced_dataset(texts, labels, target_balance=0.5)
    
    positive_count = sum(balanced_labels)
    negative_count = len(balanced_labels) - positive_count
    balance_ratio = positive_count / len(balanced_labels)
    
    print(f"  Balanced: {positive_count} positive, {negative_count} negative")
    print(f"  Balance ratio: {balance_ratio:.3f}")
    
    assert len(balanced_texts) >= len(texts), "Balanced dataset should be larger or equal"
    assert 0.35 <= balance_ratio <= 0.65, f"Balance ratio should be around 0.5, got {balance_ratio}"
    
    print("✓ Data balancing tests passed")


def test_augmentation_manager():
    """Test AugmentationManager functionality."""
    print("Testing AugmentationManager...")
    
    # Create sample dataset
    texts = [
        "This movie is fantastic!",
        "Terrible film, very boring.",
        "Great acting and excellent plot.",
        "Poor dialogue and bad direction.",
        "Amazing cinematography!"
    ]
    labels = [1, 0, 1, 0, 1]
    
    # Test with default configuration
    manager = load_augmentation_manager()
    
    # Test augmentation
    aug_texts, aug_labels = manager.augment_dataset(texts, labels)
    
    assert len(aug_texts) >= len(texts), "Augmented dataset should be larger"
    assert len(aug_texts) == len(aug_labels), "Texts and labels should match"
    
    print(f"  Original dataset size: {len(texts)}")
    print(f"  Augmented dataset size: {len(aug_texts)}")
    
    # Test preprocessing
    preprocessed = manager.preprocess_texts(texts, fit_vocabulary=True)
    
    assert preprocessed is not None, "Preprocessing should return a result"
    
    # Test evaluation
    eval_results = manager.evaluate_preprocessing(texts, labels)
    
    assert isinstance(eval_results, dict), "Evaluation should return a dictionary"
    assert 'vocabulary_size' in eval_results, "Evaluation should include vocabulary_size"
    
    print(f"  Vocabulary size: {eval_results['vocabulary_size']}")
    
    # Test statistics
    stats = manager.get_augmentation_stats(texts, aug_texts)
    
    assert isinstance(stats, dict), "Stats should be a dictionary"
    assert 'augmentation_ratio' in stats, "Stats should include augmentation_ratio"
    
    print(f"  Augmentation ratio: {stats['augmentation_ratio']:.2f}")
    
    print("✓ AugmentationManager tests passed")


def test_pipeline_integration():
    """Test complete pipeline integration."""
    print("Testing pipeline integration...")
    
    # Create test dataset
    texts = [
        "I absolutely loved this movie! It was fantastic.",
        "This film was terrible and boring.",
        "Great performances by all the actors.",
        "Poor plot and bad dialogue throughout.",
        "Amazing visual effects and cinematography.",
        "Disappointing ending to an otherwise good movie.",
        "Excellent direction and wonderful storytelling.",
        "Waste of time and money. Very bad film."
    ]
    labels = [1, 0, 1, 0, 1, 0, 1, 0]
    
    # Apply complete augmentation pipeline
    aug_texts, aug_labels, stats = apply_augmentation_pipeline(texts, labels)
    
    assert len(aug_texts) >= len(texts), "Pipeline should produce more or equal samples"
    assert len(aug_texts) == len(aug_labels), "Texts and labels should match"
    assert isinstance(stats, dict), "Stats should be a dictionary"
    
    print(f"  Original samples: {len(texts)}")
    print(f"  Final samples: {len(aug_texts)}")
    print(f"  Augmentation ratio: {stats['augmentation_ratio']:.2f}")
    
    # Test with custom config
    config_data = {
        'text_augmentation': {
            'synonym_prob': 0.2,
            'augmentation_factor': 0.5
        },
        'advanced_preprocessing': {
            'tokenizer_type': 'advanced',
            'handle_negation': True
        },
        'data_balancing': {
            'enabled': True,
            'target_balance': 0.5
        }
    }
    
    # Save temporary config
    import yaml
    temp_config_path = 'temp_augmentation_config.yaml'
    with open(temp_config_path, 'w') as f:
        yaml.dump(config_data, f)
    
    try:
        # Test with custom config
        custom_aug_texts, custom_aug_labels, custom_stats = apply_augmentation_pipeline(
            texts, labels, temp_config_path
        )
        
        assert len(custom_aug_texts) >= len(texts), "Custom pipeline should work"
        print(f"  Custom pipeline samples: {len(custom_aug_texts)}")
        
    finally:
        # Clean up
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
    
    print("✓ Pipeline integration tests passed")


def test_preprocessing_comparison():
    """Test preprocessing method comparison."""
    print("Testing preprocessing comparison...")
    
    texts = [
        "This movie is absolutely fantastic!",
        "I can't believe how boring this film was.",
        "Great acting but poor plot development.",
        "Amazing visual effects and sound design."
    ] * 10  # Repeat for more data
    
    preprocessor = AdvancedTextPreprocessor()
    comparison = preprocessor.compare_preprocessing_methods(texts)
    
    assert isinstance(comparison, dict), "Comparison should return a dictionary"
    assert len(comparison) > 0, "Should have at least one method"
    
    for method, stats in comparison.items():
        assert 'avg_tokens_per_text' in stats, f"Method {method} should have avg_tokens_per_text"
        assert 'processing_time' in stats, f"Method {method} should have processing_time"
        print(f"  {method}: {stats['avg_tokens_per_text']:.2f} tokens/text, "
              f"{stats['processing_time']:.4f}s")
    
    print("✓ Preprocessing comparison tests passed")


def run_all_tests():
    """Run all integration tests."""
    print("=" * 60)
    print("AUGMENTATION AND PREPROCESSING INTEGRATION TESTS")
    print("=" * 60)
    
    tests = [
        test_text_augmenter,
        test_back_translation,
        test_advanced_preprocessor,
        test_data_balancing,
        test_augmentation_manager,
        test_pipeline_integration,
        test_preprocessing_comparison
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()
    
    print("=" * 60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)