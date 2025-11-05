"""
Data Augmentation and Advanced Preprocessing Demo

This script demonstrates the text augmentation and advanced preprocessing
capabilities of the LSTM sentiment classifier system.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns

from src.data import (
    TextAugmenter,
    BackTranslationAugmenter,
    AdvancedTextPreprocessor,
    create_balanced_dataset,
    create_preprocessing_pipeline,
    evaluate_preprocessing_pipeline
)


def demo_text_augmentation():
    """Demonstrate text augmentation capabilities."""
    print("=" * 60)
    print("TEXT AUGMENTATION DEMO")
    print("=" * 60)
    
    # Sample movie reviews
    sample_texts = [
        "This movie was absolutely fantastic! I loved every minute of it.",
        "Terrible film. Waste of time and money. Very disappointing.",
        "The acting was good but the plot was confusing and boring.",
        "Amazing cinematography and great performances by all actors.",
        "Not the worst movie I've seen, but definitely not the best either."
    ]
    
    # Initialize augmenter
    augmenter = TextAugmenter(
        synonym_prob=0.3,
        insert_prob=0.1,
        delete_prob=0.1,
        swap_prob=0.2,
        max_augmentations=2,
        preserve_length=True
    )
    
    print("\nOriginal texts and their augmentations:")
    print("-" * 60)
    
    for i, text in enumerate(sample_texts):
        print(f"\nOriginal {i+1}: {text}")
        
        # Generate augmentations
        augmented_versions = augmenter.augment_text(text, num_augmentations=3)
        
        for j, aug_text in enumerate(augmented_versions):
            quality_ok = augmenter.validate_augmentation_quality(text, aug_text)
            status = "✓" if quality_ok else "✗"
            print(f"  Aug {j+1} {status}: {aug_text}")
    
    # Demonstrate dataset augmentation
    print("\n" + "=" * 60)
    print("DATASET AUGMENTATION DEMO")
    print("=" * 60)
    
    # Create sample dataset with imbalanced classes
    texts = [
        "Great movie, loved it!",
        "Excellent film with amazing acting.",
        "Wonderful story and great direction.",
        "Bad movie, very boring.",
        "Terrible acting and poor plot."
    ]
    labels = [1, 1, 1, 0, 0]  # Imbalanced: 3 positive, 2 negative
    
    print(f"\nOriginal dataset:")
    print(f"Positive samples: {sum(labels)}")
    print(f"Negative samples: {len(labels) - sum(labels)}")
    print(f"Total samples: {len(labels)}")
    
    # Balance the dataset
    balanced_texts, balanced_labels = create_balanced_dataset(texts, labels, target_balance=0.5)
    
    print(f"\nBalanced dataset:")
    print(f"Positive samples: {sum(balanced_labels)}")
    print(f"Negative samples: {len(balanced_labels) - sum(balanced_labels)}")
    print(f"Total samples: {len(balanced_labels)}")
    
    # Show new samples
    print(f"\nNew augmented samples:")
    for i in range(len(texts), len(balanced_texts)):
        print(f"  {i+1}: {balanced_texts[i]} (label: {balanced_labels[i]})")


def demo_back_translation():
    """Demonstrate back-translation augmentation."""
    print("\n" + "=" * 60)
    print("BACK-TRANSLATION DEMO")
    print("=" * 60)
    
    # Initialize back-translation augmenter
    back_translator = BackTranslationAugmenter(
        intermediate_languages=['es', 'fr', 'de', 'it']
    )
    
    sample_texts = [
        "This is an excellent movie with outstanding performances.",
        "The film was boring and the plot made no sense.",
        "I think this movie is good but not great."
    ]
    
    print("\nBack-translation examples:")
    print("-" * 40)
    
    for i, text in enumerate(sample_texts):
        print(f"\nOriginal: {text}")
        
        back_translated = back_translator.augment_text(text, num_augmentations=2)
        for j, bt_text in enumerate(back_translated):
            print(f"  BT {j+1}: {bt_text}")


def demo_advanced_preprocessing():
    """Demonstrate advanced preprocessing capabilities."""
    print("\n" + "=" * 60)
    print("ADVANCED PREPROCESSING DEMO")
    print("=" * 60)
    
    # Sample texts with various challenges
    sample_texts = [
        "I can't believe how AMAZING this movie was! It's the best film I've ever seen.",
        "The movie wasn't good. The acting was terrible and the plot didn't make sense.",
        "John Smith and Mary Johnson starred in this film. It was filmed in New York.",
        "LOL this movie is sooo bad!!! Why did I waste my money??? 😞",
        "The cinematography was beautiful, but the dialogue was poorly written."
    ]
    
    # Test different preprocessing configurations
    configs = {
        'basic': {
            'tokenizer_type': 'basic',
            'use_lemmatization': False,
            'use_stemming': False,
            'remove_stopwords': False,
            'handle_negation': False
        },
        'advanced': {
            'tokenizer_type': 'advanced',
            'use_lemmatization': True,
            'use_stemming': False,
            'remove_stopwords': False,
            'handle_negation': True,
            'preserve_entities': True
        },
        'nltk_based': {
            'tokenizer_type': 'nltk',
            'use_lemmatization': True,
            'use_stemming': False,
            'remove_stopwords': True,
            'handle_negation': True,
            'preserve_entities': True
        }
    }
    
    print("\nPreprocessing comparison:")
    print("-" * 60)
    
    for config_name, config in configs.items():
        print(f"\n{config_name.upper()} PREPROCESSING:")
        
        try:
            preprocessor = create_preprocessing_pipeline(config)
            
            for i, text in enumerate(sample_texts[:3]):  # Show first 3 examples
                tokens = preprocessor.tokenize(text)
                print(f"  Text {i+1}: {text}")
                print(f"    Tokens: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
                
                # Extract features
                features = preprocessor.extract_features(text)
                print(f"    Features: word_count={features['word_count']}, "
                      f"negation_count={features['negation_count']}, "
                      f"lexical_diversity={features['lexical_diversity']:.3f}")
        
        except Exception as e:
            print(f"  Error with {config_name}: {e}")


def demo_preprocessing_comparison():
    """Compare different preprocessing methods."""
    print("\n" + "=" * 60)
    print("PREPROCESSING METHOD COMPARISON")
    print("=" * 60)
    
    # Sample dataset
    sample_texts = [
        "This movie is absolutely fantastic! I loved every single moment.",
        "Terrible film. Complete waste of time and money.",
        "The acting was good, but the story wasn't very interesting.",
        "Amazing cinematography and excellent performances by the cast.",
        "Not the best movie, but definitely not the worst either.",
        "I can't believe how boring this film was. Very disappointing.",
        "Great movie! Highly recommended for everyone.",
        "The plot was confusing and the dialogue was poorly written."
    ] * 10  # Repeat to have more data
    
    # Create advanced preprocessor
    preprocessor = AdvancedTextPreprocessor(tokenizer_type='advanced')
    
    # Compare different methods
    comparison = preprocessor.compare_preprocessing_methods(sample_texts)
    
    print("\nMethod comparison results:")
    print("-" * 40)
    
    for method, stats in comparison.items():
        print(f"\n{method.upper()}:")
        print(f"  Average tokens per text: {stats['avg_tokens_per_text']:.2f}")
        print(f"  Unique tokens: {stats['unique_token_count']}")
        print(f"  Processing time: {stats['processing_time']:.4f} seconds")
    
    # Get detailed preprocessing statistics
    print(f"\nDetailed preprocessing statistics:")
    print("-" * 40)
    
    stats = preprocessor.get_preprocessing_stats(sample_texts)
    print(f"Total texts: {stats['total_texts']}")
    print(f"Average text length: {stats['avg_text_length']:.2f} characters")
    print(f"Average word count: {stats['avg_word_count']:.2f}")
    print(f"Vocabulary size: {stats['vocabulary_size']}")
    
    # Show most common tokens
    most_common = stats['token_distribution'].most_common(10)
    print(f"\nMost common tokens:")
    for token, count in most_common:
        print(f"  '{token}': {count}")


def demo_feature_extraction():
    """Demonstrate feature extraction capabilities."""
    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION DEMO")
    print("=" * 60)
    
    sample_texts = [
        "This movie is AMAZING!!! I absolutely loved it!",
        "Boring film. Not recommended.",
        "The movie wasn't bad, but it wasn't great either. Just okay.",
        "TERRIBLE MOVIE! WASTE OF TIME AND MONEY!",
        "Beautiful cinematography and excellent acting. Highly recommended."
    ]
    
    preprocessor = AdvancedTextPreprocessor()
    
    print("\nExtracted features:")
    print("-" * 40)
    
    for i, text in enumerate(sample_texts):
        features = preprocessor.extract_features(text)
        print(f"\nText {i+1}: {text}")
        print(f"  Character count: {features['char_count']}")
        print(f"  Word count: {features['word_count']}")
        print(f"  Token count: {features['token_count']}")
        print(f"  Lexical diversity: {features['lexical_diversity']:.3f}")
        print(f"  Exclamation count: {features['exclamation_count']}")
        print(f"  Uppercase ratio: {features['uppercase_ratio']:.3f}")
        print(f"  Negation count: {features['negation_count']}")
        print(f"  Average word length: {features['avg_word_length']:.2f}")


def create_augmentation_visualization():
    """Create visualizations for augmentation effects."""
    print("\n" + "=" * 60)
    print("AUGMENTATION VISUALIZATION")
    print("=" * 60)
    
    # Sample data
    original_texts = [
        "Great movie, loved it!",
        "Terrible film, very boring.",
        "The acting was excellent.",
        "Poor plot and bad dialogue.",
        "Amazing cinematography!"
    ] * 20  # Repeat to have more data
    
    labels = [1, 0, 1, 0, 1] * 20  # Corresponding labels
    
    # Create augmenter
    augmenter = TextAugmenter(
        synonym_prob=0.2,
        insert_prob=0.1,
        delete_prob=0.1,
        swap_prob=0.1
    )
    
    # Generate augmented dataset
    augmented_texts, augmented_labels = augmenter.augment_dataset(
        original_texts, labels, augmentation_factor=0.5
    )
    
    print(f"Original dataset size: {len(original_texts)}")
    print(f"Augmented dataset size: {len(augmented_texts)}")
    print(f"Augmentation ratio: {len(augmented_texts) / len(original_texts):.2f}")
    
    # Analyze text length distribution
    original_lengths = [len(text.split()) for text in original_texts]
    augmented_lengths = [len(text.split()) for text in augmented_texts[len(original_texts):]]
    
    print(f"\nText length statistics:")
    print(f"Original - Mean: {np.mean(original_lengths):.2f}, Std: {np.std(original_lengths):.2f}")
    print(f"Augmented - Mean: {np.mean(augmented_lengths):.2f}, Std: {np.std(augmented_lengths):.2f}")
    
    # Show some examples of augmented texts
    print(f"\nAugmentation examples:")
    print("-" * 40)
    
    for i in range(5):
        original_idx = i
        augmented_idx = len(original_texts) + i
        
        if augmented_idx < len(augmented_texts):
            print(f"\nOriginal: {original_texts[original_idx]}")
            print(f"Augmented: {augmented_texts[augmented_idx]}")


def main():
    """Run all demonstration functions."""
    print("LSTM Sentiment Classifier - Data Augmentation & Advanced Preprocessing Demo")
    print("=" * 80)
    
    try:
        demo_text_augmentation()
        demo_back_translation()
        demo_advanced_preprocessing()
        demo_preprocessing_comparison()
        demo_feature_extraction()
        create_augmentation_visualization()
        
        print("\n" + "=" * 80)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()