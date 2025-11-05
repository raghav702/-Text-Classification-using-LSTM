#!/usr/bin/env python3
"""
Test script for embedding integration with LSTM sentiment classifier.

This script demonstrates the embedding functionality and validates
the integration between GloVe embeddings and the LSTM model.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append('src')

from models.model_factory import ModelFactory, create_sentiment_classifier
from utils.embedding_utils import EmbeddingManager
from data.text_preprocessor import TextPreprocessor


def create_sample_vocabulary(size: int = 1000) -> dict:
    """Create a sample vocabulary for testing."""
    # Common words for sentiment analysis
    sentiment_words = [
        'good', 'bad', 'great', 'terrible', 'amazing', 'awful', 'excellent', 'horrible',
        'fantastic', 'disappointing', 'wonderful', 'dreadful', 'outstanding', 'pathetic',
        'brilliant', 'mediocre', 'superb', 'lousy', 'marvelous', 'abysmal'
    ]
    
    common_words = [
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
        'movie', 'film', 'story', 'plot', 'character', 'actor', 'actress', 'director',
        'scene', 'dialogue', 'action', 'drama', 'comedy', 'thriller', 'romance'
    ]
    
    # Create vocabulary with special tokens
    vocab = {
        '<PAD>': 0,
        '<UNK>': 1,
        '<START>': 2,
        '<END>': 3
    }
    
    # Add sentiment and common words
    all_words = sentiment_words + common_words
    for i, word in enumerate(all_words):
        if word not in vocab:
            vocab[word] = len(vocab)
    
    # Fill remaining slots with dummy words
    while len(vocab) < size:
        dummy_word = f"word_{len(vocab)}"
        vocab[dummy_word] = len(vocab)
    
    return vocab


def test_embedding_download_and_processing():
    """Test GloVe embedding download and processing."""
    print("="*60)
    print("TESTING EMBEDDING DOWNLOAD AND PROCESSING")
    print("="*60)
    
    manager = EmbeddingManager()
    
    # Test download (will skip if already exists)
    print("1. Testing GloVe download...")
    success = manager.glove_processor.download_glove_embeddings()
    print(f"Download successful: {success}")
    
    # Test loading embeddings
    print("\n2. Testing embedding loading...")
    try:
        embeddings_300d = manager.glove_processor.load_glove_embeddings(300)
        print(f"Loaded {len(embeddings_300d):,} 300d embeddings")
        
        # Test a few words
        test_words = ['good', 'bad', 'movie']
        for word in test_words:
            if word in embeddings_300d:
                print(f"  '{word}': {embeddings_300d[word][:5]}... (shape: {embeddings_300d[word].shape})")
            else:
                print(f"  '{word}': not found")
                
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return False
    
    return True


def test_vocabulary_alignment():
    """Test embedding alignment with model vocabulary."""
    print("\n" + "="*60)
    print("TESTING VOCABULARY ALIGNMENT")
    print("="*60)
    
    # Create sample vocabulary
    vocab = create_sample_vocabulary(500)
    print(f"Created vocabulary with {len(vocab)} words")
    
    manager = EmbeddingManager()
    
    # Test alignment
    print("\n1. Testing embedding alignment...")
    try:
        embedding_matrix, alignment_stats = manager.prepare_embeddings_for_model(
            vocabulary=vocab,
            embedding_dim=300,
            init_strategy='random',
            use_cache=True
        )
        
        print(f"Embedding matrix shape: {embedding_matrix.shape}")
        print(f"Coverage: {alignment_stats['coverage']:.2%}")
        print(f"Found words: {alignment_stats['found_words']}")
        print(f"OOV words: {alignment_stats['oov_words']}")
        
        # Verify padding token is zero
        pad_embedding = embedding_matrix[0]
        is_zero = torch.allclose(pad_embedding, torch.zeros_like(pad_embedding))
        print(f"Padding token is zero: {is_zero}")
        
    except Exception as e:
        print(f"Error in alignment: {e}")
        return False
    
    # Test coverage analysis
    print("\n2. Testing coverage analysis...")
    try:
        coverage_stats = manager.analyze_vocabulary_coverage(vocab, embedding_dim=300)
        print(f"Coverage rate: {coverage_stats['coverage_rate']:.2%}")
        print(f"OOV patterns: {coverage_stats['oov_patterns']}")
        
    except Exception as e:
        print(f"Error in coverage analysis: {e}")
        return False
    
    return True


def test_model_creation():
    """Test LSTM model creation with embeddings."""
    print("\n" + "="*60)
    print("TESTING MODEL CREATION WITH EMBEDDINGS")
    print("="*60)
    
    # Create sample vocabulary
    vocab = create_sample_vocabulary(200)
    
    factory = ModelFactory()
    
    # Test different embedding strategies
    strategies = ['random', 'freeze', 'fine_tune']
    
    for strategy in strategies:
        print(f"\n{strategy.upper()} STRATEGY:")
        print("-" * 30)
        
        try:
            if strategy == 'random':
                model, info = factory.create_lstm_classifier(
                    vocabulary=vocab,
                    embedding_dim=300,
                    hidden_dim=64,  # Smaller for testing
                    use_pretrained_embeddings=False
                )
            else:
                model, info = factory.create_lstm_classifier(
                    vocabulary=vocab,
                    embedding_dim=300,
                    hidden_dim=64,  # Smaller for testing
                    use_pretrained_embeddings=True,
                    embedding_strategy=strategy
                )
            
            print(f"Model created successfully")
            print(f"Total parameters: {info['total_parameters']:,}")
            print(f"Trainable parameters: {info['trainable_parameters']:,}")
            
            if 'embedding_coverage' in info:
                print(f"Embedding coverage: {info['embedding_coverage']:.2%}")
            
            # Test forward pass
            batch_size = 4
            seq_length = 10
            dummy_input = torch.randint(0, len(vocab), (batch_size, seq_length))
            
            model.eval()
            with torch.no_grad():
                output = model(dummy_input)
                print(f"Forward pass successful: {output.shape}")
                
                # Test prediction methods
                probs = model.predict_proba(dummy_input)
                preds = model.predict(dummy_input)
                print(f"Probabilities shape: {probs.shape}")
                print(f"Predictions shape: {preds.shape}")
            
            # Test embedding statistics
            if hasattr(model, 'get_embedding_statistics'):
                embed_stats = model.get_embedding_statistics()
                print(f"Embedding frozen: {embed_stats['frozen']}")
                print(f"Zero embeddings: {embed_stats['zero_embeddings']}")
            
        except Exception as e:
            print(f"Error creating {strategy} model: {e}")
            return False
    
    return True


def test_convenience_functions():
    """Test convenience functions."""
    print("\n" + "="*60)
    print("TESTING CONVENIENCE FUNCTIONS")
    print("="*60)
    
    vocab = create_sample_vocabulary(100)
    
    # Test create_sentiment_classifier
    print("1. Testing create_sentiment_classifier...")
    try:
        model, info = create_sentiment_classifier(
            vocabulary=vocab,
            embedding_dim=300,
            hidden_dim=32,
            use_pretrained_embeddings=True,
            embedding_strategy='fine_tune'
        )
        
        print(f"Model created: {type(model).__name__}")
        print(f"Parameters: {info['total_parameters']:,}")
        
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("LSTM SENTIMENT CLASSIFIER - EMBEDDING INTEGRATION TESTS")
    print("=" * 80)
    
    tests = [
        ("Embedding Download and Processing", test_embedding_download_and_processing),
        ("Vocabulary Alignment", test_vocabulary_alignment),
        ("Model Creation", test_model_creation),
        ("Convenience Functions", test_convenience_functions)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        try:
            success = test_func()
            results[test_name] = success
            status = "PASSED" if success else "FAILED"
            print(f"{test_name}: {status}")
        except Exception as e:
            print(f"{test_name}: FAILED with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Embedding integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the output above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)