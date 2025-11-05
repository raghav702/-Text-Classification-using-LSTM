#!/usr/bin/env python3
"""
Debug script to identify why the model gives consistent results
"""

import sys
import os
import torch
import numpy as np

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from inference.inference_engine import create_inference_engine

def debug_model_predictions():
    """Debug the model to understand why predictions are consistent."""
    
    # Model paths
    model_path = "models/quick/quick_lstm_model_20251101_163308.pth"
    vocab_path = "models/quick/quick_lstm_model_20251101_163308_vocabulary.pth"
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
    
    if not os.path.exists(vocab_path):
        print(f"Vocabulary not found: {vocab_path}")
        return
    
    print("Loading model...")
    engine = create_inference_engine(model_path, vocab_path, device='cpu')
    
    # Test different types of inputs
    test_texts = [
        "This movie is absolutely amazing and fantastic!",
        "This movie is terrible and awful!",
        "Great film with excellent acting",
        "Boring movie with bad plot",
        "I love this movie so much",
        "I hate this movie completely",
        "The movie was okay",
        "Best movie ever made",
        "Worst movie ever made",
        "a",  # Very short
        "the the the the the",  # Repetitive
        "",  # Empty (will cause error, but let's see)
    ]
    
    print("\n" + "="*80)
    print("DEBUGGING MODEL PREDICTIONS")
    print("="*80)
    
    for i, text in enumerate(test_texts):
        if not text.strip():
            print(f"\nTest {i+1}: [EMPTY TEXT] - Skipping")
            continue
            
        print(f"\nTest {i+1}: '{text}'")
        print("-" * 60)
        
        try:
            # Get detailed prediction
            sentiment, confidence = engine.predict_sentiment(text)
            prob_sentiment, probability, prob_confidence = engine.predict_sentiment_with_probability(text)
            
            print(f"Sentiment: {sentiment}")
            print(f"Confidence: {confidence:.4f}")
            print(f"Raw Probability: {probability:.4f}")
            
            # Debug the preprocessing
            preprocessed = engine.preprocessor.text_to_sequence(text)
            padded = engine.preprocessor.pad_sequences([preprocessed])
            
            print(f"Tokens: {engine.preprocessor.tokenize(text)}")
            print(f"Sequence length: {len(preprocessed)}")
            print(f"Sequence (first 10): {preprocessed[:10].tolist()}")
            print(f"Padded shape: {padded.shape}")
            
            # Get raw model output
            with torch.no_grad():
                logits = engine.model(padded.to(engine.device))
                raw_logit = logits.item()
                sigmoid_prob = torch.sigmoid(logits).item()
                
            print(f"Raw logit: {raw_logit:.4f}")
            print(f"Sigmoid probability: {sigmoid_prob:.4f}")
            
        except Exception as e:
            print(f"Error: {e}")
    
    # Check model weights
    print("\n" + "="*80)
    print("MODEL WEIGHT ANALYSIS")
    print("="*80)
    
    model = engine.model
    
    # Check if weights are properly initialized
    print(f"Embedding weights range: {model.embedding.weight.min():.4f} to {model.embedding.weight.max():.4f}")
    print(f"Embedding weights std: {model.embedding.weight.std():.4f}")
    
    print(f"LSTM weight_ih range: {model.lstm.weight_ih_l0.min():.4f} to {model.lstm.weight_ih_l0.max():.4f}")
    print(f"LSTM weight_hh range: {model.lstm.weight_hh_l0.min():.4f} to {model.lstm.weight_hh_l0.max():.4f}")
    
    print(f"FC1 weight range: {model.fc1.weight.min():.4f} to {model.fc1.weight.max():.4f}")
    print(f"FC2 weight range: {model.fc2.weight.min():.4f} to {model.fc2.weight.max():.4f}")
    
    # Check for dead neurons
    fc1_weights_zero = (model.fc1.weight.abs() < 1e-6).sum().item()
    fc2_weights_zero = (model.fc2.weight.abs() < 1e-6).sum().item()
    
    print(f"FC1 near-zero weights: {fc1_weights_zero}/{model.fc1.weight.numel()}")
    print(f"FC2 near-zero weights: {fc2_weights_zero}/{model.fc2.weight.numel()}")
    
    # Test with different thresholds
    print("\n" + "="*80)
    print("THRESHOLD ANALYSIS")
    print("="*80)
    
    test_text = "This movie is great!"
    analysis = engine.predict_with_threshold_analysis(test_text)
    
    print(f"Text: {test_text}")
    print(f"Raw probability: {analysis['raw_probability']:.4f}")
    
    for threshold, result in analysis['threshold_analysis'].items():
        print(f"Threshold {threshold}: {result['sentiment']} (conf: {result['confidence']:.4f})")

if __name__ == "__main__":
    debug_model_predictions()