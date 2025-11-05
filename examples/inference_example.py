#!/usr/bin/env python3
"""
Example usage of the LSTM Sentiment Classifier Inference Engine.

This script demonstrates how to use the inference engine for sentiment
classification on movie reviews.
"""

import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from inference import InferenceEngine, create_inference_engine


def example_single_prediction():
    """Example of single text prediction."""
    print("=== Single Text Prediction Example ===")
    
    # Example texts
    positive_text = "This movie was absolutely fantastic! Great acting and amazing plot."
    negative_text = "Terrible film. Boring plot and bad acting. Complete waste of time."
    
    # Note: You'll need to provide actual paths to your trained model and vocabulary
    model_path = "models/lstm_sentiment_model.pth"
    vocab_path = "models/vocabulary.pth"
    
    try:
        # Create inference engine
        engine = InferenceEngine()
        
        # Load model (this will fail without actual model files)
        # engine.load_model(model_path, vocab_path)
        
        # Example predictions (commented out since model files don't exist yet)
        # sentiment, confidence = engine.predict_sentiment(positive_text)
        # print(f"Text: {positive_text}")
        # print(f"Predicted sentiment: {sentiment}")
        # print(f"Confidence: {confidence:.4f}")
        # print()
        
        # sentiment, confidence = engine.predict_sentiment(negative_text)
        # print(f"Text: {negative_text}")
        # print(f"Predicted sentiment: {sentiment}")
        # print(f"Confidence: {confidence:.4f}")
        
        print("Note: This example requires trained model files to run.")
        print("Train a model first using the training pipeline.")
        
    except Exception as e:
        print(f"Error: {e}")
        print("This is expected if model files don't exist yet.")


def example_batch_prediction():
    """Example of batch prediction."""
    print("\n=== Batch Prediction Example ===")
    
    # Example batch of texts
    texts = [
        "Amazing movie with great storyline!",
        "Worst film I've ever seen.",
        "Pretty good, but could be better.",
        "Absolutely loved it! Highly recommend.",
        "Not my cup of tea, but decent acting."
    ]
    
    model_path = "models/lstm_sentiment_model.pth"
    vocab_path = "models/vocabulary.pth"
    
    try:
        # Create inference engine using factory function
        # engine = create_inference_engine(model_path, vocab_path)
        
        # Example batch prediction (commented out since model files don't exist yet)
        # results = engine.batch_predict(texts)
        
        # for text, (sentiment, confidence) in zip(texts, results):
        #     print(f"Text: {text}")
        #     print(f"Sentiment: {sentiment} (confidence: {confidence:.4f})")
        #     print()
        
        # Get batch statistics
        # stats = engine.get_prediction_stats(texts)
        # print("Batch Statistics:")
        # print(f"Total texts: {stats['total_texts']}")
        # print(f"Positive predictions: {stats['positive_predictions']}")
        # print(f"Negative predictions: {stats['negative_predictions']}")
        # print(f"Average confidence: {stats['average_confidence']:.4f}")
        
        print("Note: This example requires trained model files to run.")
        print("Train a model first using the training pipeline.")
        
    except Exception as e:
        print(f"Error: {e}")
        print("This is expected if model files don't exist yet.")


def example_threshold_analysis():
    """Example of threshold analysis."""
    print("\n=== Threshold Analysis Example ===")
    
    text = "This movie is okay, nothing special but watchable."
    model_path = "models/lstm_sentiment_model.pth"
    vocab_path = "models/vocabulary.pth"
    
    try:
        # engine = create_inference_engine(model_path, vocab_path)
        
        # Analyze prediction across different thresholds
        # analysis = engine.predict_with_threshold_analysis(text)
        
        # print(f"Text: {analysis['text']}")
        # print(f"Raw probability: {analysis['raw_probability']:.4f}")
        # print(f"Base confidence: {analysis['base_confidence']:.4f}")
        # print("\nThreshold Analysis:")
        
        # for threshold, result in analysis['threshold_analysis'].items():
        #     print(f"  Threshold {threshold}: {result['sentiment']} "
        #           f"(confidence: {result['confidence']:.4f})")
        
        print("Note: This example requires trained model files to run.")
        print("Train a model first using the training pipeline.")
        
    except Exception as e:
        print(f"Error: {e}")
        print("This is expected if model files don't exist yet.")


def example_model_info():
    """Example of getting model information."""
    print("\n=== Model Information Example ===")
    
    model_path = "models/lstm_sentiment_model.pth"
    vocab_path = "models/vocabulary.pth"
    
    try:
        # engine = create_inference_engine(model_path, vocab_path)
        # info = engine.get_model_info()
        
        # print("Model Configuration:")
        # for key, value in info['model_config'].items():
        #     print(f"  {key}: {value}")
        
        # print(f"\nVocabulary Info:")
        # vocab_info = info['vocab_info']
        # print(f"  Vocabulary size: {vocab_info['vocab_size']}")
        # print(f"  Max sequence length: {vocab_info['max_length']}")
        # print(f"  PAD token: {vocab_info['pad_token']}")
        # print(f"  UNK token: {vocab_info['unk_token']}")
        
        # print(f"\nModel Parameters:")
        # model_params = info['model_parameters']
        # print(f"  Total parameters: {model_params['total_parameters']:,}")
        # print(f"  Trainable parameters: {model_params['trainable_parameters']:,}")
        
        print("Note: This example requires trained model files to run.")
        print("Train a model first using the training pipeline.")
        
    except Exception as e:
        print(f"Error: {e}")
        print("This is expected if model files don't exist yet.")


def main():
    """Run all examples."""
    print("LSTM Sentiment Classifier - Inference Examples")
    print("=" * 50)
    
    example_single_prediction()
    example_batch_prediction()
    example_threshold_analysis()
    example_model_info()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nTo use these examples with real predictions:")
    print("1. Train a model using the training pipeline")
    print("2. Update the model_path and vocab_path variables")
    print("3. Uncomment the prediction code")


if __name__ == '__main__':
    main()