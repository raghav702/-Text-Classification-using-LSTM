#!/usr/bin/env python3
"""
Command-line interface for LSTM sentiment prediction.

This script provides a simple CLI for making sentiment predictions
using a trained LSTM model.
"""

import argparse
import sys
import os
import logging
from typing import List

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference_engine import InferenceEngine


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def predict_single_text(engine: InferenceEngine, text: str, threshold: float = 0.5, 
                       show_probability: bool = False) -> None:
    """
    Predict sentiment for a single text and print results.
    
    Args:
        engine: Initialized inference engine
        text: Text to classify
        threshold: Decision threshold
        show_probability: Whether to show raw probability
    """
    try:
        if show_probability:
            sentiment, probability, confidence = engine.predict_sentiment_with_probability(text)
            print(f"Text: {text}")
            print(f"Sentiment: {sentiment}")
            print(f"Probability: {probability:.4f}")
            print(f"Confidence: {confidence:.4f}")
        else:
            sentiment, confidence = engine.predict_sentiment(text, threshold)
            print(f"Text: {text}")
            print(f"Sentiment: {sentiment}")
            print(f"Confidence: {confidence:.4f}")
            
    except Exception as e:
        print(f"Error predicting sentiment: {e}")


def predict_batch_texts(engine: InferenceEngine, texts: List[str], threshold: float = 0.5,
                       show_probability: bool = False) -> None:
    """
    Predict sentiment for multiple texts and print results.
    
    Args:
        engine: Initialized inference engine
        texts: List of texts to classify
        threshold: Decision threshold
        show_probability: Whether to show raw probabilities
    """
    try:
        if show_probability:
            results = engine.batch_predict_with_probabilities(texts)
            for i, (text, (sentiment, probability, confidence)) in enumerate(zip(texts, results)):
                print(f"\n--- Prediction {i+1} ---")
                print(f"Text: {text}")
                print(f"Sentiment: {sentiment}")
                print(f"Probability: {probability:.4f}")
                print(f"Confidence: {confidence:.4f}")
        else:
            results = engine.batch_predict(texts, threshold)
            for i, (text, (sentiment, confidence)) in enumerate(zip(texts, results)):
                print(f"\n--- Prediction {i+1} ---")
                print(f"Text: {text}")
                print(f"Sentiment: {sentiment}")
                print(f"Confidence: {confidence:.4f}")
                
        # Print batch statistics
        stats = engine.get_prediction_stats(texts)
        print(f"\n--- Batch Statistics ---")
        print(f"Total texts: {stats['total_texts']}")
        print(f"Positive predictions: {stats['positive_predictions']}")
        print(f"Negative predictions: {stats['negative_predictions']}")
        print(f"Average confidence: {stats['average_confidence']:.4f}")
        
    except Exception as e:
        print(f"Error predicting batch sentiment: {e}")


def predict_from_file(engine: InferenceEngine, filepath: str, threshold: float = 0.5,
                     show_probability: bool = False) -> None:
    """
    Predict sentiment for texts from a file.
    
    Args:
        engine: Initialized inference engine
        filepath: Path to file containing texts (one per line)
        threshold: Decision threshold
        show_probability: Whether to show raw probabilities
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        if not texts:
            print("No valid texts found in file.")
            return
            
        print(f"Processing {len(texts)} texts from {filepath}")
        predict_batch_texts(engine, texts, threshold, show_probability)
        
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
    except Exception as e:
        print(f"Error reading file: {e}")


def interactive_mode(engine: InferenceEngine, threshold: float = 0.5, 
                    show_probability: bool = False) -> None:
    """
    Run interactive prediction mode.
    
    Args:
        engine: Initialized inference engine
        threshold: Decision threshold
        show_probability: Whether to show raw probabilities
    """
    print("Interactive sentiment prediction mode.")
    print("Enter text to classify (or 'quit' to exit):")
    
    while True:
        try:
            text = input("\n> ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
                
            if not text:
                print("Please enter some text.")
                continue
                
            predict_single_text(engine, text, threshold, show_probability)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="LSTM Sentiment Classifier - Prediction CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict single text
  python predict.py -m model.pth -v vocab.pth -t "This movie was great!"
  
  # Predict with custom threshold
  python predict.py -m model.pth -v vocab.pth -t "Good film" --threshold 0.6
  
  # Batch prediction from file
  python predict.py -m model.pth -v vocab.pth -f reviews.txt
  
  # Interactive mode
  python predict.py -m model.pth -v vocab.pth --interactive
  
  # Show raw probabilities
  python predict.py -m model.pth -v vocab.pth -t "Amazing!" --show-probability
        """
    )
    
    # Required arguments
    parser.add_argument('-m', '--model', required=True,
                       help='Path to trained model checkpoint file')
    parser.add_argument('-v', '--vocab', required=True,
                       help='Path to vocabulary file')
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-t', '--text',
                           help='Single text to classify')
    input_group.add_argument('-f', '--file',
                           help='File containing texts to classify (one per line)')
    input_group.add_argument('--interactive', action='store_true',
                           help='Run in interactive mode')
    
    # Optional arguments
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Decision threshold for classification (default: 0.5)')
    parser.add_argument('--show-probability', action='store_true',
                       help='Show raw probability scores')
    parser.add_argument('--device',
                       help='Device to use (cpu/cuda, default: auto-detect)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    
    # Validate threshold
    if not 0 <= args.threshold <= 1:
        print("Error: Threshold must be between 0 and 1")
        sys.exit(1)
    
    # Initialize inference engine
    try:
        print("Loading model and vocabulary...")
        engine = InferenceEngine(device=args.device)
        engine.load_model(args.model, args.vocab)
        
        # Print model info
        if args.verbose:
            info = engine.get_model_info()
            print(f"Model loaded successfully on {info['device']}")
            print(f"Vocabulary size: {info['vocab_info']['vocab_size']}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Execute based on input mode
    try:
        if args.text:
            predict_single_text(engine, args.text, args.threshold, args.show_probability)
        elif args.file:
            predict_from_file(engine, args.file, args.threshold, args.show_probability)
        elif args.interactive:
            interactive_mode(engine, args.threshold, args.show_probability)
            
    except Exception as e:
        print(f"Prediction error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()