#!/usr/bin/env python3
"""
Main inference script for LSTM Sentiment Classifier.

This script provides command-line sentiment prediction capabilities
with support for single texts, batch processing, and interactive mode.
"""

import argparse
import os
import sys
import json
import yaml
import logging
from typing import List

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from inference.inference_engine import InferenceEngine, create_inference_engine


def setup_logging(log_level: str = 'INFO'):
    """Set up logging configuration."""
    level = getattr(logging, log_level.upper())
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_config_file(config_path: str) -> dict:
    """Load configuration from YAML or JSON file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            return yaml.safe_load(f)
        elif config_path.endswith('.json'):
            return json.load(f)
        else:
            raise ValueError("Configuration file must be .yaml, .yml, or .json")


def predict_single_text(engine: InferenceEngine, text: str, args) -> None:
    """
    Predict sentiment for a single text and print results.
    
    Args:
        engine: Initialized inference engine
        text: Text to classify
        args: Command line arguments
    """
    try:
        if args.show_probability:
            sentiment, probability, confidence = engine.predict_sentiment_with_probability(text)
            print(f"Text: {text}")
            print(f"Sentiment: {sentiment}")
            print(f"Probability: {probability:.4f}")
            print(f"Confidence: {confidence:.4f}")
        else:
            sentiment, confidence = engine.predict_sentiment(text, args.threshold)
            print(f"Text: {text}")
            print(f"Sentiment: {sentiment}")
            print(f"Confidence: {confidence:.4f}")
            
        # Threshold analysis if requested
        if args.threshold_analysis:
            analysis = engine.predict_with_threshold_analysis(text)
            print(f"\nThreshold Analysis:")
            print(f"Raw probability: {analysis['raw_probability']:.4f}")
            for threshold, result in analysis['threshold_analysis'].items():
                print(f"  Threshold {threshold}: {result['sentiment']} "
                      f"(confidence: {result['confidence']:.4f})")
            
    except Exception as e:
        print(f"Error predicting sentiment: {e}")


def predict_batch_texts(engine: InferenceEngine, texts: List[str], args) -> None:
    """
    Predict sentiment for multiple texts and print results.
    
    Args:
        engine: Initialized inference engine
        texts: List of texts to classify
        args: Command line arguments
    """
    try:
        if args.show_probability:
            results = engine.batch_predict_with_probabilities(texts)
            for i, (text, (sentiment, probability, confidence)) in enumerate(zip(texts, results)):
                print(f"\n--- Prediction {i+1} ---")
                print(f"Text: {text}")
                print(f"Sentiment: {sentiment}")
                print(f"Probability: {probability:.4f}")
                print(f"Confidence: {confidence:.4f}")
        else:
            results = engine.batch_predict(texts, args.threshold)
            for i, (text, (sentiment, confidence)) in enumerate(zip(texts, results)):
                print(f"\n--- Prediction {i+1} ---")
                print(f"Text: {text}")
                print(f"Sentiment: {sentiment}")
                print(f"Confidence: {confidence:.4f}")
                
        # Print batch statistics
        if args.show_stats:
            stats = engine.get_prediction_stats(texts)
            print(f"\n--- Batch Statistics ---")
            print(f"Total texts: {stats['total_texts']}")
            print(f"Positive predictions: {stats['positive_predictions']}")
            print(f"Negative predictions: {stats['negative_predictions']}")
            print(f"Positive ratio: {stats['positive_ratio']:.2%}")
            print(f"Average confidence: {stats['average_confidence']:.4f}")
            print(f"High confidence (>0.7): {stats['high_confidence_count']}")
            print(f"Low confidence (<0.3): {stats['low_confidence_count']}")
        
    except Exception as e:
        print(f"Error predicting batch sentiment: {e}")


def predict_from_file(engine: InferenceEngine, filepath: str, args) -> None:
    """
    Predict sentiment for texts from a file.
    
    Args:
        engine: Initialized inference engine
        filepath: Path to file containing texts (one per line)
        args: Command line arguments
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        if not texts:
            print("No valid texts found in file.")
            return
            
        print(f"Processing {len(texts)} texts from {filepath}")
        predict_batch_texts(engine, texts, args)
        
        # Save results if requested
        if args.output_file:
            save_predictions_to_file(engine, texts, args)
        
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
    except Exception as e:
        print(f"Error reading file: {e}")


def save_predictions_to_file(engine: InferenceEngine, texts: List[str], args) -> None:
    """Save predictions to output file."""
    try:
        if args.show_probability:
            results = engine.batch_predict_with_probabilities(texts)
            output_data = []
            for text, (sentiment, probability, confidence) in zip(texts, results):
                output_data.append({
                    'text': text,
                    'sentiment': sentiment,
                    'probability': probability,
                    'confidence': confidence
                })
        else:
            results = engine.batch_predict(texts, args.threshold)
            output_data = []
            for text, (sentiment, confidence) in zip(texts, results):
                output_data.append({
                    'text': text,
                    'sentiment': sentiment,
                    'confidence': confidence
                })
        
        # Add batch statistics
        stats = engine.get_prediction_stats(texts)
        output = {
            'predictions': output_data,
            'statistics': stats,
            'configuration': {
                'threshold': args.threshold,
                'model_path': args.model_path,
                'vocab_path': args.vocab_path
            }
        }
        
        # Save based on file extension
        if args.output_file.endswith('.json'):
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
        elif args.output_file.endswith('.csv'):
            import pandas as pd
            df = pd.DataFrame(output_data)
            df.to_csv(args.output_file, index=False)
        else:
            # Default to JSON
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {args.output_file}")
        
    except Exception as e:
        print(f"Error saving results: {e}")


def interactive_mode(engine: InferenceEngine, args) -> None:
    """
    Run interactive prediction mode.
    
    Args:
        engine: Initialized inference engine
        args: Command line arguments
    """
    print("Interactive sentiment prediction mode.")
    print("Commands:")
    print("  - Enter text to classify")
    print("  - 'info' to show model information")
    print("  - 'config' to show current configuration")
    print("  - 'quit' or 'exit' to exit")
    print()
    
    while True:
        try:
            text = input("> ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            elif text.lower() == 'info':
                info = engine.get_model_info()
                print("\nModel Information:")
                print(f"  Device: {info['device']}")
                print(f"  Vocabulary size: {info['vocab_info']['vocab_size']}")
                print(f"  Max sequence length: {info['vocab_info']['max_length']}")
                print(f"  Total parameters: {info['model_parameters']['total_parameters']:,}")
                print()
                continue
            elif text.lower() == 'config':
                print("\nCurrent Configuration:")
                print(f"  Threshold: {args.threshold}")
                print(f"  Show probability: {args.show_probability}")
                print(f"  Device: {args.device}")
                print()
                continue
            elif not text:
                print("Please enter some text.")
                continue
                
            predict_single_text(engine, text, args)
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main inference function."""
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
  
  # Show raw probabilities and save results
  python predict.py -m model.pth -v vocab.pth -f reviews.txt --show-probability -o results.json
        """
    )
    
    # Configuration file
    parser.add_argument('--config', type=str,
                       help='Path to configuration file (YAML or JSON)')
    
    # Required arguments
    parser.add_argument('-m', '--model-path', required=True,
                       help='Path to trained model checkpoint file')
    parser.add_argument('-v', '--vocab-path', required=True,
                       help='Path to vocabulary file')
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-t', '--text',
                           help='Single text to classify')
    input_group.add_argument('-f', '--file',
                           help='File containing texts to classify (one per line)')
    input_group.add_argument('--interactive', action='store_true',
                           help='Run in interactive mode')
    
    # Prediction options
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Decision threshold for classification')
    parser.add_argument('--show-probability', action='store_true',
                       help='Show raw probability scores')
    parser.add_argument('--threshold-analysis', action='store_true',
                       help='Show threshold analysis for single text')
    parser.add_argument('--show-stats', action='store_true',
                       help='Show batch statistics')
    
    # Output options
    parser.add_argument('-o', '--output-file',
                       help='Save results to file (JSON or CSV format)')
    
    # System options
    parser.add_argument('--device',
                       help='Device to use (cpu/cuda, default: auto-detect)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='WARNING', help='Logging level')
    
    args = parser.parse_args()
    
    # Load configuration file if provided
    if args.config:
        config_data = load_config_file(args.config)
        
        # Update args with config file values (command line args take precedence)
        for key, value in config_data.items():
            if hasattr(args, key) and getattr(args, key) == parser.get_default(key):
                setattr(args, key, value)
    
    # Set up logging
    setup_logging(args.log_level)
    
    # Validate threshold
    if not 0 <= args.threshold <= 1:
        print("Error: Threshold must be between 0 and 1")
        sys.exit(1)
    
    # Initialize inference engine
    try:
        print("Loading model and vocabulary...")
        engine = create_inference_engine(args.model_path, args.vocab_path, args.device)
        
        # Print model info if verbose
        if args.log_level == 'INFO':
            info = engine.get_model_info()
            print(f"Model loaded successfully on {info['device']}")
            print(f"Vocabulary size: {info['vocab_info']['vocab_size']}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Execute based on input mode
    try:
        if args.text:
            predict_single_text(engine, args.text, args)
        elif args.file:
            predict_from_file(engine, args.file, args)
        elif args.interactive:
            interactive_mode(engine, args)
            
    except Exception as e:
        print(f"Prediction error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()