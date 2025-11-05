#!/usr/bin/env python3
"""
LSTM Sentiment Classifier - Main CLI Interface

This script provides a unified command-line interface for all operations:
training, inference, evaluation, and configuration management.
"""

import argparse
import os
import sys
import subprocess
import json
import yaml
from datetime import datetime


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{description}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✓ {description} completed successfully")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with exit code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print(f"\n⚠ {description} interrupted by user")
        return 130


def train_command(args):
    """Execute training command."""
    cmd = ['python', 'train.py']
    
    # Add arguments
    if args.config:
        cmd.extend(['--config', args.config])
    if args.data_dir:
        cmd.extend(['--data-dir', args.data_dir])
    if args.epochs:
        cmd.extend(['--epochs', str(args.epochs)])
    if args.batch_size:
        cmd.extend(['--batch-size', str(args.batch_size)])
    if args.learning_rate:
        cmd.extend(['--learning-rate', str(args.learning_rate)])
    if args.output_dir:
        cmd.extend(['--output-dir', args.output_dir])
    if args.model_name:
        cmd.extend(['--model-name', args.model_name])
    if args.device:
        cmd.extend(['--device', args.device])
    if args.no_glove:
        cmd.append('--use-glove')
        cmd.append('false')
    if args.log_level:
        cmd.extend(['--log-level', args.log_level])
    
    return run_command(cmd, "Training LSTM Sentiment Classifier")


def predict_command(args):
    """Execute prediction command."""
    cmd = ['python', 'predict.py']
    
    # Add required arguments
    cmd.extend(['--model-path', args.model_path])
    cmd.extend(['--vocab-path', args.vocab_path])
    
    # Add input arguments
    if args.text:
        cmd.extend(['--text', args.text])
    elif args.file:
        cmd.extend(['--file', args.file])
    elif args.interactive:
        cmd.append('--interactive')
    
    # Add optional arguments
    if args.config:
        cmd.extend(['--config', args.config])
    if args.threshold:
        cmd.extend(['--threshold', str(args.threshold)])
    if args.show_probability:
        cmd.append('--show-probability')
    if args.output_file:
        cmd.extend(['--output-file', args.output_file])
    if args.device:
        cmd.extend(['--device', args.device])
    if args.log_level:
        cmd.extend(['--log-level', args.log_level])
    
    return run_command(cmd, "Running sentiment prediction")


def evaluate_command(args):
    """Execute evaluation command."""
    cmd = ['python', 'evaluate.py']
    
    # Add required arguments
    if args.model_path:
        cmd.extend(['--model-path', args.model_path])
    elif args.model_paths:
        cmd.extend(['--model-paths'] + args.model_paths)
        if args.model_names:
            cmd.extend(['--model-names'] + args.model_names)
    
    cmd.extend(['--vocab-path', args.vocab_path])
    cmd.extend(['--data-dir', args.data_dir])
    
    # Add optional arguments
    if args.config:
        cmd.extend(['--config', args.config])
    if args.output_dir:
        cmd.extend(['--output-dir', args.output_dir])
    if args.batch_size:
        cmd.extend(['--batch-size', str(args.batch_size)])
    if args.device:
        cmd.extend(['--device', args.device])
    if args.show_plots:
        cmd.append('--show-plots')
    if args.no_visualizations:
        cmd.append('--no-visualizations')
    if args.no_reports:
        cmd.append('--no-reports')
    if args.log_level:
        cmd.extend(['--log-level', args.log_level])
    
    return run_command(cmd, "Evaluating model performance")


def config_command(args):
    """Handle configuration management."""
    if args.create:
        create_config(args.create, args.template, args.output)
    elif args.validate:
        validate_config(args.validate)
    elif args.show:
        show_config(args.show)
    else:
        print("No configuration action specified. Use --help for options.")
        return 1
    
    return 0


def create_config(config_type, template, output_path):
    """Create a configuration file."""
    templates = {
        'training': 'configs/training_config.yaml',
        'inference': 'configs/inference_config.yaml',
        'evaluation': 'configs/evaluation_config.yaml',
        'quick': 'configs/examples/quick_training.yaml',
        'production': 'configs/examples/production_training.yaml'
    }
    
    if config_type not in templates:
        print(f"Unknown config type: {config_type}")
        print(f"Available types: {', '.join(templates.keys())}")
        return
    
    template_path = templates[config_type]
    
    if not os.path.exists(template_path):
        print(f"Template not found: {template_path}")
        return
    
    # Use template path if no output specified
    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{config_type}_config_{timestamp}.yaml"
    
    # Copy template to output path
    import shutil
    shutil.copy2(template_path, output_path)
    
    print(f"Configuration file created: {output_path}")
    print(f"Based on template: {template_path}")
    print("\nEdit the file to customize your configuration.")


def validate_config(config_path):
    """Validate a configuration file."""
    if not os.path.exists(config_path):
        print(f"Configuration file not found: {config_path}")
        return
    
    try:
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
        
        print(f"✓ Configuration file is valid: {config_path}")
        print(f"Configuration contains {len(config)} parameters")
        
        # Show key parameters
        key_params = ['data_dir', 'model_path', 'vocab_path', 'epochs', 'batch_size']
        found_params = [param for param in key_params if param in config]
        
        if found_params:
            print("\nKey parameters found:")
            for param in found_params:
                print(f"  {param}: {config[param]}")
        
    except Exception as e:
        print(f"✗ Configuration file is invalid: {e}")


def show_config(config_path):
    """Display configuration file contents."""
    if not os.path.exists(config_path):
        print(f"Configuration file not found: {config_path}")
        return
    
    try:
        with open(config_path, 'r') as f:
            content = f.read()
        
        print(f"Configuration file: {config_path}")
        print("-" * 50)
        print(content)
        
    except Exception as e:
        print(f"Error reading configuration file: {e}")


def setup_command(args):
    """Setup project structure and download data."""
    print("Setting up LSTM Sentiment Classifier project...")
    
    # Create directories
    directories = [
        'data/imdb',
        'data/glove',
        'models',
        'checkpoints',
        'logs',
        'evaluation_results',
        'configs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Copy configuration templates if they don't exist
    config_files = [
        ('configs/training_config.yaml', 'Training configuration template'),
        ('configs/inference_config.yaml', 'Inference configuration template'),
        ('configs/evaluation_config.yaml', 'Evaluation configuration template')
    ]
    
    for config_file, description in config_files:
        if os.path.exists(config_file):
            print(f"✓ {description} already exists: {config_file}")
        else:
            print(f"⚠ {description} not found: {config_file}")
    
    print("\n✓ Project setup completed!")
    print("\nNext steps:")
    print("1. Download IMDB dataset to data/imdb/")
    print("2. Configure training parameters in configs/training_config.yaml")
    print("3. Run training: python lstm_sentiment.py train --config configs/training_config.yaml")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="LSTM Sentiment Classifier - Unified CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Setup project
  python lstm_sentiment.py setup
  
  # Train model with config file
  python lstm_sentiment.py train --config configs/training_config.yaml
  
  # Train model with command line arguments
  python lstm_sentiment.py train --data-dir data/imdb --epochs 10 --batch-size 32
  
  # Predict single text
  python lstm_sentiment.py predict -m models/model.pth -v models/vocab.pth -t "Great movie!"
  
  # Interactive prediction
  python lstm_sentiment.py predict -m models/model.pth -v models/vocab.pth --interactive
  
  # Evaluate model
  python lstm_sentiment.py evaluate -m models/model.pth -v models/vocab.pth -d data/imdb
  
  # Compare multiple models
  python lstm_sentiment.py evaluate --model-paths model1.pth model2.pth --model-names "Model 1" "Model 2" -v vocab.pth -d data/imdb
  
  # Create configuration file
  python lstm_sentiment.py config --create training --output my_config.yaml
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup project structure')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train LSTM model')
    train_parser.add_argument('--config', help='Configuration file path')
    train_parser.add_argument('--data-dir', help='IMDB dataset directory')
    train_parser.add_argument('--epochs', type=int, help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, help='Batch size')
    train_parser.add_argument('--learning-rate', type=float, help='Learning rate')
    train_parser.add_argument('--output-dir', help='Output directory for model')
    train_parser.add_argument('--model-name', help='Model name')
    train_parser.add_argument('--device', help='Device to use (cpu/cuda/auto)')
    train_parser.add_argument('--no-glove', action='store_true', help='Disable GloVe embeddings')
    train_parser.add_argument('--log-level', help='Logging level')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict sentiment')
    predict_parser.add_argument('-m', '--model-path', required=True, help='Model checkpoint path')
    predict_parser.add_argument('-v', '--vocab-path', required=True, help='Vocabulary file path')
    
    # Prediction input (mutually exclusive)
    predict_input = predict_parser.add_mutually_exclusive_group(required=True)
    predict_input.add_argument('-t', '--text', help='Text to classify')
    predict_input.add_argument('-f', '--file', help='File with texts to classify')
    predict_input.add_argument('--interactive', action='store_true', help='Interactive mode')
    
    predict_parser.add_argument('--config', help='Configuration file path')
    predict_parser.add_argument('--threshold', type=float, help='Decision threshold')
    predict_parser.add_argument('--show-probability', action='store_true', help='Show probabilities')
    predict_parser.add_argument('--output-file', help='Output file for results')
    predict_parser.add_argument('--device', help='Device to use')
    predict_parser.add_argument('--log-level', help='Logging level')
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    
    # Model input (mutually exclusive)
    eval_model = evaluate_parser.add_mutually_exclusive_group(required=True)
    eval_model.add_argument('-m', '--model-path', help='Single model path')
    eval_model.add_argument('--model-paths', nargs='+', help='Multiple model paths for comparison')
    
    evaluate_parser.add_argument('--model-names', nargs='+', help='Model names for comparison')
    evaluate_parser.add_argument('-v', '--vocab-path', required=True, help='Vocabulary file path')
    evaluate_parser.add_argument('-d', '--data-dir', required=True, help='IMDB dataset directory')
    evaluate_parser.add_argument('--config', help='Configuration file path')
    evaluate_parser.add_argument('--output-dir', help='Output directory')
    evaluate_parser.add_argument('--batch-size', type=int, help='Batch size')
    evaluate_parser.add_argument('--device', help='Device to use')
    evaluate_parser.add_argument('--show-plots', action='store_true', help='Show plots')
    evaluate_parser.add_argument('--no-visualizations', action='store_true', help='Skip visualizations')
    evaluate_parser.add_argument('--no-reports', action='store_true', help='Skip reports')
    evaluate_parser.add_argument('--log-level', help='Logging level')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_group = config_parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument('--create', choices=['training', 'inference', 'evaluation', 'quick', 'production'],
                             help='Create configuration file')
    config_group.add_argument('--validate', help='Validate configuration file')
    config_group.add_argument('--show', help='Show configuration file contents')
    
    config_parser.add_argument('--template', help='Template to use for creation')
    config_parser.add_argument('--output', help='Output path for created config')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    if args.command == 'setup':
        return setup_command(args)
    elif args.command == 'train':
        return train_command(args)
    elif args.command == 'predict':
        return predict_command(args)
    elif args.command == 'evaluate':
        return evaluate_command(args)
    elif args.command == 'config':
        return config_command(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == '__main__':
    sys.exit(main())