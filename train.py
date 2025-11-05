#!/usr/bin/env python3
"""
Main training script for LSTM Sentiment Classifier.

This script orchestrates the complete training workflow including data loading,
model initialization, training with GloVe embeddings, and result saving.
"""

import argparse
import os
import sys
import logging
import json
import torch
from datetime import datetime
import yaml

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.lstm_model import LSTMClassifier
from data.imdb_loader import IMDBDataset
from data.text_preprocessor import TextPreprocessor
from training.trainer import create_trainer
from training.glove_loader import initialize_model_with_glove
from training.checkpoint_manager import create_checkpoint_manager
import config


def setup_logging(log_level: str = 'INFO', log_file: str = None, log_dir: str = 'logs'):
    """Set up comprehensive logging configuration."""
    # Create logs directory
    if log_file and log_dir:
        os.makedirs(log_dir, exist_ok=True)
        if not os.path.isabs(log_file):
            log_file = os.path.join(log_dir, log_file)
    
    # Set logging level
    level = getattr(logging, log_level.upper())
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        print(f"Logging to file: {log_file}")
    
    return logger


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


def save_config(config_dict: dict, output_path: str):
    """Save configuration to file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        if output_path.endswith('.yaml') or output_path.endswith('.yml'):
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        else:
            json.dump(config_dict, f, indent=2)


def load_and_preprocess_data(args, logger):
    """Load and preprocess the IMDB dataset."""
    logger.info("Loading IMDB dataset...")
    
    # Load dataset
    dataset = IMDBDataset(args.data_dir)
    train_texts, train_labels, test_texts, test_labels = dataset.load_data()
    
    logger.info(f"Loaded {len(train_texts)} training samples and {len(test_texts)} test samples")
    
    # Create text preprocessor
    preprocessor = TextPreprocessor(
        max_vocab_size=args.max_vocab_size,
        min_freq=args.min_word_freq,
        max_length=args.max_sequence_length
    )
    
    # Build vocabulary from training texts
    logger.info("Building vocabulary...")
    vocab_stats = preprocessor.build_vocabulary(train_texts)
    logger.info(f"Vocabulary built: {len(vocab_stats)} unique words, final vocab size: {preprocessor.vocab_size}")
    
    # Preprocess texts
    logger.info("Preprocessing texts...")
    train_sequences = preprocessor.preprocess_texts(train_texts, fit_vocabulary=False)
    
    # Convert labels to tensors
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    
    # Create train/validation split
    num_train = len(train_sequences)
    num_val = int(num_train * args.validation_split)
    num_train = num_train - num_val
    
    # Split data
    train_data = torch.utils.data.TensorDataset(train_sequences[:num_train], train_labels[:num_train])
    val_data = torch.utils.data.TensorDataset(train_sequences[num_train:], train_labels[num_train:])
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    logger.info(f"Data loaders created: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    return train_loader, val_loader, preprocessor


def create_model(args, vocab_size, logger):
    """Create and initialize the LSTM model."""
    logger.info("Creating LSTM model...")
    
    model = LSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        output_dim=1,
        n_layers=args.n_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional
    )
    
    model_info = model.get_model_info()
    logger.info(f"Model created with {model_info['total_parameters']:,} parameters")
    
    return model


def initialize_embeddings(model, preprocessor, args, logger):
    """Initialize model embeddings with GloVe if requested."""
    if not args.use_glove:
        logger.info("Using random embedding initialization")
        return {'method': 'random'}
    
    try:
        logger.info(f"Initializing embeddings with GloVe {args.glove_corpus}.{args.glove_dim}")
        
        embedding_stats = initialize_model_with_glove(
            model=model,
            preprocessor=preprocessor,
            corpus=args.glove_corpus,
            dimension=args.glove_dim,
            freeze_embeddings=args.freeze_embeddings,
            cache_dir=args.glove_cache_dir
        )
        
        logger.info(f"GloVe initialization complete. Coverage: {embedding_stats['coverage_ratio']:.2%}")
        return embedding_stats
        
    except Exception as e:
        logger.warning(f"GloVe initialization failed: {e}")
        logger.info("Falling back to random embedding initialization")
        return {'method': 'random', 'error': str(e)}


def train_model(model, train_loader, val_loader, args, logger):
    """Train the model with specified configuration."""
    logger.info("Setting up trainer...")
    
    # Create trainer
    trainer = create_trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        scheduler_type=args.scheduler,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir
    )
    
    logger.info(f"Training configuration:")
    logger.info(f"  Device: {trainer.device}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Early stopping patience: {args.early_stopping_patience}")
    
    # Train model
    logger.info("Starting training...")
    history = trainer.train(
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        save_best=True,
        save_every=args.save_every
    )
    
    # Get training summary
    summary = trainer.get_training_summary()
    
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {summary['performance_metrics']['best_val_loss']:.6f}")
    logger.info(f"Best validation accuracy: {summary['performance_metrics']['best_val_accuracy']:.2f}%")
    logger.info(f"Total training time: {history['total_time']:.2f} seconds")
    
    return history, summary, trainer


def save_artifacts(model, preprocessor, history, summary, embedding_stats, args, logger):
    """Save all training artifacts."""
    logger.info("Saving training artifacts...")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate model name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{args.model_name}_{timestamp}" if args.add_timestamp else args.model_name
    
    # Save model
    model_path = os.path.join(args.output_dir, f'{model_name}.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model.get_model_info(),
        'training_history': history,
        'training_summary': summary,
        'embedding_stats': embedding_stats,
        'training_args': vars(args),
        'timestamp': datetime.now().isoformat()
    }, model_path)
    
    # Save vocabulary
    vocab_path = os.path.join(args.output_dir, f'{model_name}_vocabulary.pth')
    preprocessor.save_vocabulary(vocab_path)
    
    # Save training configuration
    config_path = os.path.join(args.output_dir, f'{model_name}_config.json')
    config_data = {
        'model_config': model.get_model_info(),
        'training_summary': summary,
        'embedding_stats': embedding_stats,
        'training_args': vars(args),
        'training_history': {
            'epochs': len(history['train_losses']),
            'best_val_loss': history['best_val_loss'],
            'best_val_accuracy': history['best_val_accuracy'],
            'total_time': history['total_time']
        },
        'vocab_info': preprocessor.get_vocab_info(),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    # Save training history as CSV for easy plotting
    history_path = os.path.join(args.output_dir, f'{model_name}_history.csv')
    import pandas as pd
    
    history_df = pd.DataFrame({
        'epoch': range(1, len(history['train_losses']) + 1),
        'train_loss': history['train_losses'],
        'val_loss': history['val_losses'],
        'val_accuracy': history['val_accuracies'],
        'learning_rate': history['learning_rates']
    })
    history_df.to_csv(history_path, index=False)
    
    logger.info(f"Training artifacts saved:")
    logger.info(f"  Model: {model_path}")
    logger.info(f"  Vocabulary: {vocab_path}")
    logger.info(f"  Configuration: {config_path}")
    logger.info(f"  History: {history_path}")
    
    return {
        'model_path': model_path,
        'vocab_path': vocab_path,
        'config_path': config_path,
        'history_path': history_path
    }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train LSTM Sentiment Classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration file
    parser.add_argument('--config', type=str,
                       help='Path to configuration file (YAML or JSON)')
    
    # Data arguments
    parser.add_argument('--data-dir', default='data/imdb',
                       help='Path to IMDB dataset directory')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--max-vocab-size', type=int, default=10000,
                       help='Maximum vocabulary size')
    parser.add_argument('--max-sequence-length', type=int, default=500,
                       help='Maximum sequence length')
    parser.add_argument('--min-word-freq', type=int, default=2,
                       help='Minimum word frequency for vocabulary')
    parser.add_argument('--validation-split', type=float, default=0.2,
                       help='Validation split ratio')
    
    # Model arguments
    parser.add_argument('--embedding-dim', type=int, default=300,
                       help='Embedding dimension')
    parser.add_argument('--hidden-dim', type=int, default=128,
                       help='Hidden dimension')
    parser.add_argument('--n-layers', type=int, default=2,
                       help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--bidirectional', action='store_true', default=True,
                       help='Use bidirectional LSTM')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='Weight decay for regularization')
    parser.add_argument('--gradient-clip', type=float, default=1.0,
                       help='Gradient clipping threshold')
    parser.add_argument('--scheduler', choices=['plateau', 'step', 'cosine', 'none'],
                       default='plateau', help='Learning rate scheduler')
    parser.add_argument('--early-stopping-patience', type=int, default=5,
                       help='Early stopping patience')
    
    # GloVe arguments
    parser.add_argument('--use-glove', action='store_true', default=True,
                       help='Use GloVe embeddings')
    parser.add_argument('--glove-corpus', choices=['6B', '42B', '840B'], default='6B',
                       help='GloVe corpus to use')
    parser.add_argument('--glove-dim', choices=['50d', '100d', '200d', '300d'], default='300d',
                       help='GloVe embedding dimension')
    parser.add_argument('--freeze-embeddings', action='store_true',
                       help='Freeze embedding weights during training')
    parser.add_argument('--glove-cache-dir', default='data/glove',
                       help='Directory to cache GloVe files')
    
    # Output arguments
    parser.add_argument('--output-dir', default='models',
                       help='Directory to save trained model')
    parser.add_argument('--model-name', default='lstm_sentiment_model',
                       help='Name for saved model files')
    parser.add_argument('--add-timestamp', action='store_true',
                       help='Add timestamp to model name')
    parser.add_argument('--checkpoint-dir', default='checkpoints',
                       help='Directory for training checkpoints')
    parser.add_argument('--save-every', type=int,
                       help='Save checkpoint every N epochs')
    
    # System arguments
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto',
                       help='Device to use for training')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Logging arguments
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    parser.add_argument('--log-file', help='Log file name (saved in logs directory)')
    parser.add_argument('--log-dir', default='logs', help='Directory for log files')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load configuration file if provided
    if args.config:
        config_data = load_config_file(args.config)
        
        # Update args with config file values (command line args take precedence)
        for key, value in config_data.items():
            if not hasattr(args, key) or getattr(args, key) == parser.get_default(key):
                setattr(args, key, value)
    
    # Set up logging
    if args.log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_file = f"training_{timestamp}.log"
    
    logger = setup_logging(args.log_level, args.log_file, args.log_dir)
    
    # Set device
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    logger.info("Starting LSTM Sentiment Classifier training")
    logger.info(f"Configuration: {json.dumps(vars(args), indent=2)}")
    logger.info(f"Using device: {args.device}")
    logger.info(f"Random seed: {args.seed}")
    
    try:
        # Load and preprocess data
        train_loader, val_loader, preprocessor = load_and_preprocess_data(args, logger)
        
        # Create model
        model = create_model(args, preprocessor.vocab_size, logger)
        
        # Initialize embeddings
        embedding_stats = initialize_embeddings(model, preprocessor, args, logger)
        
        # Train model
        history, summary, trainer = train_model(model, train_loader, val_loader, args, logger)
        
        # Save artifacts
        saved_paths = save_artifacts(model, preprocessor, history, summary, embedding_stats, args, logger)
        
        # Save final configuration
        final_config_path = os.path.join(args.output_dir, 'final_training_config.yaml')
        save_config(vars(args), final_config_path)
        
        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {saved_paths['model_path']}")
        logger.info(f"Final configuration saved to: {final_config_path}")
        
        # Print final summary
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Best Validation Loss: {summary['performance_metrics']['best_val_loss']:.6f}")
        print(f"Best Validation Accuracy: {summary['performance_metrics']['best_val_accuracy']:.2f}%")
        print(f"Total Training Time: {history['total_time']:.2f} seconds")
        print(f"Model saved to: {saved_paths['model_path']}")
        print("=" * 80)
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        print("\nTraining interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"\nTraining failed: {e}")
        raise


if __name__ == '__main__':
    main()