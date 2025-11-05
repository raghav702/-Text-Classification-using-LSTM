#!/usr/bin/env python3
"""
Complete training script for LSTM Sentiment Classifier.

This script provides a comprehensive training pipeline with all features:
- Data loading and preprocessing
- Model initialization with optional GloVe embeddings
- Training with validation and early stopping
- Checkpointing and model saving
- Training progress monitoring
"""

import argparse
import os
import sys
import logging
import json
from datetime import datetime
import torch
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lstm_model import LSTMClassifier
from data.imdb_loader import IMDBDataset
from data.text_preprocessor import TextPreprocessor
from training.trainer import create_trainer
from training.glove_loader import initialize_model_with_glove
from training.checkpoint_manager import create_checkpoint_manager, create_early_stopping
import config


def setup_logging(log_level: str = 'INFO', log_file: str = None):
    """Set up logging configuration."""
    level = getattr(logging, log_level.upper())
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def load_data(
    data_dir: str,
    batch_size: int = 64,
    max_vocab_size: int = 10000,
    max_length: int = 500,
    min_freq: int = 2,
    validation_split: float = 0.2,
    num_workers: int = 4
) -> tuple:
    """
    Load and preprocess IMDB dataset.
    
    Returns:
        Tuple of (train_loader, val_loader, preprocessor)
    """
    logger = logging.getLogger(__name__)
    logger.info("Loading IMDB dataset...")
    
    # Load dataset
    dataset = IMDBDataset(data_dir)
    train_texts, train_labels, test_texts, test_labels = dataset.load_data()
    
    logger.info(f"Loaded {len(train_texts)} training samples and {len(test_texts)} test samples")
    
    # Create text preprocessor
    preprocessor = TextPreprocessor(
        max_vocab_size=max_vocab_size,
        min_freq=min_freq,
        max_length=max_length
    )
    
    # Build vocabulary from training texts
    logger.info("Building vocabulary...")
    vocab_stats = preprocessor.build_vocabulary(train_texts)
    logger.info(f"Vocabulary built: {len(vocab_stats)} unique words")
    
    # Preprocess texts
    logger.info("Preprocessing texts...")
    train_sequences = preprocessor.preprocess_texts(train_texts, fit_vocabulary=False)
    test_sequences = preprocessor.preprocess_texts(test_texts, fit_vocabulary=False)
    
    # Convert labels to tensors
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_labels = torch.tensor(test_labels, dtype=torch.long)
    
    # Create train/validation split
    num_train = len(train_sequences)
    num_val = int(num_train * validation_split)
    num_train = num_train - num_val
    
    # Split data
    train_data = torch.utils.data.TensorDataset(train_sequences[:num_train], train_labels[:num_train])
    val_data = torch.utils.data.TensorDataset(train_sequences[num_train:], train_labels[num_train:])
    
    # Create data loaders
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    logger.info(f"Data loaders created: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    return train_loader, val_loader, preprocessor


def create_model(
    vocab_size: int,
    embedding_dim: int = 300,
    hidden_dim: int = 128,
    output_dim: int = 1,
    n_layers: int = 2,
    dropout: float = 0.3,
    bidirectional: bool = True
) -> LSTMClassifier:
    """Create LSTM model with specified configuration."""
    logger = logging.getLogger(__name__)
    
    model = LSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        n_layers=n_layers,
        dropout=dropout,
        bidirectional=bidirectional
    )
    
    model_info = model.get_model_info()
    logger.info(f"Model created with {model_info['total_parameters']:,} parameters")
    
    return model


def initialize_embeddings(
    model: LSTMClassifier,
    preprocessor: TextPreprocessor,
    use_glove: bool = True,
    glove_corpus: str = '6B',
    glove_dim: str = '300d',
    freeze_embeddings: bool = False,
    glove_cache_dir: str = 'data/glove'
) -> dict:
    """Initialize model embeddings with GloVe if requested."""
    logger = logging.getLogger(__name__)
    
    if not use_glove:
        logger.info("Using random embedding initialization")
        return {'method': 'random'}
    
    try:
        logger.info(f"Initializing embeddings with GloVe {glove_corpus}.{glove_dim}")
        
        embedding_stats = initialize_model_with_glove(
            model=model,
            preprocessor=preprocessor,
            corpus=glove_corpus,
            dimension=glove_dim,
            freeze_embeddings=freeze_embeddings,
            cache_dir=glove_cache_dir
        )
        
        logger.info(f"GloVe initialization complete. Coverage: {embedding_stats['coverage_ratio']:.2%}")
        
        return embedding_stats
        
    except Exception as e:
        logger.warning(f"GloVe initialization failed: {e}")
        logger.info("Falling back to random embedding initialization")
        return {'method': 'random', 'error': str(e)}


def train_model(
    model: LSTMClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 20,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    gradient_clip: float = 1.0,
    scheduler_type: str = 'plateau',
    early_stopping_patience: int = 5,
    checkpoint_dir: str = 'checkpoints',
    save_every: int = None,
    device: str = None
) -> dict:
    """Train the model with specified configuration."""
    logger = logging.getLogger(__name__)
    
    # Create trainer
    trainer = create_trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        gradient_clip=gradient_clip,
        scheduler_type=scheduler_type,
        device=device,
        checkpoint_dir=checkpoint_dir
    )
    
    logger.info(f"Starting training for {epochs} epochs")
    logger.info(f"Device: {trainer.device}")
    logger.info(f"Early stopping patience: {early_stopping_patience}")
    
    # Train model
    history = trainer.train(
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        save_best=True,
        save_every=save_every
    )
    
    # Get training summary
    summary = trainer.get_training_summary()
    
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {summary['performance_metrics']['best_val_loss']:.6f}")
    logger.info(f"Best validation accuracy: {summary['performance_metrics']['best_val_accuracy']:.2f}%")
    
    return history, summary


def save_training_artifacts(
    model: LSTMClassifier,
    preprocessor: TextPreprocessor,
    history: dict,
    summary: dict,
    embedding_stats: dict,
    output_dir: str,
    model_name: str = 'lstm_sentiment_model'
):
    """Save all training artifacts."""
    logger = logging.getLogger(__name__)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, f'{model_name}.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model.get_model_info(),
        'training_history': history,
        'training_summary': summary,
        'embedding_stats': embedding_stats,
        'timestamp': datetime.now().isoformat()
    }, model_path)
    
    # Save vocabulary
    vocab_path = os.path.join(output_dir, f'{model_name}_vocabulary.pth')
    preprocessor.save_vocabulary(vocab_path)
    
    # Save training configuration and results
    config_path = os.path.join(output_dir, f'{model_name}_config.json')
    config_data = {
        'model_config': model.get_model_info(),
        'training_summary': summary,
        'embedding_stats': embedding_stats,
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
    
    logger.info(f"Training artifacts saved to {output_dir}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Vocabulary: {vocab_path}")
    logger.info(f"Configuration: {config_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train LSTM Sentiment Classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--data-dir', default='data/imdb',
                       help='Path to IMDB dataset directory')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--max-vocab-size', type=int, default=10000,
                       help='Maximum vocabulary size')
    parser.add_argument('--max-length', type=int, default=500,
                       help='Maximum sequence length')
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
    parser.add_argument('--checkpoint-dir', default='checkpoints',
                       help='Directory for training checkpoints')
    parser.add_argument('--save-every', type=int,
                       help='Save checkpoint every N epochs')
    
    # System arguments
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto',
                       help='Device to use for training')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    parser.add_argument('--log-file', help='Log file path')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info("Starting LSTM Sentiment Classifier training")
    logger.info(f"Arguments: {vars(args)}")
    logger.info(f"Using device: {device}")
    
    try:
        # Load data
        train_loader, val_loader, preprocessor = load_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            max_vocab_size=args.max_vocab_size,
            max_length=args.max_length,
            validation_split=args.validation_split,
            num_workers=args.num_workers
        )
        
        # Create model
        model = create_model(
            vocab_size=preprocessor.vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            dropout=args.dropout,
            bidirectional=args.bidirectional
        )
        
        # Initialize embeddings
        embedding_stats = initialize_embeddings(
            model=model,
            preprocessor=preprocessor,
            use_glove=args.use_glove,
            glove_corpus=args.glove_corpus,
            glove_dim=args.glove_dim,
            freeze_embeddings=args.freeze_embeddings,
            glove_cache_dir=args.glove_cache_dir
        )
        
        # Train model
        scheduler_type = None if args.scheduler == 'none' else args.scheduler
        
        history, summary = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            gradient_clip=args.gradient_clip,
            scheduler_type=scheduler_type,
            early_stopping_patience=args.early_stopping_patience,
            checkpoint_dir=args.checkpoint_dir,
            save_every=args.save_every,
            device=device
        )
        
        # Save artifacts
        save_training_artifacts(
            model=model,
            preprocessor=preprocessor,
            history=history,
            summary=summary,
            embedding_stats=embedding_stats,
            output_dir=args.output_dir,
            model_name=args.model_name
        )
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == '__main__':
    main()