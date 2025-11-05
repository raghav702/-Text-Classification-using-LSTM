#!/usr/bin/env python3
"""
Hyperparameter optimization script for LSTM Sentiment Classifier.

This script demonstrates comprehensive hyperparameter tuning including:
- Grid search and random search
- Bayesian optimization (if scikit-optimize available)
- Cross-validation for robust evaluation
- Automated logging and comparison of results
"""

import argparse
import os
import sys
import logging
import json
import torch
from datetime import datetime
import numpy as np

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.lstm_model import LSTMClassifier
from data.imdb_loader import IMDBDataset
from data.text_preprocessor import TextPreprocessor
from training.hyperparameter_optimizer import (
    HyperparameterSpace,
    CrossValidator,
    HyperparameterOptimizer,
    create_default_hyperparameter_space
)
from training.glove_loader import initialize_model_with_glove
import config


def setup_logging(log_level: str = 'INFO', log_file: str = None, log_dir: str = 'logs'):
    """Set up comprehensive logging configuration."""
    if log_file and log_dir:
        os.makedirs(log_dir, exist_ok=True)
        if not os.path.isabs(log_file):
            log_file = os.path.join(log_dir, log_file)
    
    level = getattr(logging, log_level.upper())
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger()
    logger.setLevel(level)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        print(f"Logging to file: {log_file}")
    
    return logger


def load_and_prepare_data(args, logger):
    """Load and prepare the dataset for hyperparameter optimization."""
    logger.info("Loading IMDB dataset...")
    
    dataset = IMDBDataset(args.data_dir)
    train_texts, train_labels, test_texts, test_labels = dataset.load_data()
    
    logger.info(f"Loaded {len(train_texts)} training samples and {len(test_texts)} test samples")
    
    # Use a subset for faster hyperparameter optimization if requested
    if args.subset_size and args.subset_size < len(train_texts):
        logger.info(f"Using subset of {args.subset_size} samples for optimization")
        indices = np.random.choice(len(train_texts), args.subset_size, replace=False)
        train_texts = [train_texts[i] for i in indices]
        train_labels = [train_labels[i] for i in indices]
    
    # Create text preprocessor
    preprocessor = TextPreprocessor(
        max_vocab_size=args.max_vocab_size,
        min_freq=args.min_word_freq,
        max_length=args.max_sequence_length
    )
    
    logger.info("Building vocabulary...")
    vocab_stats = preprocessor.build_vocabulary(train_texts)
    logger.info(f"Vocabulary built: {len(vocab_stats)} unique words, final vocab size: {preprocessor.vocab_size}")
    
    logger.info("Preprocessing texts...")
    train_sequences = preprocessor.preprocess_texts(train_texts, fit_vocabulary=False)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    
    # Create dataset
    dataset = torch.utils.data.TensorDataset(train_sequences, train_labels)
    
    logger.info(f"Dataset prepared: {len(dataset)} samples")
    
    return dataset, train_labels, preprocessor


def create_model_factory(preprocessor, args, logger):
    """Create a model factory function for hyperparameter optimization."""
    
    def model_factory(hyperparams):
        """Factory function to create model with given hyperparameters."""
        # Extract model parameters from hyperparams
        model_params = {
            'vocab_size': preprocessor.vocab_size,
            'embedding_dim': hyperparams.get('embedding_dim', args.embedding_dim),
            'hidden_dim': hyperparams.get('hidden_dim', args.hidden_dim),
            'output_dim': 1,
            'n_layers': hyperparams.get('n_layers', args.n_layers),
            'dropout': hyperparams.get('dropout', args.dropout),
            'bidirectional': args.bidirectional
        }
        
        model = LSTMClassifier(**model_params)
        
        # Initialize embeddings if requested
        if args.use_glove:
            try:
                initialize_model_with_glove(
                    model=model,
                    preprocessor=preprocessor,
                    corpus=args.glove_corpus,
                    dimension=model_params['embedding_dim'],
                    freeze_embeddings=args.freeze_embeddings,
                    cache_dir=args.glove_cache_dir
                )
            except Exception as e:
                logger.warning(f"GloVe initialization failed: {e}")
        
        return model
    
    return model_factory


def create_custom_hyperparameter_space(args) -> HyperparameterSpace:
    """Create custom hyperparameter space based on arguments."""
    space = HyperparameterSpace()
    
    # Model architecture parameters
    if args.tune_hidden_dim:
        space.add_parameter('hidden_dim', 'categorical', [64, 128, 256, 512])
    
    if args.tune_n_layers:
        space.add_parameter('n_layers', 'categorical', [1, 2, 3])
    
    if args.tune_dropout:
        space.add_parameter('dropout', 'real', (0.1, 0.6))
    
    if args.tune_embedding_dim:
        space.add_parameter('embedding_dim', 'categorical', [100, 200, 300])
    
    # Training parameters
    if args.tune_learning_rate:
        space.add_parameter('learning_rate', 'real', (1e-4, 1e-2), log_scale=True)
    
    if args.tune_weight_decay:
        space.add_parameter('weight_decay', 'real', (1e-6, 1e-3), log_scale=True)
    
    if args.tune_batch_size:
        space.add_parameter('batch_size', 'categorical', [32, 64, 128])
    
    # Optimization parameters
    if args.tune_optimizer:
        space.add_parameter('optimizer_type', 'categorical', ['adam', 'adamw', 'sgd'])
    
    if args.tune_scheduler:
        space.add_parameter('scheduler_type', 'categorical', ['plateau', 'cosine', 'cosine_warm_restarts'])
    
    # Gradient management
    if args.tune_gradient_clip:
        space.add_parameter('gradient_clip_value', 'real', (0.5, 2.0))
    
    # If no parameters specified, use default space
    if not space.spaces:
        logger.warning("No hyperparameters specified for tuning, using default space")
        space = create_default_hyperparameter_space()
    
    return space


def run_hyperparameter_optimization(args, logger):
    """Run the hyperparameter optimization experiment."""
    logger.info("Starting hyperparameter optimization experiment")
    
    # Load and prepare data
    dataset, labels, preprocessor = load_and_prepare_data(args, logger)
    
    # Create model factory
    model_factory = create_model_factory(preprocessor, args, logger)
    
    # Create hyperparameter space
    if args.use_default_space:
        hyperparameter_space = create_default_hyperparameter_space()
    else:
        hyperparameter_space = create_custom_hyperparameter_space(args)
    
    logger.info(f"Hyperparameter space: {list(hyperparameter_space.spaces.keys())}")
    
    # Create cross-validator
    cross_validator = CrossValidator(
        n_folds=args.cv_folds,
        stratified=args.stratified_cv,
        shuffle=True,
        random_state=args.seed
    )
    
    # Create optimizer
    optimizer = HyperparameterOptimizer(
        hyperparameter_space=hyperparameter_space,
        cross_validator=cross_validator,
        save_dir=args.output_dir,
        n_jobs=args.n_jobs
    )
    
    # Training configuration
    training_config = {
        'optimizer_type': args.optimizer_type,
        'scheduler_type': args.scheduler_type,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'gradient_clip_type': args.gradient_clip_type,
        'gradient_clip_value': args.gradient_clip_value,
        'accumulation_steps': args.accumulation_steps,
        'early_stopping_patience': args.early_stopping_patience,
        'epochs': args.epochs,
        'checkpoint_dir': os.path.join(args.output_dir, 'checkpoints')
    }
    
    # Run optimization based on method
    if args.method == 'grid':
        logger.info("Running grid search optimization")
        results = optimizer.grid_search(
            model_factory=model_factory,
            dataset=dataset,
            labels=labels,
            training_config=training_config,
            device=args.device
        )
    
    elif args.method == 'random':
        logger.info(f"Running random search optimization with {args.n_iter} iterations")
        results = optimizer.random_search(
            model_factory=model_factory,
            dataset=dataset,
            labels=labels,
            training_config=training_config,
            n_iter=args.n_iter,
            device=args.device
        )
    
    elif args.method == 'bayesian':
        logger.info(f"Running Bayesian optimization with {args.n_calls} calls")
        try:
            results = optimizer.bayesian_optimization(
                model_factory=model_factory,
                dataset=dataset,
                labels=labels,
                training_config=training_config,
                n_calls=args.n_calls,
                n_initial_points=args.n_initial_points,
                device=args.device
            )
        except ImportError as e:
            logger.error(f"Bayesian optimization failed: {e}")
            logger.info("Falling back to random search")
            results = optimizer.random_search(
                model_factory=model_factory,
                dataset=dataset,
                labels=labels,
                training_config=training_config,
                n_iter=args.n_calls,
                device=args.device
            )
    
    else:
        raise ValueError(f"Unknown optimization method: {args.method}")
    
    # Generate analysis plots
    plot_paths = optimizer.plot_optimization_results(results, save_plots=True)
    
    # Print results summary
    logger.info("Hyperparameter optimization completed!")
    logger.info(f"Best score: {results['best_score']:.2f}%")
    logger.info(f"Best hyperparameters: {results['best_hyperparams']}")
    logger.info(f"Number of evaluations: {results['n_evaluations']}")
    logger.info(f"Total optimization time: {results['optimization_time']:.2f} seconds")
    
    return results


def main():
    """Main function for hyperparameter optimization."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter Optimization for LSTM Sentiment Classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Optimization method
    parser.add_argument('--method', choices=['grid', 'random', 'bayesian'], default='random',
                       help='Hyperparameter optimization method')
    parser.add_argument('--n-iter', type=int, default=50,
                       help='Number of iterations for random search')
    parser.add_argument('--n-calls', type=int, default=50,
                       help='Number of calls for Bayesian optimization')
    parser.add_argument('--n-initial-points', type=int, default=10,
                       help='Number of initial random points for Bayesian optimization')
    
    # Cross-validation settings
    parser.add_argument('--cv-folds', type=int, default=3,
                       help='Number of cross-validation folds')
    parser.add_argument('--stratified-cv', action='store_true', default=True,
                       help='Use stratified cross-validation')
    
    # Data arguments
    parser.add_argument('--data-dir', default='data/imdb',
                       help='Path to IMDB dataset directory')
    parser.add_argument('--subset-size', type=int,
                       help='Use subset of data for faster optimization')
    parser.add_argument('--max-vocab-size', type=int, default=10000,
                       help='Maximum vocabulary size')
    parser.add_argument('--max-sequence-length', type=int, default=500,
                       help='Maximum sequence length')
    parser.add_argument('--min-word-freq', type=int, default=2,
                       help='Minimum word frequency for vocabulary')
    
    # Hyperparameter space configuration
    parser.add_argument('--use-default-space', action='store_true',
                       help='Use default hyperparameter space')
    parser.add_argument('--tune-hidden-dim', action='store_true',
                       help='Include hidden dimension in optimization')
    parser.add_argument('--tune-n-layers', action='store_true',
                       help='Include number of layers in optimization')
    parser.add_argument('--tune-dropout', action='store_true',
                       help='Include dropout rate in optimization')
    parser.add_argument('--tune-embedding-dim', action='store_true',
                       help='Include embedding dimension in optimization')
    parser.add_argument('--tune-learning-rate', action='store_true',
                       help='Include learning rate in optimization')
    parser.add_argument('--tune-weight-decay', action='store_true',
                       help='Include weight decay in optimization')
    parser.add_argument('--tune-batch-size', action='store_true',
                       help='Include batch size in optimization')
    parser.add_argument('--tune-optimizer', action='store_true',
                       help='Include optimizer type in optimization')
    parser.add_argument('--tune-scheduler', action='store_true',
                       help='Include scheduler type in optimization')
    parser.add_argument('--tune-gradient-clip', action='store_true',
                       help='Include gradient clipping in optimization')
    
    # Default model parameters (used when not tuning)
    parser.add_argument('--embedding-dim', type=int, default=300,
                       help='Default embedding dimension')
    parser.add_argument('--hidden-dim', type=int, default=128,
                       help='Default hidden dimension')
    parser.add_argument('--n-layers', type=int, default=2,
                       help='Default number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Default dropout rate')
    parser.add_argument('--bidirectional', action='store_true', default=True,
                       help='Use bidirectional LSTM')
    
    # Default training parameters
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs per evaluation')
    parser.add_argument('--optimizer-type', choices=['adam', 'adamw', 'sgd'],
                       default='adamw', help='Default optimizer type')
    parser.add_argument('--scheduler-type', 
                       choices=['plateau', 'cosine', 'cosine_warm_restarts', 'none'],
                       default='plateau', help='Default scheduler type')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Default learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Default weight decay')
    parser.add_argument('--gradient-clip-type', choices=['norm', 'value', 'none'],
                       default='norm', help='Default gradient clipping type')
    parser.add_argument('--gradient-clip-value', type=float, default=1.0,
                       help='Default gradient clipping value')
    parser.add_argument('--accumulation-steps', type=int, default=1,
                       help='Default gradient accumulation steps')
    parser.add_argument('--early-stopping-patience', type=int, default=5,
                       help='Default early stopping patience')
    
    # GloVe arguments
    parser.add_argument('--use-glove', action='store_true', default=True,
                       help='Use GloVe embeddings')
    parser.add_argument('--glove-corpus', choices=['6B', '42B', '840B'], default='6B',
                       help='GloVe corpus to use')
    parser.add_argument('--freeze-embeddings', action='store_true',
                       help='Freeze embedding weights during training')
    parser.add_argument('--glove-cache-dir', default='data/glove',
                       help='Directory to cache GloVe files')
    
    # System arguments
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto',
                       help='Device to use for training')
    parser.add_argument('--n-jobs', type=int, default=1,
                       help='Number of parallel jobs (1 for sequential)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Output arguments
    parser.add_argument('--output-dir', default='hyperparameter_optimization',
                       help='Directory to save optimization results')
    
    # Logging arguments
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    parser.add_argument('--log-file', help='Log file name (saved in logs directory)')
    parser.add_argument('--log-dir', default='logs', help='Directory for log files')
    
    args = parser.parse_args()
    
    # Set up logging
    if args.log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_file = f"hyperparameter_optimization_{timestamp}.log"
    
    logger = setup_logging(args.log_level, args.log_file, args.log_dir)
    
    # Set device
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    logger.info("Starting Hyperparameter Optimization")
    logger.info(f"Method: {args.method}")
    logger.info(f"Device: {args.device}")
    logger.info(f"CV Folds: {args.cv_folds}")
    logger.info(f"Random seed: {args.seed}")
    
    try:
        # Run hyperparameter optimization
        results = run_hyperparameter_optimization(args, logger)
        
        print("\n" + "=" * 80)
        print("HYPERPARAMETER OPTIMIZATION COMPLETED")
        print("=" * 80)
        print(f"Method: {results['method']}")
        print(f"Best Score: {results['best_score']:.2f}%")
        print(f"Best Hyperparameters:")
        for param, value in results['best_hyperparams'].items():
            print(f"  {param}: {value}")
        print(f"Number of Evaluations: {results['n_evaluations']}")
        print(f"Total Time: {results['optimization_time']:.2f} seconds")
        print(f"Results saved to: {args.output_dir}")
        print("=" * 80)
        
    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user")
        print("\nOptimization interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        print(f"\nOptimization failed: {e}")
        raise


if __name__ == '__main__':
    main()