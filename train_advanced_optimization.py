#!/usr/bin/env python3
"""
Advanced optimization training script for LSTM Sentiment Classifier.

This script demonstrates advanced optimization strategies including:
- Multiple optimizer types comparison
- Advanced learning rate scheduling
- Gradient accumulation and enhanced clipping
- Convergence analysis and improved early stopping
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
from training.advanced_trainer import create_advanced_trainer, AdvancedTrainer
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


def load_and_preprocess_data(args, logger):
    """Load and preprocess the IMDB dataset."""
    logger.info("Loading IMDB dataset...")
    
    dataset = IMDBDataset(args.data_dir)
    train_texts, train_labels, test_texts, test_labels = dataset.load_data()
    
    logger.info(f"Loaded {len(train_texts)} training samples and {len(test_texts)} test samples")
    
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
    
    # Create train/validation split
    num_train = len(train_sequences)
    num_val = int(num_train * args.validation_split)
    num_train = num_train - num_val
    
    train_data = torch.utils.data.TensorDataset(train_sequences[:num_train], train_labels[:num_train])
    val_data = torch.utils.data.TensorDataset(train_sequences[num_train:], train_labels[num_train:])
    
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


def run_single_optimization_experiment(
    model, train_loader, val_loader, args, logger,
    optimizer_type: str, scheduler_type: str, config_name: str
):
    """Run a single optimization experiment."""
    logger.info(f"Running experiment: {config_name}")
    logger.info(f"Optimizer: {optimizer_type}, Scheduler: {scheduler_type}")
    
    # Create advanced trainer
    trainer = create_advanced_trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer_type=optimizer_type,
        scheduler_type=scheduler_type,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_type=args.gradient_clip_type,
        gradient_clip_value=args.gradient_clip_value,
        accumulation_steps=args.accumulation_steps,
        early_stopping_patience=args.early_stopping_patience,
        device=args.device,
        checkpoint_dir=os.path.join(args.checkpoint_dir, config_name),
        comparison_mode=args.comparison_mode
    )
    
    # Train model
    results = trainer.train(
        epochs=args.epochs,
        save_best=True,
        save_every=args.save_every,
        config_name=config_name
    )
    
    # Generate analysis plots
    plot_paths = trainer.plot_advanced_training_analysis(save_plots=True)
    
    logger.info(f"Experiment {config_name} completed")
    logger.info(f"Best validation accuracy: {results['final_metrics']['best_val_accuracy']:.2f}%")
    logger.info(f"Convergence analysis: {results['convergence_analysis']}")
    
    return results, trainer


def run_optimizer_comparison(model, train_loader, val_loader, args, logger):
    """Run comprehensive optimizer comparison."""
    logger.info("Starting comprehensive optimizer comparison...")
    
    # Define optimization configurations to compare
    optimization_configs = [
        ('adam', 'plateau', 'Adam_Plateau'),
        ('adamw', 'plateau', 'AdamW_Plateau'),
        ('adamw', 'cosine', 'AdamW_Cosine'),
        ('adamw', 'cosine_warm_restarts', 'AdamW_CosineWarmRestarts'),
        ('adamw', 'onecycle', 'AdamW_OneCycle'),
        ('sgd', 'plateau', 'SGD_Plateau'),
        ('sgd', 'cosine', 'SGD_Cosine'),
    ]
    
    results = {}
    trainers = {}
    
    for optimizer_type, scheduler_type, config_name in optimization_configs:
        try:
            # Create fresh model copy for each experiment
            model_copy = create_model(args, model.vocab_size, logger)
            model_copy.load_state_dict(model.state_dict())  # Copy weights
            
            # Run experiment
            result, trainer = run_single_optimization_experiment(
                model_copy, train_loader, val_loader, args, logger,
                optimizer_type, scheduler_type, config_name
            )
            
            results[config_name] = result
            trainers[config_name] = trainer
            
        except Exception as e:
            logger.error(f"Experiment {config_name} failed: {e}")
            continue
    
    # Generate comparison report
    if results and args.comparison_mode:
        # Use the first trainer's comparison framework
        first_trainer = next(iter(trainers.values()))
        if hasattr(first_trainer, 'optimizer_comparison') and first_trainer.optimizer_comparison:
            comparison_report = first_trainer.get_optimizer_comparison_report()
            
            if comparison_report:
                logger.info("Optimizer comparison completed!")
                logger.info(f"Best overall: {comparison_report['summary']['best_accuracy_config']}")
                logger.info(f"Fastest convergence: {comparison_report['summary']['best_convergence_config']}")
                
                # Save detailed comparison
                comparison_path = os.path.join(args.output_dir, 'optimizer_comparison_report.json')
                os.makedirs(os.path.dirname(comparison_path), exist_ok=True)
                with open(comparison_path, 'w') as f:
                    json.dump(comparison_report, f, indent=2)
                
                logger.info(f"Detailed comparison report saved to: {comparison_path}")
                
                return comparison_report
    
    return results


def save_experiment_results(results, args, logger):
    """Save comprehensive experiment results."""
    logger.info("Saving experiment results...")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(args.output_dir, f'advanced_optimization_results_{timestamp}.json')
    
    # Prepare results for JSON serialization
    serializable_results = {}
    for config_name, result in results.items():
        serializable_results[config_name] = {
            'final_metrics': result['final_metrics'],
            'convergence_analysis': result['convergence_analysis'],
            'gradient_analysis': result['gradient_analysis'],
            'configuration': result['configuration']
        }
    
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Experiment results saved to: {results_path}")
    
    return results_path


def main():
    """Main function for advanced optimization experiments."""
    parser = argparse.ArgumentParser(
        description="Advanced Optimization Training for LSTM Sentiment Classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Experiment mode
    parser.add_argument('--mode', choices=['single', 'comparison'], default='single',
                       help='Experiment mode: single optimization or comparison')
    parser.add_argument('--comparison-mode', action='store_true',
                       help='Enable optimizer comparison framework')
    
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
    
    # Advanced optimization arguments
    parser.add_argument('--optimizer-type', choices=['adam', 'adamw', 'sgd', 'rmsprop'],
                       default='adamw', help='Optimizer type')
    parser.add_argument('--scheduler-type', 
                       choices=['plateau', 'cosine', 'cosine_warm_restarts', 'onecycle', 'step', 'exponential', 'none'],
                       default='cosine_warm_restarts', help='Learning rate scheduler type')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay for regularization')
    
    # Gradient management arguments
    parser.add_argument('--gradient-clip-type', choices=['norm', 'value', 'none'],
                       default='norm', help='Type of gradient clipping')
    parser.add_argument('--gradient-clip-value', type=float, default=1.0,
                       help='Gradient clipping threshold')
    parser.add_argument('--accumulation-steps', type=int, default=1,
                       help='Number of gradient accumulation steps')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--early-stopping-patience', type=int, default=10,
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
    parser.add_argument('--output-dir', default='models/advanced_optimization',
                       help='Directory to save results')
    parser.add_argument('--checkpoint-dir', default='checkpoints/advanced_optimization',
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
    
    args = parser.parse_args()
    
    # Set up logging
    if args.log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_file = f"advanced_optimization_{timestamp}.log"
    
    logger = setup_logging(args.log_level, args.log_file, args.log_dir)
    
    # Set device
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    logger.info("Starting Advanced Optimization Experiments")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Random seed: {args.seed}")
    
    try:
        # Load and preprocess data
        train_loader, val_loader, preprocessor = load_and_preprocess_data(args, logger)
        
        # Create base model
        model = create_model(args, preprocessor.vocab_size, logger)
        
        # Initialize embeddings
        embedding_stats = initialize_embeddings(model, preprocessor, args, logger)
        
        # Run experiments based on mode
        if args.mode == 'single':
            # Single optimization experiment
            results, trainer = run_single_optimization_experiment(
                model, train_loader, val_loader, args, logger,
                args.optimizer_type, args.scheduler_type, 
                f"{args.optimizer_type}_{args.scheduler_type}"
            )
            
            # Save results
            results_path = save_experiment_results(
                {f"{args.optimizer_type}_{args.scheduler_type}": results}, 
                args, logger
            )
            
            logger.info("Single optimization experiment completed successfully!")
            logger.info(f"Best validation accuracy: {results['final_metrics']['best_val_accuracy']:.2f}%")
            
        elif args.mode == 'comparison':
            # Comprehensive optimizer comparison
            args.comparison_mode = True
            comparison_results = run_optimizer_comparison(
                model, train_loader, val_loader, args, logger
            )
            
            # Save results
            if isinstance(comparison_results, dict) and 'summary' in comparison_results:
                logger.info("Optimizer comparison completed successfully!")
                logger.info(f"Results saved to: {args.output_dir}")
            else:
                results_path = save_experiment_results(comparison_results, args, logger)
                logger.info(f"Comparison results saved to: {results_path}")
        
        print("\n" + "=" * 80)
        print("ADVANCED OPTIMIZATION EXPERIMENTS COMPLETED")
        print("=" * 80)
        print(f"Results saved to: {args.output_dir}")
        print(f"Logs saved to: {os.path.join(args.log_dir, args.log_file)}")
        print("=" * 80)
        
    except KeyboardInterrupt:
        logger.info("Experiments interrupted by user")
        print("\nExperiments interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Experiments failed: {e}")
        print(f"\nExperiments failed: {e}")
        raise


if __name__ == '__main__':
    main()