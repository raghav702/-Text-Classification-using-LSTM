#!/usr/bin/env python3
"""
Main evaluation script for LSTM Sentiment Classifier.

This script provides comprehensive model evaluation capabilities
including metrics calculation, visualization, and reporting.
"""

import argparse
import os
import sys
import json
import yaml
import logging
import torch
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from evaluation.evaluate_model import (
    load_test_data, evaluate_model_performance, calculate_comprehensive_metrics,
    create_visualizations, generate_reports, load_training_history, print_summary
)


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


def run_evaluation(args):
    """Run comprehensive model evaluation."""
    logger = logging.getLogger(__name__)
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info("Starting LSTM Sentiment Classifier evaluation")
    logger.info(f"Using device: {device}")
    
    try:
        # Load test data
        test_loader, preprocessor, test_texts, test_labels = load_test_data(
            args.data_dir,
            args.vocab_path,
            args.batch_size,
            args.num_workers
        )
        
        # Evaluate model
        y_true, y_pred, y_prob, model_info = evaluate_model_performance(
            args.model_path,
            args.vocab_path,
            test_loader,
            device
        )
        
        # Calculate metrics
        metrics = calculate_comprehensive_metrics(y_true, y_pred, y_prob)
        
        # Load training history if available
        training_history = load_training_history(args.model_path)
        
        # Print summary to console
        print_summary(metrics, model_info)
        
        # Create output directory
        if args.output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(args.output_dir, f"evaluation_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = None
        
        # Create visualizations
        figures = {}
        if not args.no_visualizations:
            figures = create_visualizations(
                metrics,
                training_history,
                output_dir,
                args.show_plots
            )
            logger.info(f"Created {len(figures)} visualizations")
        
        # Generate reports
        reports = {}
        if not args.no_reports:
            reports = generate_reports(
                metrics,
                model_info,
                None,  # training_summary not available from checkpoint
                output_dir
            )
            logger.info(f"Generated reports: {list(reports.keys())}")
        
        # Save evaluation configuration
        if output_dir:
            eval_config = {
                'evaluation_args': vars(args),
                'model_info': model_info,
                'evaluation_timestamp': datetime.now().isoformat(),
                'metrics_summary': {
                    'accuracy': metrics['basic_metrics']['accuracy'],
                    'f1_score': metrics['basic_metrics']['f1_score'],
                    'roc_auc': metrics['basic_metrics'].get('roc_auc'),
                    'optimal_threshold': metrics.get('optimal_threshold', {}).get('threshold')
                }
            }
            
            config_path = os.path.join(output_dir, 'evaluation_config.json')
            with open(config_path, 'w') as f:
                json.dump(eval_config, f, indent=2)
        
        # Print final results
        print(f"\nEvaluation completed successfully!")
        if output_dir:
            print(f"Results saved to: {output_dir}")
        
        return {
            'metrics': metrics,
            'figures': figures,
            'reports': reports,
            'output_dir': output_dir
        }
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


def compare_models(args):
    """Compare multiple models."""
    logger = logging.getLogger(__name__)
    
    if not args.model_paths or not args.model_names:
        raise ValueError("Model comparison requires --model-paths and --model-names")
    
    if len(args.model_paths) != len(args.model_names):
        raise ValueError("Number of model paths must match number of model names")
    
    logger.info(f"Comparing {len(args.model_paths)} models")
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Load test data once
    test_loader, preprocessor, test_texts, test_labels = load_test_data(
        args.data_dir,
        args.vocab_path,
        args.batch_size,
        args.num_workers
    )
    
    # Evaluate each model
    comparison_results = {}
    
    for model_path, model_name in zip(args.model_paths, args.model_names):
        logger.info(f"Evaluating model: {model_name}")
        
        try:
            # Evaluate model
            y_true, y_pred, y_prob, model_info = evaluate_model_performance(
                model_path,
                args.vocab_path,
                test_loader,
                device
            )
            
            # Calculate metrics
            metrics = calculate_comprehensive_metrics(y_true, y_pred, y_prob)
            
            comparison_results[model_name] = {
                'metrics': metrics,
                'model_info': model_info,
                'model_path': model_path
            }
            
            # Print individual summary
            print(f"\n{'='*20} {model_name} {'='*20}")
            print_summary(metrics, model_info)
            
        except Exception as e:
            logger.error(f"Failed to evaluate {model_name}: {e}")
            comparison_results[model_name] = {'error': str(e)}
    
    # Create comparison report
    if args.output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(args.output_dir, f"model_comparison_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save comparison results
        comparison_path = os.path.join(output_dir, 'model_comparison.json')
        with open(comparison_path, 'w') as f:
            json.dump(comparison_results, f, indent=2, default=str)
        
        # Create comparison summary
        summary_lines = []
        summary_lines.append("MODEL COMPARISON SUMMARY")
        summary_lines.append("=" * 50)
        
        for model_name, result in comparison_results.items():
            if 'error' in result:
                summary_lines.append(f"{model_name}: FAILED - {result['error']}")
            else:
                metrics = result['metrics']['basic_metrics']
                summary_lines.append(f"{model_name}:")
                summary_lines.append(f"  Accuracy: {metrics['accuracy']:.4f}")
                summary_lines.append(f"  F1-Score: {metrics['f1_score']:.4f}")
                summary_lines.append(f"  ROC AUC:  {metrics.get('roc_auc', 'N/A')}")
                summary_lines.append("")
        
        summary_path = os.path.join(output_dir, 'comparison_summary.txt')
        with open(summary_path, 'w') as f:
            f.write('\n'.join(summary_lines))
        
        print(f"\nComparison results saved to: {output_dir}")
    
    return comparison_results


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate LSTM Sentiment Classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration file
    parser.add_argument('--config', type=str,
                       help='Path to configuration file (YAML or JSON)')
    
    # Model arguments
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--model-path',
                           help='Path to trained model checkpoint')
    model_group.add_argument('--model-paths', nargs='+',
                           help='Paths to multiple models for comparison')
    
    parser.add_argument('--model-names', nargs='+',
                       help='Names for models (required for comparison)')
    parser.add_argument('--vocab-path', required=True,
                       help='Path to vocabulary file')
    parser.add_argument('--data-dir', required=True,
                       help='Path to IMDB dataset directory')
    
    # Evaluation options
    parser.add_argument('--output-dir', default='evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for evaluation')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto',
                       help='Device to use for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loader workers')
    
    # Visualization options
    parser.add_argument('--show-plots', action='store_true',
                       help='Display plots during evaluation')
    parser.add_argument('--no-visualizations', action='store_true',
                       help='Skip visualization generation')
    parser.add_argument('--no-reports', action='store_true',
                       help='Skip report generation')
    
    # System options
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    parser.add_argument('--log-file', help='Log file path')
    
    args = parser.parse_args()
    
    # Load configuration file if provided
    if args.config:
        config_data = load_config_file(args.config)
        
        # Update args with config file values (command line args take precedence)
        for key, value in config_data.items():
            if hasattr(args, key) and getattr(args, key) == parser.get_default(key):
                setattr(args, key, value)
    
    # Set up logging
    setup_logging(args.log_level, args.log_file)
    
    # Import torch here to avoid import issues
    import torch
    
    try:
        if args.model_paths:
            # Model comparison mode
            results = compare_models(args)
        else:
            # Single model evaluation mode
            results = run_evaluation(args)
        
        print("\nEvaluation completed successfully!")
        
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nEvaluation failed: {e}")
        raise


if __name__ == '__main__':
    main()