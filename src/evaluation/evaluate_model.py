#!/usr/bin/env python3
"""
Comprehensive model evaluation script for LSTM Sentiment Classifier.

This script provides complete evaluation functionality including
metrics calculation, visualization generation, and report creation.
"""

import argparse
import os
import sys
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader
from datetime import datetime
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lstm_model import LSTMClassifier
from data.imdb_loader import IMDBDataset
from data.text_preprocessor import TextPreprocessor
from inference.inference_engine import InferenceEngine
from evaluation.metrics import MetricsCalculator
from evaluation.visualization import EvaluationVisualizer, EvaluationReporter
from evaluation.advanced_metrics import AdvancedMetricsCalculator
from evaluation.interpretability import ModelInterpreter


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


def load_test_data(
    data_dir: str,
    vocab_path: str,
    batch_size: int = 64,
    num_workers: int = 4
) -> tuple:
    """
    Load test dataset for evaluation.
    
    Returns:
        Tuple of (test_loader, preprocessor, test_texts, test_labels)
    """
    logger = logging.getLogger(__name__)
    logger.info("Loading test dataset...")
    
    # Load IMDB dataset
    dataset = IMDBDataset(data_dir)
    _, _, test_texts, test_labels = dataset.load_data()
    
    logger.info(f"Loaded {len(test_texts)} test samples")
    
    # Load preprocessor
    preprocessor = TextPreprocessor()
    preprocessor.load_vocabulary(vocab_path)
    
    logger.info(f"Loaded vocabulary with {preprocessor.vocab_size} words")
    
    # Preprocess test data
    test_sequences = preprocessor.preprocess_texts(test_texts, fit_vocabulary=False)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)
    
    # Create data loader
    test_dataset = torch.utils.data.TensorDataset(test_sequences, test_labels_tensor)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    logger.info(f"Created test loader with {len(test_loader)} batches")
    
    return test_loader, preprocessor, test_texts, test_labels


def evaluate_model_performance(
    model_path: str,
    vocab_path: str,
    test_loader: DataLoader,
    device: str = None
) -> tuple:
    """
    Evaluate model performance on test data.
    
    Returns:
        Tuple of (y_true, y_pred, y_prob, model_info)
    """
    logger = logging.getLogger(__name__)
    
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model configuration
    if 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
    else:
        # Fallback configuration
        model_config = {
            'vocab_size': 10000,
            'embedding_dim': 300,
            'hidden_dim': 128,
            'output_dim': 1,
            'n_layers': 2,
            'dropout': 0.3,
            'bidirectional': True
        }
        logger.warning("Model config not found in checkpoint, using default values")
    
    # Create model
    # Filter out non-constructor parameters
    constructor_params = {
        'vocab_size', 'embedding_dim', 'hidden_dim', 'output_dim', 
        'n_layers', 'dropout', 'bidirectional', 'pad_idx'
    }
    filtered_config = {k: v for k, v in model_config.items() if k in constructor_params}
    model = LSTMClassifier(**filtered_config)
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    logger.info("Model loaded successfully")
    
    # Evaluate on test data
    logger.info("Evaluating model on test data...")
    
    y_true = []
    y_pred = []
    y_prob = []
    
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(test_loader):
            data, targets = data.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(data)
            probabilities = torch.sigmoid(outputs.squeeze())
            predictions = (probabilities >= 0.5).float()
            
            # Store results
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())
            y_prob.extend(probabilities.cpu().numpy())
            
            if (batch_idx + 1) % 50 == 0:
                logger.info(f"Processed {batch_idx + 1}/{len(test_loader)} batches")
    
    logger.info("Evaluation completed")
    
    return np.array(y_true), np.array(y_pred), np.array(y_prob), model_config


def calculate_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: list = None
) -> dict:
    """Calculate comprehensive evaluation metrics including advanced metrics."""
    logger = logging.getLogger(__name__)
    logger.info("Calculating comprehensive metrics...")
    
    if class_names is None:
        class_names = ['Negative', 'Positive']
    
    # Create metrics calculator
    calculator = MetricsCalculator(class_names)
    
    # Calculate basic comprehensive metrics
    metrics = calculator.calculate_comprehensive_metrics(y_true, y_pred, y_prob)
    
    # Find optimal threshold
    optimal_threshold, optimal_f1 = calculator.find_optimal_threshold(y_true, y_prob, 'f1_score')
    metrics['optimal_threshold'] = {
        'threshold': optimal_threshold,
        'f1_score': optimal_f1
    }
    
    # Calculate threshold analysis
    thresholds = [i / 20.0 for i in range(1, 20)]  # 0.05 to 0.95 in 0.05 steps
    threshold_metrics = calculator.calculate_threshold_metrics(y_true, y_prob, thresholds)
    metrics['threshold_analysis'] = threshold_metrics
    
    # Calculate advanced metrics
    logger.info("Calculating advanced metrics...")
    try:
        advanced_calculator = AdvancedMetricsCalculator(class_names)
        advanced_metrics = advanced_calculator.calculate_advanced_metrics(y_true, y_pred, y_prob)
        metrics['advanced_metrics'] = advanced_metrics
        logger.info("Advanced metrics calculation completed")
    except Exception as e:
        logger.warning(f"Could not calculate advanced metrics: {e}")
        metrics['advanced_metrics'] = {'error': str(e)}
    
    logger.info("Metrics calculation completed")
    
    return metrics


def create_visualizations(
    metrics: dict,
    training_history: dict = None,
    output_dir: str = None,
    show_plots: bool = False
) -> dict:
    """Create evaluation visualizations."""
    logger = logging.getLogger(__name__)
    logger.info("Creating visualizations...")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    visualizer = EvaluationVisualizer()
    
    # Create comprehensive dashboard
    figures = visualizer.create_evaluation_dashboard(
        metrics,
        history=training_history,
        save_dir=output_dir,
        show_plots=show_plots
    )
    
    # Create threshold analysis plot
    if 'threshold_analysis' in metrics:
        threshold_save_path = os.path.join(output_dir, 'threshold_analysis.png') if output_dir else None
        
        fig_threshold = visualizer.plot_threshold_analysis(
            metrics['threshold_analysis'],
            title='Threshold Analysis',
            save_path=threshold_save_path,
            show_plot=show_plots
        )
        figures['threshold_analysis'] = fig_threshold
    
    logger.info(f"Created {len(figures)} visualizations")
    
    return figures


def generate_reports(
    metrics: dict,
    model_info: dict = None,
    training_summary: dict = None,
    output_dir: str = None
) -> dict:
    """Generate evaluation reports."""
    logger = logging.getLogger(__name__)
    logger.info("Generating evaluation reports...")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    reporter = EvaluationReporter()
    
    reports = {}
    
    # Generate text report
    if output_dir:
        text_path = os.path.join(output_dir, 'evaluation_report.txt')
        reporter.save_report(metrics, text_path, 'text', model_info, training_summary)
        reports['text_report'] = text_path
    
    # Generate JSON report
    if output_dir:
        json_path = os.path.join(output_dir, 'evaluation_report.json')
        reporter.save_report(metrics, json_path, 'json', model_info, training_summary)
        reports['json_report'] = json_path
    
    # Generate in-memory reports
    reports['text_content'] = reporter.generate_text_report(metrics, model_info, training_summary)
    reports['json_content'] = reporter.generate_json_report(metrics, model_info, training_summary)
    
    logger.info("Report generation completed")
    
    return reports


def load_training_history(model_path: str) -> dict:
    """Load training history from model checkpoint if available."""
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Try to extract training history
        if 'training_history' in checkpoint:
            return checkpoint['training_history']
        elif 'history' in checkpoint:
            return checkpoint['history']
        else:
            return None
    except Exception:
        return None


def print_summary(metrics: dict, model_info: dict = None):
    """Print evaluation summary to console."""
    logger = logging.getLogger(__name__)
    
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    
    # Basic metrics
    if 'basic_metrics' in metrics:
        basic = metrics['basic_metrics']
        print(f"Accuracy:     {basic.get('accuracy', 0):.4f} ({basic.get('accuracy', 0)*100:.2f}%)")
        print(f"Precision:    {basic.get('precision', 0):.4f}")
        print(f"Recall:       {basic.get('recall', 0):.4f}")
        print(f"F1-Score:     {basic.get('f1_score', 0):.4f}")
        print(f"Specificity:  {basic.get('specificity', 0):.4f}")
        
        if basic.get('roc_auc') is not None:
            print(f"ROC AUC:      {basic.get('roc_auc', 0):.4f}")
        if basic.get('pr_auc') is not None:
            print(f"PR AUC:       {basic.get('pr_auc', 0):.4f}")
        
        print(f"Support:      {basic.get('support', 0)} samples")
    
    # Optimal threshold
    if 'optimal_threshold' in metrics:
        opt = metrics['optimal_threshold']
        print(f"\nOptimal Threshold: {opt['threshold']:.3f} (F1-Score: {opt['f1_score']:.4f})")
    
    # Model info
    if model_info:
        print(f"\nModel Parameters: {model_info.get('total_parameters', 'N/A'):,}")
    
    print("=" * 80)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate LSTM Sentiment Classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--model-path', required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--vocab-path', required=True,
                       help='Path to vocabulary file')
    parser.add_argument('--data-dir', required=True,
                       help='Path to IMDB dataset directory')
    
    # Optional arguments
    parser.add_argument('--output-dir', default='evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for evaluation')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto',
                       help='Device to use for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loader workers')
    
    # Visualization arguments
    parser.add_argument('--show-plots', action='store_true',
                       help='Display plots during evaluation')
    parser.add_argument('--no-visualizations', action='store_true',
                       help='Skip visualization generation')
    parser.add_argument('--no-reports', action='store_true',
                       help='Skip report generation')
    
    # System arguments
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
    
    logger.info("Starting LSTM Sentiment Classifier evaluation")
    logger.info(f"Arguments: {vars(args)}")
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
        
        # Print summary
        print_summary(metrics, model_info)
        
        # Create visualizations
        if not args.no_visualizations:
            figures = create_visualizations(
                metrics,
                training_history,
                args.output_dir,
                args.show_plots
            )
            logger.info(f"Created {len(figures)} visualizations")
        
        # Generate reports
        if not args.no_reports:
            reports = generate_reports(
                metrics,
                model_info,
                None,  # training_summary not available from checkpoint
                args.output_dir
            )
            logger.info(f"Generated reports: {list(reports.keys())}")
        
        logger.info("Evaluation completed successfully!")
        logger.info(f"Results saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == '__main__':
    main()