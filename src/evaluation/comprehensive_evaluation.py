#!/usr/bin/env python3
"""
Comprehensive evaluation script with advanced metrics and interpretability.

This script demonstrates the full capabilities of the advanced evaluation system
including statistical analysis, model interpretability, and adversarial testing.
"""

import argparse
import os
import sys
import logging
import torch
import numpy as np
from datetime import datetime
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lstm_model import LSTMClassifier
from data.imdb_loader import IMDBDataset
from data.text_preprocessor import TextPreprocessor
from evaluation.advanced_metrics import AdvancedMetricsCalculator, compare_models_statistical
from evaluation.interpretability import ModelInterpreter, explain_prediction, analyze_word_importance
from evaluation.metrics import MetricsCalculator
from evaluation.visualization import EvaluationVisualizer, EvaluationReporter


def setup_logging(log_level: str = 'INFO'):
    """Set up logging configuration."""
    level = getattr(logging, log_level.upper())
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'comprehensive_evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )


def load_model_and_data(model_path: str, vocab_path: str, data_dir: str, device: str) -> tuple:
    """Load model, preprocessor, and test data."""
    logger = logging.getLogger(__name__)
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model configuration
    if 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
    else:
        model_config = {
            'vocab_size': 10000,
            'embedding_dim': 300,
            'hidden_dim': 128,
            'output_dim': 1,
            'n_layers': 2,
            'dropout': 0.3,
            'bidirectional': True
        }
        logger.warning("Model config not found, using defaults")
    
    # Create and load model
    constructor_params = {
        'vocab_size', 'embedding_dim', 'hidden_dim', 'output_dim', 
        'n_layers', 'dropout', 'bidirectional', 'pad_idx'
    }
    filtered_config = {k: v for k, v in model_config.items() if k in constructor_params}
    model = LSTMClassifier(**filtered_config)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Load preprocessor
    logger.info(f"Loading preprocessor from {vocab_path}")
    preprocessor = TextPreprocessor()
    preprocessor.load_vocabulary(vocab_path)
    
    # Load test data
    logger.info(f"Loading test data from {data_dir}")
    dataset = IMDBDataset(data_dir)
    _, _, test_texts, test_labels = dataset.load_data()
    
    logger.info(f"Loaded {len(test_texts)} test samples")
    
    return model, preprocessor, test_texts, test_labels, model_config


def run_basic_evaluation(model, preprocessor, test_texts, test_labels, device: str) -> dict:
    """Run basic model evaluation."""
    logger = logging.getLogger(__name__)
    logger.info("Running basic evaluation...")
    
    # Preprocess test data
    test_sequences = preprocessor.preprocess_texts(test_texts, fit_vocabulary=False)
    
    # Get predictions
    y_true = np.array(test_labels)
    y_pred = []
    y_prob = []
    
    batch_size = 64
    with torch.no_grad():
        for i in range(0, len(test_sequences), batch_size):
            batch_sequences = test_sequences[i:i+batch_size]
            batch_tensor = torch.stack(batch_sequences).to(device)
            
            outputs = model(batch_tensor)
            probabilities = torch.sigmoid(outputs.squeeze())
            predictions = (probabilities >= 0.5).float()
            
            y_pred.extend(predictions.cpu().numpy())
            y_prob.extend(probabilities.cpu().numpy())
    
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    
    # Calculate basic metrics
    calculator = MetricsCalculator(['Negative', 'Positive'])
    basic_metrics = calculator.calculate_comprehensive_metrics(y_true, y_pred, y_prob)
    
    logger.info("Basic evaluation completed")
    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'basic_metrics': basic_metrics
    }


def run_advanced_evaluation(y_true, y_pred, y_prob) -> dict:
    """Run advanced metrics evaluation."""
    logger = logging.getLogger(__name__)
    logger.info("Running advanced evaluation...")
    
    # Calculate advanced metrics
    advanced_calculator = AdvancedMetricsCalculator(['Negative', 'Positive'])
    advanced_metrics = advanced_calculator.calculate_advanced_metrics(y_true, y_pred, y_prob)
    
    logger.info("Advanced evaluation completed")
    return advanced_metrics


def run_interpretability_analysis(model, preprocessor, test_texts, test_labels, device: str, num_samples: int = 50) -> dict:
    """Run model interpretability analysis."""
    logger = logging.getLogger(__name__)
    logger.info("Running interpretability analysis...")
    
    # Create interpreter
    interpreter = ModelInterpreter(model, preprocessor, device)
    
    # Sample texts for analysis
    sample_indices = np.random.choice(len(test_texts), min(num_samples, len(test_texts)), replace=False)
    sample_texts = [test_texts[i] for i in sample_indices]
    sample_labels = [test_labels[i] for i in sample_indices]
    
    # Analyze word importance
    word_importance = interpreter.analyze_word_importance(
        sample_texts, 
        sample_labels, 
        method='gradient',
        top_k=20
    )
    
    # Generate explanations for a few examples
    explanations = []
    for i in range(min(5, len(sample_texts))):
        text = sample_texts[i]
        explanation = interpreter.explain_prediction(text, method='gradient')
        explanations.append(explanation)
    
    # Generate adversarial examples
    adversarial_results = []
    for i in range(min(3, len(sample_texts))):
        text = sample_texts[i]
        adversarial = interpreter.generate_adversarial_examples(
            text, 
            method='word_substitution',
            max_perturbations=3
        )
        adversarial_results.append(adversarial)
    
    logger.info("Interpretability analysis completed")
    return {
        'word_importance': word_importance,
        'sample_explanations': explanations,
        'adversarial_examples': adversarial_results
    }


def create_comprehensive_report(
    basic_results: dict,
    advanced_results: dict,
    interpretability_results: dict,
    model_config: dict,
    output_dir: str
) -> dict:
    """Create comprehensive evaluation report."""
    logger = logging.getLogger(__name__)
    logger.info("Creating comprehensive report...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine all results
    comprehensive_report = {
        'evaluation_metadata': {
            'timestamp': datetime.now().isoformat(),
            'evaluation_type': 'comprehensive_advanced',
            'version': '1.0'
        },
        'model_configuration': model_config,
        'basic_metrics': basic_results['basic_metrics'],
        'advanced_metrics': advanced_results,
        'interpretability_analysis': interpretability_results
    }
    
    # Save JSON report
    json_path = os.path.join(output_dir, 'comprehensive_evaluation_report.json')
    with open(json_path, 'w') as f:
        json.dump(comprehensive_report, f, indent=2, default=str)
    
    # Create text summary
    text_summary = generate_text_summary(comprehensive_report)
    text_path = os.path.join(output_dir, 'evaluation_summary.txt')
    with open(text_path, 'w') as f:
        f.write(text_summary)
    
    # Create visualizations
    visualizer = EvaluationVisualizer()
    
    # Basic metrics visualization
    basic_figures = visualizer.create_evaluation_dashboard(
        basic_results['basic_metrics'],
        save_dir=os.path.join(output_dir, 'basic_plots'),
        show_plots=False
    )
    
    # Advanced metrics visualizations
    if 'roc_analysis' in advanced_results and 'error' not in advanced_results['roc_analysis']:
        roc_data = advanced_results['roc_analysis']
        fig_roc = visualizer.plot_roc_curve(
            roc_data['fpr'],
            roc_data['tpr'],
            roc_data['auc_score'],
            title='Advanced ROC Analysis',
            save_path=os.path.join(output_dir, 'advanced_roc_curve.png'),
            show_plot=False
        )
    
    # Calibration plot
    if 'calibration_analysis' in advanced_results and 'error' not in advanced_results['calibration_analysis']:
        cal_data = advanced_results['calibration_analysis']
        if 'reliability_diagram' in cal_data:
            fig_cal = create_calibration_plot(cal_data, output_dir)
    
    logger.info(f"Comprehensive report saved to {output_dir}")
    return {
        'report_path': json_path,
        'summary_path': text_path,
        'output_directory': output_dir
    }


def create_calibration_plot(calibration_data: dict, output_dir: str) -> None:
    """Create calibration reliability diagram."""
    import matplotlib.pyplot as plt
    
    reliability = calibration_data['reliability_diagram']
    fraction_pos = reliability['fraction_of_positives']
    mean_pred = reliability['mean_predicted_value']
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot reliability diagram
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    ax.plot(mean_pred, fraction_pos, 'bo-', label='Model Calibration')
    
    # Fill area between perfect and actual
    ax.fill_between(mean_pred, fraction_pos, mean_pred, alpha=0.3, color='red')
    
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Calibration Reliability Diagram')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add calibration metrics as text
    ece = calibration_data['expected_calibration_error']
    brier = calibration_data['brier_score']
    
    text = f'ECE: {ece:.4f}\nBrier Score: {brier:.4f}'
    ax.text(0.05, 0.95, text, transform=ax.transAxes, 
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'calibration_diagram.png'), dpi=300, bbox_inches='tight')
    plt.close()


def generate_text_summary(report: dict) -> str:
    """Generate text summary of comprehensive evaluation."""
    lines = []
    
    lines.append("=" * 80)
    lines.append("COMPREHENSIVE LSTM SENTIMENT CLASSIFIER EVALUATION")
    lines.append("=" * 80)
    lines.append(f"Generated: {report['evaluation_metadata']['timestamp']}")
    lines.append("")
    
    # Basic metrics summary
    if 'basic_metrics' in report:
        basic = report['basic_metrics']['basic_metrics']
        lines.append("BASIC PERFORMANCE METRICS")
        lines.append("-" * 40)
        lines.append(f"Accuracy:     {basic['accuracy']:.4f}")
        lines.append(f"Precision:    {basic['precision']:.4f}")
        lines.append(f"Recall:       {basic['recall']:.4f}")
        lines.append(f"F1-Score:     {basic['f1_score']:.4f}")
        if 'roc_auc' in basic and basic['roc_auc'] is not None:
            lines.append(f"ROC AUC:      {basic['roc_auc']:.4f}")
        lines.append("")
    
    # Advanced metrics summary
    if 'advanced_metrics' in report:
        adv = report['advanced_metrics']
        
        if 'roc_analysis' in adv and 'error' not in adv['roc_analysis']:
            roc = adv['roc_analysis']
            lines.append("ADVANCED ROC ANALYSIS")
            lines.append("-" * 40)
            lines.append(f"AUC Score:    {roc['auc_score']:.4f}")
            if 'auc_confidence_interval' in roc:
                ci = roc['auc_confidence_interval']
                lines.append(f"AUC 95% CI:   [{ci[0]:.4f}, {ci[1]:.4f}]")
            lines.append(f"Interpretation: {roc.get('auc_interpretation', 'N/A')}")
            lines.append("")
        
        if 'calibration_analysis' in adv and 'error' not in adv['calibration_analysis']:
            cal = adv['calibration_analysis']
            lines.append("CALIBRATION ANALYSIS")
            lines.append("-" * 40)
            lines.append(f"Brier Score:  {cal['brier_score']:.4f}")
            lines.append(f"ECE:          {cal['expected_calibration_error']:.4f}")
            lines.append(f"MCE:          {cal['maximum_calibration_error']:.4f}")
            lines.append(f"Calibration:  {cal.get('calibration_interpretation', 'N/A')}")
            lines.append("")
    
    # Interpretability summary
    if 'interpretability_analysis' in report:
        interp = report['interpretability_analysis']
        
        if 'word_importance' in interp and 'error' not in interp['word_importance']:
            word_imp = interp['word_importance']
            lines.append("WORD IMPORTANCE ANALYSIS")
            lines.append("-" * 40)
            lines.append(f"Texts Analyzed: {word_imp['total_texts_analyzed']}")
            lines.append(f"Unique Words:   {word_imp['total_unique_words']}")
            
            if 'top_words_overall' in word_imp:
                lines.append("\nTop Important Words:")
                for word, stats in word_imp['top_words_overall'][:5]:
                    lines.append(f"  {word:15}: {stats['mean_attribution']:.4f}")
            lines.append("")
        
        if 'adversarial_examples' in interp:
            adv_examples = interp['adversarial_examples']
            successful_attacks = sum(1 for ex in adv_examples if 'successful_attacks' in ex and ex['successful_attacks'] > 0)
            lines.append("ADVERSARIAL ROBUSTNESS")
            lines.append("-" * 40)
            lines.append(f"Tests Performed: {len(adv_examples)}")
            lines.append(f"Successful Attacks: {successful_attacks}")
            lines.append("")
    
    lines.append("=" * 80)
    lines.append("End of Summary")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def main():
    """Main comprehensive evaluation function."""
    parser = argparse.ArgumentParser(
        description="Comprehensive LSTM Sentiment Classifier Evaluation",
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
    parser.add_argument('--output-dir', default='comprehensive_evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto',
                       help='Device to use for evaluation')
    parser.add_argument('--interpretability-samples', type=int, default=50,
                       help='Number of samples for interpretability analysis')
    
    # System arguments
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info("Starting comprehensive LSTM sentiment classifier evaluation")
    logger.info(f"Using device: {device}")
    
    try:
        # Load model and data
        model, preprocessor, test_texts, test_labels, model_config = load_model_and_data(
            args.model_path, args.vocab_path, args.data_dir, device
        )
        
        # Run basic evaluation
        basic_results = run_basic_evaluation(model, preprocessor, test_texts, test_labels, device)
        
        # Run advanced evaluation
        advanced_results = run_advanced_evaluation(
            basic_results['y_true'], 
            basic_results['y_pred'], 
            basic_results['y_prob']
        )
        
        # Run interpretability analysis
        interpretability_results = run_interpretability_analysis(
            model, preprocessor, test_texts, test_labels, device, args.interpretability_samples
        )
        
        # Create comprehensive report
        report_info = create_comprehensive_report(
            basic_results, advanced_results, interpretability_results, 
            model_config, args.output_dir
        )
        
        print("\n" + "=" * 80)
        print("COMPREHENSIVE EVALUATION COMPLETED")
        print("=" * 80)
        print(f"Results saved to: {report_info['output_directory']}")
        print(f"Summary report: {report_info['summary_path']}")
        print(f"Detailed report: {report_info['report_path']}")
        print("=" * 80)
        
        logger.info("Comprehensive evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Comprehensive evaluation failed: {e}")
        raise


if __name__ == '__main__':
    main()