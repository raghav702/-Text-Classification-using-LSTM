#!/usr/bin/env python3
"""
Advanced Evaluation Demo for LSTM Sentiment Classifier.

This script demonstrates the advanced evaluation capabilities including
statistical analysis, model interpretability, and adversarial testing.
"""

import os
import sys
import numpy as np
import torch

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from evaluation.advanced_metrics import AdvancedMetricsCalculator, compare_models_statistical
from evaluation.interpretability import ModelInterpreter, explain_prediction
from evaluation.metrics import MetricsCalculator


def demo_advanced_metrics():
    """Demonstrate advanced metrics calculation."""
    print("=" * 60)
    print("ADVANCED METRICS DEMONSTRATION")
    print("=" * 60)
    
    # Generate synthetic evaluation data
    np.random.seed(42)
    n_samples = 1000
    
    # Create realistic predictions with some calibration issues
    y_true = np.random.binomial(1, 0.4, n_samples)  # 40% positive class
    
    # Generate probabilities that are somewhat overconfident
    y_prob = np.where(y_true == 1, 
                     np.random.beta(3, 1, n_samples),  # Higher probs for positive class
                     np.random.beta(1, 3, n_samples))  # Lower probs for negative class
    
    # Add some noise and overconfidence
    y_prob = np.clip(y_prob * 1.2 - 0.1, 0.01, 0.99)
    y_pred = (y_prob >= 0.5).astype(int)
    
    print(f"Dataset: {n_samples} samples, {np.sum(y_true)} positive ({np.mean(y_true):.1%})")
    print(f"Model accuracy: {np.mean(y_true == y_pred):.3f}")
    print()
    
    # Calculate advanced metrics
    calculator = AdvancedMetricsCalculator(['Negative', 'Positive'])
    advanced_metrics = calculator.calculate_advanced_metrics(y_true, y_pred, y_prob)
    
    # Display ROC analysis
    if 'roc_analysis' in advanced_metrics and 'error' not in advanced_metrics['roc_analysis']:
        roc = advanced_metrics['roc_analysis']
        print("ROC ANALYSIS:")
        print(f"  AUC Score: {roc['auc_score']:.4f}")
        if 'auc_confidence_interval' in roc:
            ci = roc['auc_confidence_interval']
            print(f"  AUC 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
        print(f"  Interpretation: {roc.get('auc_interpretation', 'N/A')}")
        
        opt = roc['optimal_threshold']
        print(f"  Optimal Threshold: {opt['threshold']:.3f}")
        print(f"  Sensitivity: {opt['sensitivity']:.3f}")
        print(f"  Specificity: {opt['specificity']:.3f}")
        print()
    
    # Display calibration analysis
    if 'calibration_analysis' in advanced_metrics and 'error' not in advanced_metrics['calibration_analysis']:
        cal = advanced_metrics['calibration_analysis']
        print("CALIBRATION ANALYSIS:")
        print(f"  Brier Score: {cal['brier_score']:.4f}")
        print(f"  Expected Calibration Error: {cal['expected_calibration_error']:.4f}")
        print(f"  Maximum Calibration Error: {cal['maximum_calibration_error']:.4f}")
        print(f"  Calibration Quality: {cal.get('calibration_interpretation', 'N/A')}")
        print()
    
    # Display confidence intervals
    if 'confidence_intervals' in advanced_metrics and 'error' not in advanced_metrics['confidence_intervals']:
        ci = advanced_metrics['confidence_intervals']
        print("CONFIDENCE INTERVALS:")
        for metric, data in ci.items():
            if isinstance(data, dict) and 'confidence_interval' in data:
                ci_range = data['confidence_interval']
                print(f"  {metric.replace('_', ' ').title()}: {data['value']:.4f} [{ci_range[0]:.4f}, {ci_range[1]:.4f}]")
        print()
    
    return advanced_metrics


def demo_model_comparison():
    """Demonstrate statistical model comparison."""
    print("=" * 60)
    print("STATISTICAL MODEL COMPARISON DEMONSTRATION")
    print("=" * 60)
    
    # Generate synthetic data for two models
    np.random.seed(42)
    n_samples = 500
    
    y_true = np.random.binomial(1, 0.5, n_samples)
    
    # Model 1: Slightly better performance
    y_pred1 = y_true.copy()
    # Add some errors (15% error rate)
    error_indices1 = np.random.choice(n_samples, int(0.15 * n_samples), replace=False)
    y_pred1[error_indices1] = 1 - y_pred1[error_indices1]
    
    # Model 2: Slightly worse performance
    y_pred2 = y_true.copy()
    # Add some errors (20% error rate)
    error_indices2 = np.random.choice(n_samples, int(0.20 * n_samples), replace=False)
    y_pred2[error_indices2] = 1 - y_pred2[error_indices2]
    
    print(f"Model 1 Accuracy: {np.mean(y_true == y_pred1):.3f}")
    print(f"Model 2 Accuracy: {np.mean(y_true == y_pred2):.3f}")
    print()
    
    # Perform statistical comparison
    calculator = AdvancedMetricsCalculator()
    comparison = calculator.compare_models_statistical(
        y_true, y_pred1, y_pred2, "LSTM Model A", "LSTM Model B"
    )
    
    if 'error' not in comparison:
        print("STATISTICAL COMPARISON RESULTS:")
        
        # McNemar's test
        mcnemar = comparison['mcnemar_test']
        print(f"McNemar's Test:")
        print(f"  Statistic: {mcnemar['statistic']:.4f}")
        print(f"  P-value: {mcnemar['p_value']:.6f}")
        print(f"  Significant Difference: {'Yes' if mcnemar['significant'] else 'No'}")
        
        # Effect size
        effect = comparison['effect_size']
        print(f"Effect Size (Cohen's h): {effect['cohens_h']:.4f} ({effect['interpretation']})")
        
        # Performance differences
        perf_diff = comparison['performance_difference']
        print(f"Accuracy Difference: {perf_diff['accuracy_diff']:.4f}")
        print(f"F1-Score Difference: {perf_diff['f1_diff']:.4f}")
        print()
    
    return comparison


def demo_interpretability():
    """Demonstrate model interpretability (simplified version)."""
    print("=" * 60)
    print("MODEL INTERPRETABILITY DEMONSTRATION")
    print("=" * 60)
    
    # Note: This is a simplified demo without actual model loading
    # In practice, you would load a trained model and preprocessor
    
    print("Interpretability Features Available:")
    print("1. Gradient-based Attribution")
    print("   - Explains predictions using input gradients")
    print("   - Shows which words contribute most to the prediction")
    print()
    
    print("2. LIME Explanations (if installed)")
    print("   - Local Interpretable Model-agnostic Explanations")
    print("   - Perturbs input to understand model behavior")
    print()
    
    print("3. SHAP Values (if installed)")
    print("   - SHapley Additive exPlanations")
    print("   - Game-theoretic approach to feature attribution")
    print()
    
    print("4. Attention Visualization")
    print("   - Shows attention weights (if model has attention)")
    print("   - Highlights which words the model focuses on")
    print()
    
    print("5. Word Importance Analysis")
    print("   - Aggregates attributions across multiple texts")
    print("   - Identifies most important words for each class")
    print()
    
    print("6. Adversarial Example Generation")
    print("   - Tests model robustness")
    print("   - Finds minimal changes that flip predictions")
    print("   - Methods: word substitution, deletion, insertion")
    print()
    
    # Example of what an explanation would look like
    print("Example Explanation Output:")
    print("-" * 30)
    example_explanation = {
        'method': 'gradient',
        'prediction': 0.85,
        'predicted_class': 'Positive',
        'confidence': 0.85,
        'word_attributions': [
            {'word': 'excellent', 'attribution': 0.45, 'position': 3},
            {'word': 'amazing', 'attribution': 0.32, 'position': 7},
            {'word': 'love', 'attribution': 0.28, 'position': 1},
            {'word': 'great', 'attribution': 0.21, 'position': 5},
            {'word': 'not', 'attribution': -0.15, 'position': 8}
        ],
        'text': 'I love this excellent and great movie, amazing but not perfect'
    }
    
    print(f"Text: '{example_explanation['text']}'")
    print(f"Prediction: {example_explanation['predicted_class']} (confidence: {example_explanation['confidence']:.3f})")
    print("Word Attributions:")
    for attr in example_explanation['word_attributions']:
        sign = "+" if attr['attribution'] >= 0 else ""
        print(f"  {attr['word']:12}: {sign}{attr['attribution']:6.3f}")
    print()


def main():
    """Run all demonstrations."""
    print("LSTM SENTIMENT CLASSIFIER - ADVANCED EVALUATION DEMO")
    print("=" * 80)
    print()
    
    try:
        # Demo 1: Advanced Metrics
        advanced_metrics = demo_advanced_metrics()
        
        # Demo 2: Model Comparison
        comparison_results = demo_model_comparison()
        
        # Demo 3: Interpretability
        demo_interpretability()
        
        print("=" * 80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print()
        print("To use these features with your trained model:")
        print("1. Load your trained LSTM model and preprocessor")
        print("2. Use AdvancedMetricsCalculator for detailed performance analysis")
        print("3. Use ModelInterpreter for explanation and interpretability")
        print("4. Use compare_models_statistical for comparing multiple models")
        print()
        print("For a complete evaluation, run:")
        print("python src/evaluation/comprehensive_evaluation.py --model-path <path> --vocab-path <path> --data-dir <path>")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()