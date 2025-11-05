"""
Evaluation module for LSTM Sentiment Classifier.

This module provides comprehensive evaluation functionality including
metrics calculation, visualization, reporting tools, advanced metrics,
and model interpretability capabilities.
"""

from .metrics import (
    MetricsCalculator, calculate_metrics, calculate_confusion_matrix, 
    find_optimal_threshold
)
from .visualization import (
    EvaluationVisualizer, EvaluationReporter,
    plot_confusion_matrix, plot_training_history, generate_evaluation_report
)
from .advanced_metrics import (
    AdvancedMetricsCalculator, calculate_advanced_metrics, compare_models_statistical
)
from .interpretability import (
    ModelInterpreter, explain_prediction, analyze_word_importance, generate_adversarial_examples
)

__all__ = [
    # Basic Metrics
    'MetricsCalculator', 'calculate_metrics', 'calculate_confusion_matrix', 'find_optimal_threshold',
    
    # Advanced Metrics
    'AdvancedMetricsCalculator', 'calculate_advanced_metrics', 'compare_models_statistical',
    
    # Interpretability
    'ModelInterpreter', 'explain_prediction', 'analyze_word_importance', 'generate_adversarial_examples',
    
    # Visualization and Reporting
    'EvaluationVisualizer', 'EvaluationReporter',
    'plot_confusion_matrix', 'plot_training_history', 'generate_evaluation_report'
]