"""
Comprehensive metrics calculation for LSTM sentiment classifier evaluation.

This module provides functions for calculating various performance metrics
including accuracy, precision, recall, F1-score, and confusion matrices.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Union, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
import logging


class MetricsCalculator:
    """
    Comprehensive metrics calculator for binary sentiment classification.
    
    Provides methods to calculate various performance metrics and generate
    detailed evaluation reports.
    """
    
    def __init__(self, class_names: List[str] = None):
        """
        Initialize metrics calculator.
        
        Args:
            class_names: Names for the classes (default: ['Negative', 'Positive'])
        """
        self.class_names = class_names or ['Negative', 'Positive']
        self.logger = logging.getLogger(__name__)
    
    def calculate_basic_metrics(
        self,
        y_true: Union[np.ndarray, torch.Tensor, List],
        y_pred: Union[np.ndarray, torch.Tensor, List],
        y_prob: Union[np.ndarray, torch.Tensor, List] = None
    ) -> Dict[str, float]:
        """
        Calculate basic classification metrics.
        
        Args:
            y_true: True labels (0 or 1)
            y_pred: Predicted labels (0 or 1)
            y_prob: Predicted probabilities (optional, for AUC calculation)
            
        Returns:
            Dictionary containing basic metrics
        """
        # Convert to numpy arrays
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        
        # Validate inputs
        self._validate_inputs(y_true, y_pred)
        
        # Calculate basic metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'specificity': self._calculate_specificity(y_true, y_pred),
            'support': len(y_true)
        }
        
        # Add AUC metrics if probabilities provided
        if y_prob is not None:
            y_prob = self._to_numpy(y_prob)
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
                metrics['pr_auc'] = average_precision_score(y_true, y_prob)
            except ValueError as e:
                self.logger.warning(f"Could not calculate AUC metrics: {e}")
                metrics['roc_auc'] = None
                metrics['pr_auc'] = None
        
        return metrics
    
    def calculate_confusion_matrix(
        self,
        y_true: Union[np.ndarray, torch.Tensor, List],
        y_pred: Union[np.ndarray, torch.Tensor, List],
        normalize: str = None
    ) -> Tuple[np.ndarray, Dict[str, any]]:
        """
        Calculate confusion matrix with additional statistics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: Normalization mode ('true', 'pred', 'all', or None)
            
        Returns:
            Tuple of (confusion_matrix, statistics_dict)
        """
        # Convert to numpy arrays
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        
        # Validate inputs
        self._validate_inputs(y_true, y_pred)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred, normalize=normalize)
        
        # Calculate statistics from confusion matrix
        if normalize is None:
            tn, fp, fn, tp = cm.ravel()
            
            stats = {
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp),
                'total_samples': int(tn + fp + fn + tp),
                'positive_samples': int(tp + fn),
                'negative_samples': int(tn + fp),
                'predicted_positive': int(tp + fp),
                'predicted_negative': int(tn + fn)
            }
            
            # Calculate rates
            total = stats['total_samples']
            if total > 0:
                stats['true_positive_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0
                stats['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
                stats['true_negative_rate'] = tn / (tn + fp) if (tn + fp) > 0 else 0
                stats['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
                stats['positive_predictive_value'] = tp / (tp + fp) if (tp + fp) > 0 else 0
                stats['negative_predictive_value'] = tn / (tn + fn) if (tn + fn) > 0 else 0
        else:
            stats = {
                'normalization': normalize,
                'matrix_sum': float(cm.sum())
            }
        
        return cm, stats
    
    def calculate_per_class_metrics(
        self,
        y_true: Union[np.ndarray, torch.Tensor, List],
        y_pred: Union[np.ndarray, torch.Tensor, List]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate metrics for each class separately.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with per-class metrics
        """
        # Convert to numpy arrays
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        
        # Validate inputs
        self._validate_inputs(y_true, y_pred)
        
        # Calculate per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # Get support (number of samples) for each class
        unique_labels, counts = np.unique(y_true, return_counts=True)
        support_dict = dict(zip(unique_labels, counts))
        
        # Organize results
        per_class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            class_idx = i
            per_class_metrics[class_name] = {
                'precision': float(precision_per_class[class_idx]) if class_idx < len(precision_per_class) else 0.0,
                'recall': float(recall_per_class[class_idx]) if class_idx < len(recall_per_class) else 0.0,
                'f1_score': float(f1_per_class[class_idx]) if class_idx < len(f1_per_class) else 0.0,
                'support': int(support_dict.get(class_idx, 0))
            }
        
        return per_class_metrics
    
    def calculate_comprehensive_metrics(
        self,
        y_true: Union[np.ndarray, torch.Tensor, List],
        y_pred: Union[np.ndarray, torch.Tensor, List],
        y_prob: Union[np.ndarray, torch.Tensor, List] = None,
        threshold: float = 0.5
    ) -> Dict[str, any]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels (or probabilities if y_prob is None)
            y_prob: Predicted probabilities (optional)
            threshold: Decision threshold for converting probabilities to predictions
            
        Returns:
            Dictionary containing all calculated metrics
        """
        # Convert to numpy arrays
        y_true = self._to_numpy(y_true)
        
        # Handle case where y_pred contains probabilities
        if y_prob is None and y_pred is not None:
            y_pred_array = self._to_numpy(y_pred)
            if np.all((y_pred_array >= 0) & (y_pred_array <= 1)) and not np.all(np.isin(y_pred_array, [0, 1])):
                # y_pred contains probabilities
                y_prob = y_pred_array
                y_pred = (y_pred_array >= threshold).astype(int)
            else:
                y_pred = y_pred_array
        else:
            y_pred = self._to_numpy(y_pred)
            if y_prob is not None:
                y_prob = self._to_numpy(y_prob)
        
        # Calculate all metrics
        results = {
            'threshold': threshold,
            'basic_metrics': self.calculate_basic_metrics(y_true, y_pred, y_prob),
            'per_class_metrics': self.calculate_per_class_metrics(y_true, y_pred)
        }
        
        # Add confusion matrix
        cm, cm_stats = self.calculate_confusion_matrix(y_true, y_pred)
        results['confusion_matrix'] = cm.tolist()
        results['confusion_matrix_stats'] = cm_stats
        
        # Add classification report
        try:
            class_report = classification_report(
                y_true, y_pred,
                target_names=self.class_names,
                output_dict=True,
                zero_division=0
            )
            results['classification_report'] = class_report
        except Exception as e:
            self.logger.warning(f"Could not generate classification report: {e}")
            results['classification_report'] = None
        
        # Add ROC and PR curve data if probabilities available
        if y_prob is not None:
            try:
                # ROC curve
                fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
                results['roc_curve'] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'thresholds': roc_thresholds.tolist()
                }
                
                # Precision-Recall curve
                precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
                results['pr_curve'] = {
                    'precision': precision.tolist(),
                    'recall': recall.tolist(),
                    'thresholds': pr_thresholds.tolist()
                }
            except Exception as e:
                self.logger.warning(f"Could not calculate curve data: {e}")
                results['roc_curve'] = None
                results['pr_curve'] = None
        
        return results
    
    def calculate_threshold_metrics(
        self,
        y_true: Union[np.ndarray, torch.Tensor, List],
        y_prob: Union[np.ndarray, torch.Tensor, List],
        thresholds: List[float] = None
    ) -> Dict[float, Dict[str, float]]:
        """
        Calculate metrics across different decision thresholds.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            thresholds: List of thresholds to evaluate (default: 0.1 to 0.9 in 0.1 steps)
            
        Returns:
            Dictionary mapping thresholds to their metrics
        """
        if thresholds is None:
            thresholds = [i / 10.0 for i in range(1, 10)]  # 0.1 to 0.9
        
        # Convert to numpy arrays
        y_true = self._to_numpy(y_true)
        y_prob = self._to_numpy(y_prob)
        
        threshold_metrics = {}
        
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            metrics = self.calculate_basic_metrics(y_true, y_pred, y_prob)
            threshold_metrics[threshold] = metrics
        
        return threshold_metrics
    
    def find_optimal_threshold(
        self,
        y_true: Union[np.ndarray, torch.Tensor, List],
        y_prob: Union[np.ndarray, torch.Tensor, List],
        metric: str = 'f1_score',
        thresholds: List[float] = None
    ) -> Tuple[float, float]:
        """
        Find optimal threshold based on specified metric.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            metric: Metric to optimize ('f1_score', 'accuracy', 'precision', 'recall')
            thresholds: List of thresholds to evaluate
            
        Returns:
            Tuple of (optimal_threshold, best_metric_value)
        """
        if thresholds is None:
            thresholds = [i / 100.0 for i in range(1, 100)]  # 0.01 to 0.99
        
        threshold_metrics = self.calculate_threshold_metrics(y_true, y_prob, thresholds)
        
        best_threshold = 0.5
        best_value = 0.0
        
        for threshold, metrics in threshold_metrics.items():
            if metric in metrics and metrics[metric] > best_value:
                best_value = metrics[metric]
                best_threshold = threshold
        
        return best_threshold, best_value
    
    def _to_numpy(self, data: Union[np.ndarray, torch.Tensor, List]) -> np.ndarray:
        """Convert input data to numpy array."""
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        elif isinstance(data, list):
            return np.array(data)
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
    
    def _validate_inputs(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Validate input arrays."""
        if len(y_true) != len(y_pred):
            raise ValueError(f"Length mismatch: y_true ({len(y_true)}) vs y_pred ({len(y_pred)})")
        
        if len(y_true) == 0:
            raise ValueError("Empty input arrays")
        
        # Check if labels are binary
        unique_true = np.unique(y_true)
        unique_pred = np.unique(y_pred)
        
        if not np.all(np.isin(unique_true, [0, 1])):
            raise ValueError(f"y_true contains non-binary values: {unique_true}")
        
        if not np.all(np.isin(unique_pred, [0, 1])):
            raise ValueError(f"y_pred contains non-binary values: {unique_pred}")
    
    def _calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate specificity (true negative rate)."""
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            return tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return 0.0


def calculate_metrics(
    y_true: Union[np.ndarray, torch.Tensor, List],
    y_pred: Union[np.ndarray, torch.Tensor, List],
    y_prob: Union[np.ndarray, torch.Tensor, List] = None,
    class_names: List[str] = None,
    threshold: float = 0.5
) -> Dict[str, any]:
    """
    Convenience function to calculate comprehensive metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        class_names: Names for the classes
        threshold: Decision threshold
        
    Returns:
        Dictionary containing all calculated metrics
    """
    calculator = MetricsCalculator(class_names)
    return calculator.calculate_comprehensive_metrics(y_true, y_pred, y_prob, threshold)


def calculate_confusion_matrix(
    y_true: Union[np.ndarray, torch.Tensor, List],
    y_pred: Union[np.ndarray, torch.Tensor, List],
    class_names: List[str] = None,
    normalize: str = None
) -> Tuple[np.ndarray, Dict[str, any]]:
    """
    Convenience function to calculate confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names for the classes
        normalize: Normalization mode
        
    Returns:
        Tuple of (confusion_matrix, statistics)
    """
    calculator = MetricsCalculator(class_names)
    return calculator.calculate_confusion_matrix(y_true, y_pred, normalize)


def find_optimal_threshold(
    y_true: Union[np.ndarray, torch.Tensor, List],
    y_prob: Union[np.ndarray, torch.Tensor, List],
    metric: str = 'f1_score'
) -> Tuple[float, float]:
    """
    Convenience function to find optimal threshold.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        metric: Metric to optimize
        
    Returns:
        Tuple of (optimal_threshold, best_metric_value)
    """
    calculator = MetricsCalculator()
    return calculator.find_optimal_threshold(y_true, y_prob, metric)