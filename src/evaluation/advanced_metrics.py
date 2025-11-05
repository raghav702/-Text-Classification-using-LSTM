"""
Advanced evaluation metrics for LSTM sentiment classifier.

This module provides advanced metrics beyond basic accuracy, including
calibration analysis, statistical significance testing, and detailed
performance analysis with confidence intervals.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Union, Optional, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, brier_score_loss
)
from sklearn.calibration import calibration_curve
from scipy import stats
import logging
import warnings
warnings.filterwarnings('ignore')


class AdvancedMetricsCalculator:
    """
    Advanced metrics calculator for comprehensive model evaluation.
    
    Provides methods for calculating advanced performance metrics including
    calibration analysis, statistical significance testing, and confidence intervals.
    """
    
    def __init__(self, class_names: List[str] = None, confidence_level: float = 0.95):
        """
        Initialize advanced metrics calculator.
        
        Args:
            class_names: Names for the classes (default: ['Negative', 'Positive'])
            confidence_level: Confidence level for statistical tests (default: 0.95)
        """
        self.class_names = class_names or ['Negative', 'Positive']
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.logger = logging.getLogger(__name__)
    
    def calculate_advanced_metrics(
        self,
        y_true: Union[np.ndarray, torch.Tensor, List],
        y_pred: Union[np.ndarray, torch.Tensor, List],
        y_prob: Union[np.ndarray, torch.Tensor, List] = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive advanced metrics.
        
        Args:
            y_true: True labels (0 or 1)
            y_pred: Predicted labels (0 or 1)
            y_prob: Predicted probabilities (required for advanced metrics)
            
        Returns:
            Dictionary containing advanced metrics
        """
        # Convert to numpy arrays
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        
        if y_prob is not None:
            y_prob = self._to_numpy(y_prob)
        else:
            self.logger.warning("Probabilities not provided, some advanced metrics will be unavailable")
        
        # Validate inputs
        self._validate_inputs(y_true, y_pred)
        
        advanced_metrics = {}
        
        # 1. Enhanced AUC-ROC with confidence intervals
        if y_prob is not None:
            advanced_metrics['roc_analysis'] = self._calculate_roc_analysis(y_true, y_prob)
        
        # 2. Precision-Recall analysis with confidence intervals
        if y_prob is not None:
            advanced_metrics['pr_analysis'] = self._calculate_pr_analysis(y_true, y_prob)
        
        # 3. Calibration analysis
        if y_prob is not None:
            advanced_metrics['calibration_analysis'] = self._calculate_calibration_analysis(y_true, y_prob)
        
        # 4. Class-specific detailed metrics with confidence intervals
        advanced_metrics['detailed_class_metrics'] = self._calculate_detailed_class_metrics(y_true, y_pred)
        
        # 5. Performance confidence intervals
        advanced_metrics['confidence_intervals'] = self._calculate_confidence_intervals(y_true, y_pred, y_prob)
        
        # 6. Error analysis
        advanced_metrics['error_analysis'] = self._calculate_error_analysis(y_true, y_pred, y_prob)
        
        # 7. Threshold optimization analysis
        if y_prob is not None:
            advanced_metrics['threshold_optimization'] = self._calculate_threshold_optimization(y_true, y_prob)
        
        return advanced_metrics
    
    def _calculate_roc_analysis(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
        """Calculate detailed ROC analysis with confidence intervals."""
        try:
            # Basic ROC calculation
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            auc_score = roc_auc_score(y_true, y_prob)
            
            # Bootstrap confidence interval for AUC
            auc_ci = self._bootstrap_auc_ci(y_true, y_prob)
            
            # Find optimal threshold (Youden's J statistic)
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = thresholds[optimal_idx]
            optimal_sensitivity = tpr[optimal_idx]
            optimal_specificity = 1 - fpr[optimal_idx]
            
            return {
                'auc_score': float(auc_score),
                'auc_confidence_interval': auc_ci,
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist(),
                'optimal_threshold': {
                    'threshold': float(optimal_threshold),
                    'sensitivity': float(optimal_sensitivity),
                    'specificity': float(optimal_specificity),
                    'youden_j': float(j_scores[optimal_idx])
                },
                'auc_interpretation': self._interpret_auc(auc_score)
            }
        except Exception as e:
            self.logger.error(f"Error calculating ROC analysis: {e}")
            return {'error': str(e)}
    
    def _calculate_pr_analysis(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
        """Calculate detailed Precision-Recall analysis."""
        try:
            # Basic PR calculation
            precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
            ap_score = average_precision_score(y_true, y_prob)
            
            # Bootstrap confidence interval for AP
            ap_ci = self._bootstrap_ap_ci(y_true, y_prob)
            
            # Find optimal threshold (F1-score maximization)
            f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx]
            optimal_precision = precision[optimal_idx]
            optimal_recall = recall[optimal_idx]
            optimal_f1 = f1_scores[optimal_idx]
            
            # Calculate baseline (random classifier performance)
            baseline_ap = np.sum(y_true) / len(y_true)
            
            return {
                'average_precision': float(ap_score),
                'ap_confidence_interval': ap_ci,
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'thresholds': thresholds.tolist(),
                'optimal_threshold': {
                    'threshold': float(optimal_threshold),
                    'precision': float(optimal_precision),
                    'recall': float(optimal_recall),
                    'f1_score': float(optimal_f1)
                },
                'baseline_ap': float(baseline_ap),
                'ap_improvement': float(ap_score - baseline_ap)
            }
        except Exception as e:
            self.logger.error(f"Error calculating PR analysis: {e}")
            return {'error': str(e)}
    
    def _calculate_calibration_analysis(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
        """Calculate model calibration analysis."""
        try:
            # Reliability diagram (calibration curve)
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_prob, n_bins=10
            )
            
            # Brier score (lower is better)
            brier_score = brier_score_loss(y_true, y_prob)
            
            # Expected Calibration Error (ECE)
            ece = self._calculate_ece(y_true, y_prob, n_bins=10)
            
            # Maximum Calibration Error (MCE)
            mce = self._calculate_mce(y_true, y_prob, n_bins=10)
            
            # Calibration slope and intercept
            calibration_slope, calibration_intercept = self._calculate_calibration_slope(y_true, y_prob)
            
            return {
                'brier_score': float(brier_score),
                'expected_calibration_error': float(ece),
                'maximum_calibration_error': float(mce),
                'calibration_slope': float(calibration_slope),
                'calibration_intercept': float(calibration_intercept),
                'reliability_diagram': {
                    'fraction_of_positives': fraction_of_positives.tolist(),
                    'mean_predicted_value': mean_predicted_value.tolist()
                },
                'calibration_interpretation': self._interpret_calibration(ece, brier_score)
            }
        except Exception as e:
            self.logger.error(f"Error calculating calibration analysis: {e}")
            return {'error': str(e)}
    
    def _calculate_detailed_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate detailed per-class metrics with confidence intervals."""
        try:
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            if cm.shape != (2, 2):
                return {'error': 'Only binary classification supported'}
            
            tn, fp, fn, tp = cm.ravel()
            
            # Calculate metrics with confidence intervals
            metrics = {}
            
            # Sensitivity (Recall, True Positive Rate)
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            sensitivity_ci = self._wilson_ci(tp, tp + fn)
            
            # Specificity (True Negative Rate)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificity_ci = self._wilson_ci(tn, tn + fp)
            
            # Positive Predictive Value (Precision)
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            ppv_ci = self._wilson_ci(tp, tp + fp)
            
            # Negative Predictive Value
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            npv_ci = self._wilson_ci(tn, tn + fn)
            
            # Likelihood ratios
            lr_positive = sensitivity / (1 - specificity) if specificity < 1 else float('inf')
            lr_negative = (1 - sensitivity) / specificity if specificity > 0 else float('inf')
            
            # Diagnostic odds ratio
            dor = (tp * tn) / (fp * fn) if (fp * fn) > 0 else float('inf')
            
            return {
                'confusion_matrix': cm.tolist(),
                'sensitivity': {
                    'value': float(sensitivity),
                    'confidence_interval': sensitivity_ci
                },
                'specificity': {
                    'value': float(specificity),
                    'confidence_interval': specificity_ci
                },
                'positive_predictive_value': {
                    'value': float(ppv),
                    'confidence_interval': ppv_ci
                },
                'negative_predictive_value': {
                    'value': float(npv),
                    'confidence_interval': npv_ci
                },
                'likelihood_ratio_positive': float(lr_positive),
                'likelihood_ratio_negative': float(lr_negative),
                'diagnostic_odds_ratio': float(dor),
                'prevalence': float((tp + fn) / (tp + tn + fp + fn))
            }
        except Exception as e:
            self.logger.error(f"Error calculating detailed class metrics: {e}")
            return {'error': str(e)}
    
    def _calculate_confidence_intervals(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_prob: np.ndarray = None
    ) -> Dict[str, Any]:
        """Calculate confidence intervals for key performance metrics."""
        try:
            n = len(y_true)
            
            # Accuracy confidence interval
            accuracy = accuracy_score(y_true, y_pred)
            accuracy_ci = self._wilson_ci(int(accuracy * n), n)
            
            # F1-score confidence interval (bootstrap)
            f1_ci = self._bootstrap_metric_ci(y_true, y_pred, f1_score)
            
            # Precision confidence interval (bootstrap)
            precision_ci = self._bootstrap_metric_ci(y_true, y_pred, precision_score, zero_division=0)
            
            # Recall confidence interval (bootstrap)
            recall_ci = self._bootstrap_metric_ci(y_true, y_pred, recall_score, zero_division=0)
            
            confidence_intervals = {
                'accuracy': {
                    'value': float(accuracy),
                    'confidence_interval': accuracy_ci
                },
                'f1_score': {
                    'value': float(f1_score(y_true, y_pred, zero_division=0)),
                    'confidence_interval': f1_ci
                },
                'precision': {
                    'value': float(precision_score(y_true, y_pred, zero_division=0)),
                    'confidence_interval': precision_ci
                },
                'recall': {
                    'value': float(recall_score(y_true, y_pred, zero_division=0)),
                    'confidence_interval': recall_ci
                }
            }
            
            return confidence_intervals
        except Exception as e:
            self.logger.error(f"Error calculating confidence intervals: {e}")
            return {'error': str(e)}
    
    def _calculate_error_analysis(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_prob: np.ndarray = None
    ) -> Dict[str, Any]:
        """Analyze prediction errors and their characteristics."""
        try:
            # Basic error counts
            correct_predictions = (y_true == y_pred)
            total_errors = np.sum(~correct_predictions)
            error_rate = total_errors / len(y_true)
            
            # Type I and Type II errors
            false_positives = np.sum((y_true == 0) & (y_pred == 1))
            false_negatives = np.sum((y_true == 1) & (y_pred == 0))
            
            error_analysis = {
                'total_errors': int(total_errors),
                'error_rate': float(error_rate),
                'false_positives': int(false_positives),
                'false_negatives': int(false_negatives),
                'type_i_error_rate': float(false_positives / np.sum(y_true == 0)) if np.sum(y_true == 0) > 0 else 0,
                'type_ii_error_rate': float(false_negatives / np.sum(y_true == 1)) if np.sum(y_true == 1) > 0 else 0
            }
            
            # Confidence-based error analysis
            if y_prob is not None:
                # Errors by confidence level
                high_confidence_errors = np.sum(~correct_predictions & ((y_prob >= 0.8) | (y_prob <= 0.2)))
                low_confidence_errors = np.sum(~correct_predictions & (y_prob > 0.2) & (y_prob < 0.8))
                
                error_analysis.update({
                    'high_confidence_errors': int(high_confidence_errors),
                    'low_confidence_errors': int(low_confidence_errors),
                    'high_confidence_error_rate': float(high_confidence_errors / total_errors) if total_errors > 0 else 0
                })
                
                # Average confidence for correct vs incorrect predictions
                if np.sum(correct_predictions) > 0:
                    avg_confidence_correct = np.mean(np.maximum(y_prob[correct_predictions], 1 - y_prob[correct_predictions]))
                else:
                    avg_confidence_correct = 0
                
                if np.sum(~correct_predictions) > 0:
                    avg_confidence_incorrect = np.mean(np.maximum(y_prob[~correct_predictions], 1 - y_prob[~correct_predictions]))
                else:
                    avg_confidence_incorrect = 0
                
                error_analysis.update({
                    'avg_confidence_correct': float(avg_confidence_correct),
                    'avg_confidence_incorrect': float(avg_confidence_incorrect)
                })
            
            return error_analysis
        except Exception as e:
            self.logger.error(f"Error calculating error analysis: {e}")
            return {'error': str(e)}
    
    def _calculate_threshold_optimization(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
        """Comprehensive threshold optimization analysis."""
        try:
            thresholds = np.arange(0.01, 1.0, 0.01)
            
            metrics_by_threshold = {}
            
            for threshold in thresholds:
                y_pred_thresh = (y_prob >= threshold).astype(int)
                
                # Calculate metrics for this threshold
                acc = accuracy_score(y_true, y_pred_thresh)
                prec = precision_score(y_true, y_pred_thresh, zero_division=0)
                rec = recall_score(y_true, y_pred_thresh, zero_division=0)
                f1 = f1_score(y_true, y_pred_thresh, zero_division=0)
                
                # Calculate specificity
                cm = confusion_matrix(y_true, y_pred_thresh)
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                else:
                    spec = 0
                
                metrics_by_threshold[float(threshold)] = {
                    'accuracy': float(acc),
                    'precision': float(prec),
                    'recall': float(rec),
                    'f1_score': float(f1),
                    'specificity': float(spec)
                }
            
            # Find optimal thresholds for different metrics
            optimal_thresholds = {}
            
            for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'specificity']:
                values = [metrics_by_threshold[t][metric] for t in thresholds]
                optimal_idx = np.argmax(values)
                optimal_thresholds[metric] = {
                    'threshold': float(thresholds[optimal_idx]),
                    'value': float(values[optimal_idx])
                }
            
            return {
                'metrics_by_threshold': metrics_by_threshold,
                'optimal_thresholds': optimal_thresholds
            }
        except Exception as e:
            self.logger.error(f"Error calculating threshold optimization: {e}")
            return {'error': str(e)}
    
    def compare_models_statistical(
        self,
        y_true: np.ndarray,
        y_pred1: np.ndarray,
        y_pred2: np.ndarray,
        model1_name: str = "Model 1",
        model2_name: str = "Model 2"
    ) -> Dict[str, Any]:
        """
        Statistical comparison between two models.
        
        Args:
            y_true: True labels
            y_pred1: Predictions from first model
            y_pred2: Predictions from second model
            model1_name: Name of first model
            model2_name: Name of second model
            
        Returns:
            Dictionary containing statistical comparison results
        """
        try:
            # Convert to numpy arrays
            y_true = self._to_numpy(y_true)
            y_pred1 = self._to_numpy(y_pred1)
            y_pred2 = self._to_numpy(y_pred2)
            
            # McNemar's test for comparing two models
            mcnemar_result = self._mcnemar_test(y_true, y_pred1, y_pred2)
            
            # Performance comparison
            acc1 = accuracy_score(y_true, y_pred1)
            acc2 = accuracy_score(y_true, y_pred2)
            
            f1_1 = f1_score(y_true, y_pred1, zero_division=0)
            f1_2 = f1_score(y_true, y_pred2, zero_division=0)
            
            # Effect size (Cohen's h for proportions)
            effect_size = self._cohens_h(acc1, acc2)
            
            return {
                'model_comparison': {
                    model1_name: {
                        'accuracy': float(acc1),
                        'f1_score': float(f1_1)
                    },
                    model2_name: {
                        'accuracy': float(acc2),
                        'f1_score': float(f1_2)
                    }
                },
                'mcnemar_test': mcnemar_result,
                'effect_size': {
                    'cohens_h': float(effect_size),
                    'interpretation': self._interpret_effect_size(effect_size)
                },
                'performance_difference': {
                    'accuracy_diff': float(acc2 - acc1),
                    'f1_diff': float(f1_2 - f1_1)
                }
            }
        except Exception as e:
            self.logger.error(f"Error in statistical model comparison: {e}")
            return {'error': str(e)}
    
    # Helper methods
    def _bootstrap_auc_ci(self, y_true: np.ndarray, y_prob: np.ndarray, n_bootstrap: int = 1000) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for AUC."""
        bootstrap_aucs = []
        n_samples = len(y_true)
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_prob_boot = y_prob[indices]
            
            # Skip if all labels are the same
            if len(np.unique(y_true_boot)) < 2:
                continue
                
            try:
                auc_boot = roc_auc_score(y_true_boot, y_prob_boot)
                bootstrap_aucs.append(auc_boot)
            except:
                continue
        
        if len(bootstrap_aucs) == 0:
            return (0.0, 1.0)
        
        lower = np.percentile(bootstrap_aucs, (self.alpha/2) * 100)
        upper = np.percentile(bootstrap_aucs, (1 - self.alpha/2) * 100)
        
        return (float(lower), float(upper))
    
    def _bootstrap_ap_ci(self, y_true: np.ndarray, y_prob: np.ndarray, n_bootstrap: int = 1000) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for Average Precision."""
        bootstrap_aps = []
        n_samples = len(y_true)
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_prob_boot = y_prob[indices]
            
            # Skip if all labels are the same
            if len(np.unique(y_true_boot)) < 2:
                continue
                
            try:
                ap_boot = average_precision_score(y_true_boot, y_prob_boot)
                bootstrap_aps.append(ap_boot)
            except:
                continue
        
        if len(bootstrap_aps) == 0:
            return (0.0, 1.0)
        
        lower = np.percentile(bootstrap_aps, (self.alpha/2) * 100)
        upper = np.percentile(bootstrap_aps, (1 - self.alpha/2) * 100)
        
        return (float(lower), float(upper))
    
    def _bootstrap_metric_ci(self, y_true: np.ndarray, y_pred: np.ndarray, metric_func, n_bootstrap: int = 1000, **kwargs) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for any metric function."""
        bootstrap_metrics = []
        n_samples = len(y_true)
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            
            try:
                metric_boot = metric_func(y_true_boot, y_pred_boot, **kwargs)
                bootstrap_metrics.append(metric_boot)
            except:
                continue
        
        if len(bootstrap_metrics) == 0:
            return (0.0, 1.0)
        
        lower = np.percentile(bootstrap_metrics, (self.alpha/2) * 100)
        upper = np.percentile(bootstrap_metrics, (1 - self.alpha/2) * 100)
        
        return (float(lower), float(upper))
    
    def _wilson_ci(self, successes: int, trials: int) -> Tuple[float, float]:
        """Calculate Wilson confidence interval for proportions."""
        if trials == 0:
            return (0.0, 1.0)
        
        p = successes / trials
        z = stats.norm.ppf(1 - self.alpha/2)
        
        denominator = 1 + z**2 / trials
        centre = (p + z**2 / (2 * trials)) / denominator
        delta = z * np.sqrt((p * (1 - p) + z**2 / (4 * trials)) / trials) / denominator
        
        lower = max(0, centre - delta)
        upper = min(1, centre + delta)
        
        return (float(lower), float(upper))
    
    def _calculate_ece(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return float(ece)
    
    def _calculate_mce(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Maximum Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mce = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            
            if in_bin.sum() > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        return float(mce)
    
    def _calculate_calibration_slope(self, y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
        """Calculate calibration slope and intercept using logistic regression."""
        from sklearn.linear_model import LogisticRegression
        
        # Convert probabilities to logits
        logits = np.log(y_prob / (1 - y_prob + 1e-8))
        
        # Fit logistic regression
        lr = LogisticRegression()
        lr.fit(logits.reshape(-1, 1), y_true)
        
        slope = lr.coef_[0][0]
        intercept = lr.intercept_[0]
        
        return slope, intercept
    
    def _mcnemar_test(self, y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray) -> Dict[str, Any]:
        """Perform McNemar's test for comparing two models."""
        # Create contingency table
        correct1 = (y_true == y_pred1)
        correct2 = (y_true == y_pred2)
        
        # McNemar's table
        both_correct = np.sum(correct1 & correct2)
        model1_correct_only = np.sum(correct1 & ~correct2)
        model2_correct_only = np.sum(~correct1 & correct2)
        both_incorrect = np.sum(~correct1 & ~correct2)
        
        # McNemar's test statistic with continuity correction
        b = model1_correct_only  # Model 1 correct, Model 2 incorrect
        c = model2_correct_only  # Model 1 incorrect, Model 2 correct
        
        if b + c > 0:
            # McNemar's test statistic with continuity correction
            if b + c >= 25:
                # Use normal approximation with continuity correction
                mcnemar_stat = (abs(b - c) - 1)**2 / (b + c)
            else:
                # Use exact binomial test for small samples
                mcnemar_stat = (abs(b - c))**2 / (b + c) if (b + c) > 0 else 0
            
            # Calculate p-value using chi-square distribution
            p_value = 1 - stats.chi2.cdf(mcnemar_stat, 1)
        else:
            mcnemar_stat = 0
            p_value = 1.0
        
        return {
            'statistic': float(mcnemar_stat),
            'p_value': float(p_value),
            'significant': p_value < self.alpha,
            'contingency_table': {
                'both_correct': int(both_correct),
                'model1_only_correct': int(model1_correct_only),
                'model2_only_correct': int(model2_correct_only),
                'both_incorrect': int(both_incorrect)
            }
        }
    
    def _cohens_h(self, p1: float, p2: float) -> float:
        """Calculate Cohen's h effect size for proportions."""
        return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))
    
    def _interpret_auc(self, auc: float) -> str:
        """Interpret AUC score."""
        if auc >= 0.9:
            return "Excellent"
        elif auc >= 0.8:
            return "Good"
        elif auc >= 0.7:
            return "Fair"
        elif auc >= 0.6:
            return "Poor"
        else:
            return "Fail"
    
    def _interpret_calibration(self, ece: float, brier_score: float) -> str:
        """Interpret calibration quality."""
        if ece <= 0.05:
            return "Well calibrated"
        elif ece <= 0.1:
            return "Moderately calibrated"
        else:
            return "Poorly calibrated"
    
    def _interpret_effect_size(self, cohens_h: float) -> str:
        """Interpret Cohen's h effect size."""
        abs_h = abs(cohens_h)
        if abs_h < 0.2:
            return "Small"
        elif abs_h < 0.5:
            return "Medium"
        else:
            return "Large"
    
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


# Convenience functions
def calculate_advanced_metrics(
    y_true: Union[np.ndarray, torch.Tensor, List],
    y_pred: Union[np.ndarray, torch.Tensor, List],
    y_prob: Union[np.ndarray, torch.Tensor, List] = None,
    class_names: List[str] = None,
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """
    Convenience function to calculate advanced metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        class_names: Names for the classes
        confidence_level: Confidence level for statistical tests
        
    Returns:
        Dictionary containing advanced metrics
    """
    calculator = AdvancedMetricsCalculator(class_names, confidence_level)
    return calculator.calculate_advanced_metrics(y_true, y_pred, y_prob)


def compare_models_statistical(
    y_true: Union[np.ndarray, torch.Tensor, List],
    y_pred1: Union[np.ndarray, torch.Tensor, List],
    y_pred2: Union[np.ndarray, torch.Tensor, List],
    model1_name: str = "Model 1",
    model2_name: str = "Model 2",
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """
    Convenience function for statistical model comparison.
    
    Args:
        y_true: True labels
        y_pred1: Predictions from first model
        y_pred2: Predictions from second model
        model1_name: Name of first model
        model2_name: Name of second model
        confidence_level: Confidence level for statistical tests
        
    Returns:
        Dictionary containing statistical comparison results
    """
    calculator = AdvancedMetricsCalculator(confidence_level=confidence_level)
    return calculator.compare_models_statistical(y_true, y_pred1, y_pred2, model1_name, model2_name)