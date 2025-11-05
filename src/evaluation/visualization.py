"""
Visualization and reporting tools for LSTM sentiment classifier evaluation.

This module provides functions for creating plots, charts, and comprehensive
evaluation reports for model performance analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import os
from datetime import datetime
import json
import logging

from .metrics import MetricsCalculator


class EvaluationVisualizer:
    """
    Comprehensive visualization tools for model evaluation.
    
    Provides methods to create various plots and charts for analyzing
    model performance and training progress.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (10, 8)):
        """
        Initialize evaluation visualizer.
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        self.logger = logging.getLogger(__name__)
        
        # Set style
        try:
            plt.style.use(style)
        except OSError:
            # Fallback to default style if seaborn not available
            plt.style.use('default')
            self.logger.warning(f"Style '{style}' not available, using default")
        
        # Set color palette
        self.colors = plt.cm.Set2(np.linspace(0, 1, 8))
    
    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        class_names: List[str] = None,
        normalize: bool = False,
        title: str = 'Confusion Matrix',
        save_path: str = None,
        show_plot: bool = True,
        figsize: Tuple[int, int] = None
    ) -> plt.Figure:
        """
        Plot confusion matrix with proper formatting.
        
        Args:
            confusion_matrix: 2D array representing confusion matrix
            class_names: Names for the classes
            normalize: Whether to normalize the matrix
            title: Plot title
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            figsize: Figure size override
            
        Returns:
            Matplotlib figure object
        """
        if figsize is None:
            figsize = (8, 6)
        
        if class_names is None:
            class_names = ['Negative', 'Positive']
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Normalize if requested
        cm_display = confusion_matrix.copy()
        if normalize:
            cm_display = cm_display.astype('float') / cm_display.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
        else:
            fmt = 'd'
        
        # Create heatmap
        sns.heatmap(
            cm_display,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            cbar_kws={'label': 'Percentage' if normalize else 'Count'}
        )
        
        # Set labels and title
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add statistics text
        if not normalize and confusion_matrix.shape == (2, 2):
            tn, fp, fn, tp = confusion_matrix.ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            stats_text = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1-Score: {f1:.3f}'
            ax.text(2.1, 0.5, stats_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Confusion matrix saved to {save_path}")
        
        # Show if requested
        if show_plot:
            plt.show()
        
        return fig
    
    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        title: str = 'Training History',
        save_path: str = None,
        show_plot: bool = True,
        figsize: Tuple[int, int] = None
    ) -> plt.Figure:
        """
        Plot training history including loss and accuracy curves.
        
        Args:
            history: Dictionary containing training history
            title: Plot title
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            figsize: Figure size override
            
        Returns:
            Matplotlib figure object
        """
        if figsize is None:
            figsize = self.figsize
        
        # Determine number of subplots needed
        metrics_to_plot = []
        if 'train_losses' in history and 'val_losses' in history:
            metrics_to_plot.append('loss')
        if 'val_accuracies' in history:
            metrics_to_plot.append('accuracy')
        if 'learning_rates' in history:
            metrics_to_plot.append('learning_rate')
        
        n_plots = len(metrics_to_plot)
        if n_plots == 0:
            raise ValueError("No plottable metrics found in history")
        
        # Create subplots
        fig, axes = plt.subplots(n_plots, 1, figsize=(figsize[0], figsize[1] * n_plots / 2))
        if n_plots == 1:
            axes = [axes]
        
        epochs = range(1, len(history.get('train_losses', history.get('val_losses', [1]))) + 1)
        
        plot_idx = 0
        
        # Plot loss curves
        if 'loss' in metrics_to_plot:
            ax = axes[plot_idx]
            
            if 'train_losses' in history:
                ax.plot(epochs, history['train_losses'], 'b-', label='Training Loss', linewidth=2)
            if 'val_losses' in history:
                ax.plot(epochs, history['val_losses'], 'r-', label='Validation Loss', linewidth=2)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training and Validation Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Highlight best epoch
            if 'val_losses' in history:
                best_epoch = np.argmin(history['val_losses']) + 1
                best_loss = min(history['val_losses'])
                ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
                ax.plot(best_epoch, best_loss, 'go', markersize=8)
                ax.legend()
            
            plot_idx += 1
        
        # Plot accuracy curves
        if 'accuracy' in metrics_to_plot:
            ax = axes[plot_idx]
            
            if 'val_accuracies' in history:
                ax.plot(epochs, history['val_accuracies'], 'g-', label='Validation Accuracy', linewidth=2)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy (%)')
            ax.set_title('Validation Accuracy')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Highlight best epoch
            if 'val_accuracies' in history:
                best_epoch = np.argmax(history['val_accuracies']) + 1
                best_acc = max(history['val_accuracies'])
                ax.axvline(x=best_epoch, color='purple', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
                ax.plot(best_epoch, best_acc, 'ro', markersize=8)
                ax.legend()
            
            plot_idx += 1
        
        # Plot learning rate
        if 'learning_rate' in metrics_to_plot:
            ax = axes[plot_idx]
            
            ax.plot(epochs, history['learning_rates'], 'orange', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Training history plot saved to {save_path}")
        
        # Show if requested
        if show_plot:
            plt.show()
        
        return fig
    
    def plot_roc_curve(
        self,
        fpr: List[float],
        tpr: List[float],
        auc_score: float = None,
        title: str = 'ROC Curve',
        save_path: str = None,
        show_plot: bool = True,
        figsize: Tuple[int, int] = None
    ) -> plt.Figure:
        """
        Plot ROC curve.
        
        Args:
            fpr: False positive rates
            tpr: True positive rates
            auc_score: AUC score to display
            title: Plot title
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            figsize: Figure size override
            
        Returns:
            Matplotlib figure object
        """
        if figsize is None:
            figsize = (8, 8)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot ROC curve
        label = f'ROC Curve (AUC = {auc_score:.3f})' if auc_score else 'ROC Curve'
        ax.plot(fpr, tpr, 'b-', linewidth=2, label=label)
        
        # Plot diagonal line (random classifier)
        ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
        
        # Set labels and formatting
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"ROC curve saved to {save_path}")
        
        # Show if requested
        if show_plot:
            plt.show()
        
        return fig
    
    def plot_precision_recall_curve(
        self,
        precision: List[float],
        recall: List[float],
        ap_score: float = None,
        title: str = 'Precision-Recall Curve',
        save_path: str = None,
        show_plot: bool = True,
        figsize: Tuple[int, int] = None
    ) -> plt.Figure:
        """
        Plot Precision-Recall curve.
        
        Args:
            precision: Precision values
            recall: Recall values
            ap_score: Average precision score to display
            title: Plot title
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            figsize: Figure size override
            
        Returns:
            Matplotlib figure object
        """
        if figsize is None:
            figsize = (8, 8)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot PR curve
        label = f'PR Curve (AP = {ap_score:.3f})' if ap_score else 'PR Curve'
        ax.plot(recall, precision, 'b-', linewidth=2, label=label)
        
        # Plot baseline (random classifier)
        baseline = len([p for p in precision if p > 0]) / len(precision) if precision else 0.5
        ax.axhline(y=baseline, color='r', linestyle='--', linewidth=1, label=f'Baseline (AP = {baseline:.3f})')
        
        # Set labels and formatting
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"PR curve saved to {save_path}")
        
        # Show if requested
        if show_plot:
            plt.show()
        
        return fig
    
    def plot_threshold_analysis(
        self,
        threshold_metrics: Dict[float, Dict[str, float]],
        metrics_to_plot: List[str] = None,
        title: str = 'Threshold Analysis',
        save_path: str = None,
        show_plot: bool = True,
        figsize: Tuple[int, int] = None
    ) -> plt.Figure:
        """
        Plot metrics across different thresholds.
        
        Args:
            threshold_metrics: Dictionary mapping thresholds to metrics
            metrics_to_plot: List of metrics to plot
            title: Plot title
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            figsize: Figure size override
            
        Returns:
            Matplotlib figure object
        """
        if figsize is None:
            figsize = self.figsize
        
        if metrics_to_plot is None:
            metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
        
        # Extract data
        thresholds = sorted(threshold_metrics.keys())
        metrics_data = {metric: [] for metric in metrics_to_plot}
        
        for threshold in thresholds:
            for metric in metrics_to_plot:
                if metric in threshold_metrics[threshold]:
                    metrics_data[metric].append(threshold_metrics[threshold][metric])
                else:
                    metrics_data[metric].append(0)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(metrics_to_plot)))
        
        for i, metric in enumerate(metrics_to_plot):
            ax.plot(thresholds, metrics_data[metric], 'o-', 
                   color=colors[i], linewidth=2, markersize=4, label=metric.replace('_', ' ').title())
        
        # Find and mark optimal points
        for i, metric in enumerate(metrics_to_plot):
            if metrics_data[metric]:
                best_idx = np.argmax(metrics_data[metric])
                best_threshold = thresholds[best_idx]
                best_value = metrics_data[metric][best_idx]
                ax.plot(best_threshold, best_value, 's', color=colors[i], markersize=8, 
                       markeredgecolor='black', markeredgewidth=1)
        
        # Set labels and formatting
        ax.set_xlabel('Threshold', fontsize=12)
        ax.set_ylabel('Metric Value', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([min(thresholds), max(thresholds)])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Threshold analysis plot saved to {save_path}")
        
        # Show if requested
        if show_plot:
            plt.show()
        
        return fig
    
    def create_evaluation_dashboard(
        self,
        metrics: Dict[str, Any],
        history: Dict[str, List[float]] = None,
        save_dir: str = None,
        show_plots: bool = True
    ) -> Dict[str, plt.Figure]:
        """
        Create comprehensive evaluation dashboard with multiple plots.
        
        Args:
            metrics: Comprehensive metrics dictionary
            history: Training history (optional)
            save_dir: Directory to save plots
            show_plots: Whether to display plots
            
        Returns:
            Dictionary of figure objects
        """
        figures = {}
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # 1. Confusion Matrix
        if 'confusion_matrix' in metrics:
            cm = np.array(metrics['confusion_matrix'])
            save_path = os.path.join(save_dir, 'confusion_matrix.png') if save_dir else None
            
            fig = self.plot_confusion_matrix(
                cm, 
                title='Confusion Matrix',
                save_path=save_path,
                show_plot=show_plots
            )
            figures['confusion_matrix'] = fig
        
        # 2. Training History
        if history:
            save_path = os.path.join(save_dir, 'training_history.png') if save_dir else None
            
            fig = self.plot_training_history(
                history,
                title='Training Progress',
                save_path=save_path,
                show_plot=show_plots
            )
            figures['training_history'] = fig
        
        # 3. ROC Curve
        if 'roc_curve' in metrics and metrics['roc_curve']:
            roc_data = metrics['roc_curve']
            auc_score = metrics['basic_metrics'].get('roc_auc')
            save_path = os.path.join(save_dir, 'roc_curve.png') if save_dir else None
            
            fig = self.plot_roc_curve(
                roc_data['fpr'],
                roc_data['tpr'],
                auc_score=auc_score,
                save_path=save_path,
                show_plot=show_plots
            )
            figures['roc_curve'] = fig
        
        # 4. Precision-Recall Curve
        if 'pr_curve' in metrics and metrics['pr_curve']:
            pr_data = metrics['pr_curve']
            ap_score = metrics['basic_metrics'].get('pr_auc')
            save_path = os.path.join(save_dir, 'pr_curve.png') if save_dir else None
            
            fig = self.plot_precision_recall_curve(
                pr_data['precision'],
                pr_data['recall'],
                ap_score=ap_score,
                save_path=save_path,
                show_plot=show_plots
            )
            figures['pr_curve'] = fig
        
        return figures


class EvaluationReporter:
    """
    Comprehensive evaluation report generator.
    
    Creates detailed reports in various formats (text, JSON, HTML)
    summarizing model performance and evaluation results.
    """
    
    def __init__(self):
        """Initialize evaluation reporter."""
        self.logger = logging.getLogger(__name__)
    
    def generate_text_report(
        self,
        metrics: Dict[str, Any],
        model_info: Dict[str, Any] = None,
        training_summary: Dict[str, Any] = None
    ) -> str:
        """
        Generate comprehensive text evaluation report.
        
        Args:
            metrics: Comprehensive metrics dictionary
            model_info: Model configuration information
            training_summary: Training summary information
            
        Returns:
            Formatted text report
        """
        report_lines = []
        
        # Header
        report_lines.append("=" * 80)
        report_lines.append("LSTM SENTIMENT CLASSIFIER - EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Model Information
        if model_info:
            report_lines.append("MODEL CONFIGURATION")
            report_lines.append("-" * 40)
            for key, value in model_info.items():
                report_lines.append(f"{key:25}: {value}")
            report_lines.append("")
        
        # Training Summary
        if training_summary:
            report_lines.append("TRAINING SUMMARY")
            report_lines.append("-" * 40)
            
            if 'training_status' in training_summary:
                status = training_summary['training_status']
                report_lines.append(f"{'Epochs Completed':25}: {status.get('epochs_completed', 'N/A')}")
                report_lines.append(f"{'Best Epoch':25}: {status.get('best_epoch', 'N/A')}")
            
            if 'performance_metrics' in training_summary:
                perf = training_summary['performance_metrics']
                report_lines.append(f"{'Best Val Loss':25}: {perf.get('best_val_loss', 'N/A'):.6f}")
                report_lines.append(f"{'Best Val Accuracy':25}: {perf.get('best_val_accuracy', 'N/A'):.2f}%")
            
            report_lines.append("")
        
        # Basic Metrics
        if 'basic_metrics' in metrics:
            basic = metrics['basic_metrics']
            report_lines.append("PERFORMANCE METRICS")
            report_lines.append("-" * 40)
            report_lines.append(f"{'Accuracy':25}: {basic.get('accuracy', 0):.4f} ({basic.get('accuracy', 0)*100:.2f}%)")
            report_lines.append(f"{'Precision':25}: {basic.get('precision', 0):.4f}")
            report_lines.append(f"{'Recall':25}: {basic.get('recall', 0):.4f}")
            report_lines.append(f"{'F1-Score':25}: {basic.get('f1_score', 0):.4f}")
            report_lines.append(f"{'Specificity':25}: {basic.get('specificity', 0):.4f}")
            
            if basic.get('roc_auc') is not None:
                report_lines.append(f"{'ROC AUC':25}: {basic.get('roc_auc', 0):.4f}")
            if basic.get('pr_auc') is not None:
                report_lines.append(f"{'PR AUC':25}: {basic.get('pr_auc', 0):.4f}")
            
            report_lines.append(f"{'Support':25}: {basic.get('support', 0)} samples")
            report_lines.append("")
        
        # Confusion Matrix Statistics
        if 'confusion_matrix_stats' in metrics:
            cm_stats = metrics['confusion_matrix_stats']
            report_lines.append("CONFUSION MATRIX STATISTICS")
            report_lines.append("-" * 40)
            report_lines.append(f"{'True Positives':25}: {cm_stats.get('true_positives', 0)}")
            report_lines.append(f"{'True Negatives':25}: {cm_stats.get('true_negatives', 0)}")
            report_lines.append(f"{'False Positives':25}: {cm_stats.get('false_positives', 0)}")
            report_lines.append(f"{'False Negatives':25}: {cm_stats.get('false_negatives', 0)}")
            report_lines.append(f"{'Total Samples':25}: {cm_stats.get('total_samples', 0)}")
            report_lines.append("")
        
        # Per-Class Metrics
        if 'per_class_metrics' in metrics:
            per_class = metrics['per_class_metrics']
            report_lines.append("PER-CLASS METRICS")
            report_lines.append("-" * 40)
            
            for class_name, class_metrics in per_class.items():
                report_lines.append(f"{class_name}:")
                report_lines.append(f"  {'Precision':20}: {class_metrics.get('precision', 0):.4f}")
                report_lines.append(f"  {'Recall':20}: {class_metrics.get('recall', 0):.4f}")
                report_lines.append(f"  {'F1-Score':20}: {class_metrics.get('f1_score', 0):.4f}")
                report_lines.append(f"  {'Support':20}: {class_metrics.get('support', 0)}")
                report_lines.append("")
        
        # Threshold Information
        if 'threshold' in metrics:
            report_lines.append("THRESHOLD INFORMATION")
            report_lines.append("-" * 40)
            report_lines.append(f"{'Decision Threshold':25}: {metrics['threshold']}")
            report_lines.append("")
        
        # Footer
        report_lines.append("=" * 80)
        report_lines.append("End of Report")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def generate_json_report(
        self,
        metrics: Dict[str, Any],
        model_info: Dict[str, Any] = None,
        training_summary: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate JSON evaluation report.
        
        Args:
            metrics: Comprehensive metrics dictionary
            model_info: Model configuration information
            training_summary: Training summary information
            
        Returns:
            Dictionary suitable for JSON serialization
        """
        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'report_type': 'evaluation_report',
                'version': '1.0'
            },
            'evaluation_metrics': metrics
        }
        
        if model_info:
            report['model_info'] = model_info
        
        if training_summary:
            report['training_summary'] = training_summary
        
        return report
    
    def save_report(
        self,
        metrics: Dict[str, Any],
        output_path: str,
        format_type: str = 'text',
        model_info: Dict[str, Any] = None,
        training_summary: Dict[str, Any] = None
    ) -> str:
        """
        Save evaluation report to file.
        
        Args:
            metrics: Comprehensive metrics dictionary
            output_path: Path to save the report
            format_type: Report format ('text' or 'json')
            model_info: Model configuration information
            training_summary: Training summary information
            
        Returns:
            Path to saved report file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if format_type.lower() == 'text':
            report_content = self.generate_text_report(metrics, model_info, training_summary)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
        
        elif format_type.lower() == 'json':
            report_content = self.generate_json_report(metrics, model_info, training_summary)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_content, f, indent=2, ensure_ascii=False)
        
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
        
        self.logger.info(f"Evaluation report saved to {output_path}")
        return output_path


# Convenience functions
def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str] = None,
    normalize: bool = False,
    title: str = 'Confusion Matrix',
    save_path: str = None,
    show_plot: bool = True
) -> plt.Figure:
    """Convenience function to plot confusion matrix."""
    visualizer = EvaluationVisualizer()
    return visualizer.plot_confusion_matrix(
        confusion_matrix, class_names, normalize, title, save_path, show_plot
    )


def plot_training_history(
    history: Dict[str, List[float]],
    title: str = 'Training History',
    save_path: str = None,
    show_plot: bool = True
) -> plt.Figure:
    """Convenience function to plot training history."""
    visualizer = EvaluationVisualizer()
    return visualizer.plot_training_history(history, title, save_path, show_plot)


def generate_evaluation_report(
    metrics: Dict[str, Any],
    output_path: str = None,
    format_type: str = 'text',
    model_info: Dict[str, Any] = None,
    training_summary: Dict[str, Any] = None
) -> str:
    """Convenience function to generate evaluation report."""
    reporter = EvaluationReporter()
    
    if output_path:
        return reporter.save_report(metrics, output_path, format_type, model_info, training_summary)
    else:
        if format_type.lower() == 'text':
            return reporter.generate_text_report(metrics, model_info, training_summary)
        elif format_type.lower() == 'json':
            return reporter.generate_json_report(metrics, model_info, training_summary)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")