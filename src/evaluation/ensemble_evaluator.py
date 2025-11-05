"""
Ensemble Evaluation and Comparison Tools
Utilities for comprehensive evaluation and comparison of ensemble models
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple, Any
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from pathlib import Path
import json

from src.models.ensemble import ModelEnsemble


class EnsembleEvaluator:
    """
    Comprehensive evaluation toolkit for model ensembles.
    """
    
    def __init__(self, device: torch.device = None):
        """
        Initialize ensemble evaluator.
        
        Args:
            device: Device to run evaluations on
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.evaluation_results = {}
    
    def evaluate_ensemble_comprehensive(
        self,
        ensemble: ModelEnsemble,
        test_dataloader,
        ensemble_name: str = "Ensemble"
    ) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation of an ensemble model.
        
        Args:
            ensemble: ModelEnsemble to evaluate
            test_dataloader: Test data loader
            ensemble_name: Name identifier for the ensemble
            
        Returns:
            Dictionary with comprehensive evaluation results
        """
        print(f"Evaluating {ensemble_name}...")
        
        all_predictions = []
        all_probabilities = []
        all_targets = []
        all_uncertainties = []
        individual_results = {'predictions': [], 'probabilities': []}
        
        ensemble.models[0].eval()  # Ensure models are in eval mode
        
        with torch.no_grad():
            for batch_idx, (data, targets, lengths) in enumerate(test_dataloader):
                data, targets = data.to(self.device), targets.to(self.device)
                if lengths is not None:
                    lengths = lengths.to(self.device)
                
                # Get ensemble predictions with uncertainty
                predictions, probabilities, uncertainties = ensemble.predict_with_uncertainty(data, lengths)
                
                # Get individual model results for analysis
                _, _, individual_batch = ensemble.predict(data, lengths, return_individual=True)
                
                # Store results
                all_predictions.extend(predictions.squeeze().cpu().numpy())
                all_probabilities.extend(probabilities.squeeze().cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_uncertainties.extend(uncertainties.squeeze().cpu().numpy())
                
                # Store individual model results
                for i, (ind_pred, ind_prob) in enumerate(zip(
                    individual_batch['predictions'], individual_batch['probabilities']
                )):
                    if len(individual_results['predictions']) <= i:
                        individual_results['predictions'].append([])
                        individual_results['probabilities'].append([])
                    
                    individual_results['predictions'][i].extend(ind_pred.squeeze().cpu().numpy())
                    individual_results['probabilities'][i].extend(ind_prob.squeeze().cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_targets = np.array(all_targets)
        all_uncertainties = np.array(all_uncertainties)
        
        # Calculate comprehensive metrics
        results = self._calculate_comprehensive_metrics(
            all_targets, all_predictions, all_probabilities, all_uncertainties
        )
        
        # Add individual model analysis
        results['individual_model_analysis'] = self._analyze_individual_models(
            individual_results, all_targets, ensemble.model_names
        )
        
        # Add ensemble-specific metrics
        results['ensemble_metrics'] = self._calculate_ensemble_metrics(
            individual_results, all_targets, all_predictions
        )
        
        # Store results
        self.evaluation_results[ensemble_name] = results
        
        return results
    
    def _calculate_comprehensive_metrics(
        self,
        targets: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        uncertainties: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics."""
        
        # Basic classification metrics
        accuracy = np.mean(predictions == targets)
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        tn, fp, fn, tp = cm.ravel()
        
        # Precision, Recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Specificity and Sensitivity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivity = recall  # Same as recall
        
        # ROC AUC
        fpr, tpr, _ = roc_curve(targets, probabilities)
        roc_auc = auc(fpr, tpr)
        
        # Calibration metrics
        calibration_error = self._calculate_calibration_error(targets, probabilities)
        
        # Uncertainty analysis
        uncertainty_stats = {
            'mean_uncertainty': float(np.mean(uncertainties)),
            'std_uncertainty': float(np.std(uncertainties)),
            'uncertainty_correct_correlation': float(np.corrcoef(
                uncertainties, (predictions == targets).astype(float)
            )[0, 1]) if len(uncertainties) > 1 else 0.0
        }
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'specificity': float(specificity),
            'sensitivity': float(sensitivity),
            'roc_auc': float(roc_auc),
            'confusion_matrix': cm.tolist(),
            'calibration_error': float(calibration_error),
            'uncertainty_analysis': uncertainty_stats,
            'roc_curve': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist()
            }
        }
    
    def _calculate_calibration_error(self, targets: np.ndarray, probabilities: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error (ECE)."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = targets[in_bin].mean()
                avg_confidence_in_bin = probabilities[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _analyze_individual_models(
        self,
        individual_results: Dict,
        targets: np.ndarray,
        model_names: List[str]
    ) -> Dict[str, Any]:
        """Analyze individual model performance within the ensemble."""
        
        individual_analysis = {}
        
        for i, model_name in enumerate(model_names):
            model_predictions = np.array(individual_results['predictions'][i])
            model_probabilities = np.array(individual_results['probabilities'][i])
            
            # Calculate metrics for this model
            accuracy = np.mean(model_predictions == targets)
            
            # Confusion matrix
            cm = confusion_matrix(targets, model_predictions)
            tn, fp, fn, tp = cm.ravel()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # ROC AUC
            fpr, tpr, _ = roc_curve(targets, model_probabilities)
            roc_auc = auc(fpr, tpr)
            
            individual_analysis[model_name] = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'roc_auc': float(roc_auc),
                'confusion_matrix': cm.tolist()
            }
        
        return individual_analysis
    
    def _calculate_ensemble_metrics(
        self,
        individual_results: Dict,
        targets: np.ndarray,
        ensemble_predictions: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate ensemble-specific metrics."""
        
        # Model agreement analysis
        individual_preds = np.array(individual_results['predictions']).T  # (n_samples, n_models)
        
        # Calculate agreement statistics
        agreement_rates = []
        for i in range(len(individual_preds)):
            sample_preds = individual_preds[i]
            agreement_rate = np.mean(sample_preds == sample_preds[0])  # Agreement with first model
            agreement_rates.append(agreement_rate)
        
        agreement_rates = np.array(agreement_rates)
        
        # Diversity metrics
        pairwise_disagreements = []
        n_models = len(individual_results['predictions'])
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                disagreement = np.mean(
                    np.array(individual_results['predictions'][i]) != 
                    np.array(individual_results['predictions'][j])
                )
                pairwise_disagreements.append(disagreement)
        
        # Ensemble improvement analysis
        individual_accuracies = []
        for i in range(n_models):
            individual_acc = np.mean(np.array(individual_results['predictions'][i]) == targets)
            individual_accuracies.append(individual_acc)
        
        ensemble_accuracy = np.mean(ensemble_predictions == targets)
        best_individual_accuracy = max(individual_accuracies)
        ensemble_improvement = ensemble_accuracy - best_individual_accuracy
        
        return {
            'model_agreement': {
                'mean_agreement_rate': float(np.mean(agreement_rates)),
                'std_agreement_rate': float(np.std(agreement_rates))
            },
            'model_diversity': {
                'mean_pairwise_disagreement': float(np.mean(pairwise_disagreements)),
                'std_pairwise_disagreement': float(np.std(pairwise_disagreements))
            },
            'ensemble_improvement': {
                'ensemble_accuracy': float(ensemble_accuracy),
                'best_individual_accuracy': float(best_individual_accuracy),
                'improvement': float(ensemble_improvement),
                'individual_accuracies': [float(acc) for acc in individual_accuracies]
            }
        }
    
    def compare_ensembles(
        self,
        ensemble_results: Dict[str, Dict],
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Compare multiple ensemble evaluation results.
        
        Args:
            ensemble_results: Dictionary of ensemble evaluation results
            save_path: Optional path to save comparison results
            
        Returns:
            DataFrame with comparison results
        """
        comparison_data = []
        
        for ensemble_name, results in ensemble_results.items():
            comparison_data.append({
                'Ensemble': ensemble_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score'],
                'ROC-AUC': results['roc_auc'],
                'Calibration Error': results['calibration_error'],
                'Mean Uncertainty': results['uncertainty_analysis']['mean_uncertainty'],
                'Ensemble Improvement': results['ensemble_metrics']['ensemble_improvement']['improvement'],
                'Model Diversity': results['ensemble_metrics']['model_diversity']['mean_pairwise_disagreement']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        if save_path:
            comparison_df.to_csv(save_path, index=False)
            print(f"Ensemble comparison saved to: {save_path}")
        
        return comparison_df
    
    def plot_ensemble_comparison(
        self,
        ensemble_results: Dict[str, Dict],
        metrics: List[str] = None,
        figsize: Tuple[int, int] = (15, 10),
        save_path: Optional[str] = None
    ):
        """
        Create visualization comparing multiple ensembles.
        
        Args:
            ensemble_results: Dictionary of ensemble evaluation results
            metrics: List of metrics to compare
            figsize: Figure size tuple
            save_path: Optional path to save the plot
        """
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        # Prepare data for plotting
        ensemble_names = list(ensemble_results.keys())
        metric_values = {metric: [] for metric in metrics}
        
        for ensemble_name in ensemble_names:
            results = ensemble_results[ensemble_name]
            for metric in metrics:
                metric_values[metric].append(results[metric])
        
        # Create subplots
        n_metrics = len(metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            bars = ax.bar(ensemble_names, metric_values[metric], alpha=0.7)
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_ylabel('Score')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, metric_values[metric]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        
        # Remove empty subplots
        for i in range(n_metrics, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if n_rows > 1:
                fig.delaxes(axes[row, col])
            else:
                fig.delaxes(axes[col])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_roc_comparison(
        self,
        ensemble_results: Dict[str, Dict],
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None
    ):
        """
        Plot ROC curves for multiple ensembles.
        
        Args:
            ensemble_results: Dictionary of ensemble evaluation results
            figsize: Figure size tuple
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=figsize)
        
        for ensemble_name, results in ensemble_results.items():
            roc_data = results['roc_curve']
            plt.plot(
                roc_data['fpr'], 
                roc_data['tpr'], 
                label=f"{ensemble_name} (AUC = {results['roc_auc']:.3f})",
                linewidth=2
            )
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_evaluation_report(
        self,
        ensemble_results: Dict[str, Dict],
        output_dir: str
    ):
        """
        Generate comprehensive evaluation report.
        
        Args:
            ensemble_results: Dictionary of ensemble evaluation results
            output_dir: Directory to save the report
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results as JSON
        results_path = output_path / 'detailed_results.json'
        with open(results_path, 'w') as f:
            json.dump(ensemble_results, f, indent=2)
        
        # Create comparison DataFrame and save as CSV
        comparison_df = self.compare_ensembles(ensemble_results)
        comparison_path = output_path / 'ensemble_comparison.csv'
        comparison_df.to_csv(comparison_path, index=False)
        
        # Generate plots
        plots_dir = output_path / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # Comparison plot
        self.plot_ensemble_comparison(
            ensemble_results,
            save_path=plots_dir / 'ensemble_comparison.png'
        )
        
        # ROC comparison
        self.plot_roc_comparison(
            ensemble_results,
            save_path=plots_dir / 'roc_comparison.png'
        )
        
        # Generate summary report
        self._generate_summary_report(ensemble_results, output_path / 'summary_report.txt')
        
        print(f"Comprehensive evaluation report generated in: {output_path}")
    
    def _generate_summary_report(self, ensemble_results: Dict[str, Dict], output_path: Path):
        """Generate text summary report."""
        
        with open(output_path, 'w') as f:
            f.write("ENSEMBLE EVALUATION SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Overall comparison
            f.write("PERFORMANCE COMPARISON\n")
            f.write("-" * 25 + "\n")
            
            best_ensemble = max(ensemble_results.items(), key=lambda x: x[1]['accuracy'])
            f.write(f"Best performing ensemble: {best_ensemble[0]} (Accuracy: {best_ensemble[1]['accuracy']:.4f})\n\n")
            
            # Detailed results for each ensemble
            for ensemble_name, results in ensemble_results.items():
                f.write(f"\n{ensemble_name.upper()}\n")
                f.write("-" * len(ensemble_name) + "\n")
                f.write(f"Accuracy: {results['accuracy']:.4f}\n")
                f.write(f"Precision: {results['precision']:.4f}\n")
                f.write(f"Recall: {results['recall']:.4f}\n")
                f.write(f"F1-Score: {results['f1_score']:.4f}\n")
                f.write(f"ROC-AUC: {results['roc_auc']:.4f}\n")
                f.write(f"Calibration Error: {results['calibration_error']:.4f}\n")
                
                # Ensemble-specific metrics
                ensemble_metrics = results['ensemble_metrics']
                f.write(f"Ensemble Improvement: {ensemble_metrics['ensemble_improvement']['improvement']:.4f}\n")
                f.write(f"Model Diversity: {ensemble_metrics['model_diversity']['mean_pairwise_disagreement']:.4f}\n")
                
                # Individual model performance
                f.write("\nIndividual Model Performance:\n")
                for model_name, model_results in results['individual_model_analysis'].items():
                    f.write(f"  {model_name}: Accuracy = {model_results['accuracy']:.4f}, "
                           f"F1 = {model_results['f1_score']:.4f}\n")
        
        print(f"Summary report saved to: {output_path}")