"""
Tests for the evaluation module.

These tests verify the correctness of metrics calculation,
visualization generation, and report creation functionality.
"""

import os
import tempfile
import shutil
import numpy as np
import matplotlib.pyplot as plt
import pytest
import json

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from evaluation.metrics import MetricsCalculator, calculate_metrics, find_optimal_threshold
from evaluation.visualization import EvaluationVisualizer, EvaluationReporter


class TestMetricsCalculator:
    """Tests for metrics calculation functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample prediction data for testing."""
        # Perfect predictions
        y_true_perfect = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        y_pred_perfect = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        y_prob_perfect = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.85, 0.15])
        
        # Imperfect predictions
        y_true_imperfect = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
        y_pred_imperfect = np.array([0, 1, 1, 0, 0, 1, 1, 1, 1, 0])
        y_prob_imperfect = np.array([0.2, 0.6, 0.8, 0.4, 0.3, 0.7, 0.85, 0.55, 0.9, 0.1])
        
        return {
            'perfect': (y_true_perfect, y_pred_perfect, y_prob_perfect),
            'imperfect': (y_true_imperfect, y_pred_imperfect, y_prob_imperfect)
        }
    
    def test_basic_metrics_perfect_predictions(self, sample_data):
        """Test basic metrics calculation with perfect predictions."""
        y_true, y_pred, y_prob = sample_data['perfect']
        
        calculator = MetricsCalculator()
        metrics = calculator.calculate_basic_metrics(y_true, y_pred, y_prob)
        
        # Perfect predictions should have accuracy = 1.0
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1_score'] == 1.0
        assert metrics['specificity'] == 1.0
        assert metrics['support'] == len(y_true)
        
        # AUC should be 1.0 for perfect predictions
        assert metrics['roc_auc'] == 1.0
        assert metrics['pr_auc'] == 1.0
    
    def test_basic_metrics_imperfect_predictions(self, sample_data):
        """Test basic metrics calculation with imperfect predictions."""
        y_true, y_pred, y_prob = sample_data['imperfect']
        
        calculator = MetricsCalculator()
        metrics = calculator.calculate_basic_metrics(y_true, y_pred, y_prob)
        
        # Check that metrics are reasonable
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
        assert 0 <= metrics['specificity'] <= 1
        assert metrics['support'] == len(y_true)
        
        # Check specific values for this test case
        # TP=4, TN=3, FP=2, FN=1
        expected_accuracy = 7/10  # (TP + TN) / total
        expected_precision = 4/6  # TP / (TP + FP)
        expected_recall = 4/5     # TP / (TP + FN)
        
        assert abs(metrics['accuracy'] - expected_accuracy) < 1e-6
        assert abs(metrics['precision'] - expected_precision) < 1e-6
        assert abs(metrics['recall'] - expected_recall) < 1e-6
    
    def test_confusion_matrix_calculation(self, sample_data):
        """Test confusion matrix calculation."""
        y_true, y_pred, _ = sample_data['imperfect']
        
        calculator = MetricsCalculator()
        cm, stats = calculator.calculate_confusion_matrix(y_true, y_pred)
        
        # Check confusion matrix shape and values
        assert cm.shape == (2, 2)
        
        # For the imperfect test case: TN=3, FP=2, FN=1, TP=4
        expected_cm = np.array([[3, 2], [1, 4]])
        np.testing.assert_array_equal(cm, expected_cm)
        
        # Check statistics
        assert stats['true_negatives'] == 3
        assert stats['false_positives'] == 2
        assert stats['false_negatives'] == 1
        assert stats['true_positives'] == 4
        assert stats['total_samples'] == 10
    
    def test_per_class_metrics(self, sample_data):
        """Test per-class metrics calculation."""
        y_true, y_pred, _ = sample_data['imperfect']
        
        calculator = MetricsCalculator(['Negative', 'Positive'])
        per_class = calculator.calculate_per_class_metrics(y_true, y_pred)
        
        # Check structure
        assert 'Negative' in per_class
        assert 'Positive' in per_class
        
        # Check that all metrics are present
        for class_name in ['Negative', 'Positive']:
            assert 'precision' in per_class[class_name]
            assert 'recall' in per_class[class_name]
            assert 'f1_score' in per_class[class_name]
            assert 'support' in per_class[class_name]
            
            # Check value ranges
            assert 0 <= per_class[class_name]['precision'] <= 1
            assert 0 <= per_class[class_name]['recall'] <= 1
            assert 0 <= per_class[class_name]['f1_score'] <= 1
            assert per_class[class_name]['support'] >= 0
    
    def test_comprehensive_metrics(self, sample_data):
        """Test comprehensive metrics calculation."""
        y_true, y_pred, y_prob = sample_data['imperfect']
        
        calculator = MetricsCalculator()
        metrics = calculator.calculate_comprehensive_metrics(y_true, y_pred, y_prob)
        
        # Check that all sections are present
        assert 'basic_metrics' in metrics
        assert 'per_class_metrics' in metrics
        assert 'confusion_matrix' in metrics
        assert 'confusion_matrix_stats' in metrics
        assert 'classification_report' in metrics
        assert 'roc_curve' in metrics
        assert 'pr_curve' in metrics
        assert 'threshold' in metrics
        
        # Check basic structure
        assert isinstance(metrics['confusion_matrix'], list)
        assert isinstance(metrics['confusion_matrix_stats'], dict)
        assert isinstance(metrics['roc_curve'], dict)
        assert isinstance(metrics['pr_curve'], dict)
    
    def test_threshold_optimization(self, sample_data):
        """Test threshold optimization functionality."""
        y_true, _, y_prob = sample_data['imperfect']
        
        calculator = MetricsCalculator()
        
        # Test F1-score optimization
        best_threshold, best_f1 = calculator.find_optimal_threshold(y_true, y_prob, 'f1_score')
        
        assert 0 <= best_threshold <= 1
        assert 0 <= best_f1 <= 1
        
        # Test accuracy optimization
        best_threshold_acc, best_acc = calculator.find_optimal_threshold(y_true, y_prob, 'accuracy')
        
        assert 0 <= best_threshold_acc <= 1
        assert 0 <= best_acc <= 1
    
    def test_input_validation(self):
        """Test input validation and error handling."""
        calculator = MetricsCalculator()
        
        # Test mismatched lengths
        with pytest.raises(ValueError):
            calculator.calculate_basic_metrics([0, 1], [0])
        
        # Test empty inputs
        with pytest.raises(ValueError):
            calculator.calculate_basic_metrics([], [])
        
        # Test non-binary labels
        with pytest.raises(ValueError):
            calculator.calculate_basic_metrics([0, 1, 2], [0, 1, 1])
    
    def test_convenience_functions(self, sample_data):
        """Test convenience functions."""
        y_true, y_pred, y_prob = sample_data['imperfect']
        
        # Test calculate_metrics function
        metrics = calculate_metrics(y_true, y_pred, y_prob)
        assert isinstance(metrics, dict)
        assert 'basic_metrics' in metrics
        
        # Test find_optimal_threshold function
        threshold, score = find_optimal_threshold(y_true, y_prob)
        assert 0 <= threshold <= 1
        assert 0 <= score <= 1


class TestEvaluationVisualizer:
    """Tests for visualization functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_confusion_matrix(self):
        """Create sample confusion matrix for testing."""
        return np.array([[85, 15], [10, 90]])
    
    @pytest.fixture
    def sample_training_history(self):
        """Create sample training history for testing."""
        epochs = 10
        return {
            'train_losses': [1.0 - i*0.08 for i in range(epochs)],
            'val_losses': [0.9 - i*0.06 for i in range(epochs)],
            'val_accuracies': [60 + i*3 for i in range(epochs)],
            'learning_rates': [0.001 * (0.9**i) for i in range(epochs)]
        }
    
    def test_confusion_matrix_plot(self, sample_confusion_matrix, temp_dir):
        """Test confusion matrix plotting."""
        visualizer = EvaluationVisualizer()
        
        save_path = os.path.join(temp_dir, 'test_cm.png')
        
        fig = visualizer.plot_confusion_matrix(
            sample_confusion_matrix,
            class_names=['Negative', 'Positive'],
            title='Test Confusion Matrix',
            save_path=save_path,
            show_plot=False
        )
        
        # Check that figure was created
        assert isinstance(fig, plt.Figure)
        
        # Check that file was saved
        assert os.path.exists(save_path)
        
        plt.close(fig)
    
    def test_training_history_plot(self, sample_training_history, temp_dir):
        """Test training history plotting."""
        visualizer = EvaluationVisualizer()
        
        save_path = os.path.join(temp_dir, 'test_history.png')
        
        fig = visualizer.plot_training_history(
            sample_training_history,
            title='Test Training History',
            save_path=save_path,
            show_plot=False
        )
        
        # Check that figure was created
        assert isinstance(fig, plt.Figure)
        
        # Check that file was saved
        assert os.path.exists(save_path)
        
        plt.close(fig)
    
    def test_roc_curve_plot(self, temp_dir):
        """Test ROC curve plotting."""
        visualizer = EvaluationVisualizer()
        
        # Sample ROC data
        fpr = [0.0, 0.1, 0.2, 0.3, 1.0]
        tpr = [0.0, 0.6, 0.8, 0.9, 1.0]
        auc_score = 0.85
        
        save_path = os.path.join(temp_dir, 'test_roc.png')
        
        fig = visualizer.plot_roc_curve(
            fpr, tpr, auc_score,
            title='Test ROC Curve',
            save_path=save_path,
            show_plot=False
        )
        
        # Check that figure was created
        assert isinstance(fig, plt.Figure)
        
        # Check that file was saved
        assert os.path.exists(save_path)
        
        plt.close(fig)
    
    def test_precision_recall_curve_plot(self, temp_dir):
        """Test Precision-Recall curve plotting."""
        visualizer = EvaluationVisualizer()
        
        # Sample PR data
        precision = [1.0, 0.9, 0.8, 0.7, 0.5]
        recall = [0.0, 0.2, 0.5, 0.8, 1.0]
        ap_score = 0.82
        
        save_path = os.path.join(temp_dir, 'test_pr.png')
        
        fig = visualizer.plot_precision_recall_curve(
            precision, recall, ap_score,
            title='Test PR Curve',
            save_path=save_path,
            show_plot=False
        )
        
        # Check that figure was created
        assert isinstance(fig, plt.Figure)
        
        # Check that file was saved
        assert os.path.exists(save_path)
        
        plt.close(fig)
    
    def test_threshold_analysis_plot(self, temp_dir):
        """Test threshold analysis plotting."""
        visualizer = EvaluationVisualizer()
        
        # Sample threshold data
        threshold_metrics = {
            0.3: {'accuracy': 0.75, 'precision': 0.70, 'recall': 0.85, 'f1_score': 0.77},
            0.5: {'accuracy': 0.80, 'precision': 0.78, 'recall': 0.82, 'f1_score': 0.80},
            0.7: {'accuracy': 0.78, 'precision': 0.85, 'recall': 0.72, 'f1_score': 0.78}
        }
        
        save_path = os.path.join(temp_dir, 'test_threshold.png')
        
        fig = visualizer.plot_threshold_analysis(
            threshold_metrics,
            title='Test Threshold Analysis',
            save_path=save_path,
            show_plot=False
        )
        
        # Check that figure was created
        assert isinstance(fig, plt.Figure)
        
        # Check that file was saved
        assert os.path.exists(save_path)
        
        plt.close(fig)
    
    def test_evaluation_dashboard(self, sample_confusion_matrix, sample_training_history, temp_dir):
        """Test comprehensive evaluation dashboard creation."""
        visualizer = EvaluationVisualizer()
        
        # Sample metrics data
        metrics = {
            'confusion_matrix': sample_confusion_matrix.tolist(),
            'basic_metrics': {
                'accuracy': 0.85,
                'precision': 0.80,
                'recall': 0.82,
                'f1_score': 0.81,
                'roc_auc': 0.88,
                'pr_auc': 0.85
            },
            'roc_curve': {
                'fpr': [0.0, 0.1, 0.2, 1.0],
                'tpr': [0.0, 0.7, 0.9, 1.0]
            },
            'pr_curve': {
                'precision': [1.0, 0.9, 0.8, 0.5],
                'recall': [0.0, 0.3, 0.7, 1.0]
            }
        }
        
        figures = visualizer.create_evaluation_dashboard(
            metrics,
            history=sample_training_history,
            save_dir=temp_dir,
            show_plots=False
        )
        
        # Check that figures were created
        assert isinstance(figures, dict)
        assert len(figures) > 0
        
        # Check that files were saved
        expected_files = ['confusion_matrix.png', 'training_history.png', 'roc_curve.png', 'pr_curve.png']
        for filename in expected_files:
            filepath = os.path.join(temp_dir, filename)
            assert os.path.exists(filepath)
        
        # Close all figures
        for fig in figures.values():
            plt.close(fig)


class TestEvaluationReporter:
    """Tests for evaluation reporting functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_metrics(self):
        """Create sample metrics for testing."""
        return {
            'threshold': 0.5,
            'basic_metrics': {
                'accuracy': 0.85,
                'precision': 0.80,
                'recall': 0.82,
                'f1_score': 0.81,
                'specificity': 0.88,
                'roc_auc': 0.87,
                'pr_auc': 0.84,
                'support': 1000
            },
            'confusion_matrix_stats': {
                'true_positives': 410,
                'true_negatives': 440,
                'false_positives': 60,
                'false_negatives': 90,
                'total_samples': 1000
            },
            'per_class_metrics': {
                'Negative': {
                    'precision': 0.83,
                    'recall': 0.88,
                    'f1_score': 0.85,
                    'support': 500
                },
                'Positive': {
                    'precision': 0.87,
                    'recall': 0.82,
                    'f1_score': 0.84,
                    'support': 500
                }
            }
        }
    
    @pytest.fixture
    def sample_model_info(self):
        """Create sample model info for testing."""
        return {
            'vocab_size': 10000,
            'embedding_dim': 300,
            'hidden_dim': 128,
            'n_layers': 2,
            'dropout': 0.3,
            'bidirectional': True,
            'total_parameters': 1234567
        }
    
    @pytest.fixture
    def sample_training_summary(self):
        """Create sample training summary for testing."""
        return {
            'training_status': {
                'epochs_completed': 15,
                'best_epoch': 12
            },
            'performance_metrics': {
                'best_val_loss': 0.3456,
                'best_val_accuracy': 85.67
            }
        }
    
    def test_text_report_generation(self, sample_metrics, sample_model_info, sample_training_summary):
        """Test text report generation."""
        reporter = EvaluationReporter()
        
        report = reporter.generate_text_report(
            sample_metrics,
            sample_model_info,
            sample_training_summary
        )
        
        # Check that report is a string
        assert isinstance(report, str)
        assert len(report) > 0
        
        # Check that key information is included
        assert 'EVALUATION REPORT' in report
        assert 'MODEL CONFIGURATION' in report
        assert 'TRAINING SUMMARY' in report
        assert 'PERFORMANCE METRICS' in report
        assert 'CONFUSION MATRIX STATISTICS' in report
        assert 'PER-CLASS METRICS' in report
        
        # Check specific values
        assert '0.8500' in report  # accuracy
        assert '15' in report      # epochs completed
        assert '10000' in report   # vocab size
    
    def test_json_report_generation(self, sample_metrics, sample_model_info, sample_training_summary):
        """Test JSON report generation."""
        reporter = EvaluationReporter()
        
        report = reporter.generate_json_report(
            sample_metrics,
            sample_model_info,
            sample_training_summary
        )
        
        # Check that report is a dictionary
        assert isinstance(report, dict)
        
        # Check structure
        assert 'report_metadata' in report
        assert 'evaluation_metrics' in report
        assert 'model_info' in report
        assert 'training_summary' in report
        
        # Check metadata
        assert 'generated_at' in report['report_metadata']
        assert 'report_type' in report['report_metadata']
        assert 'version' in report['report_metadata']
        
        # Check that metrics are preserved
        assert report['evaluation_metrics'] == sample_metrics
        assert report['model_info'] == sample_model_info
        assert report['training_summary'] == sample_training_summary
    
    def test_report_saving_text(self, sample_metrics, temp_dir):
        """Test saving text report to file."""
        reporter = EvaluationReporter()
        
        output_path = os.path.join(temp_dir, 'test_report.txt')
        
        saved_path = reporter.save_report(
            sample_metrics,
            output_path,
            format_type='text'
        )
        
        # Check that file was created
        assert os.path.exists(saved_path)
        assert saved_path == output_path
        
        # Check file content
        with open(saved_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert 'EVALUATION REPORT' in content
        assert '0.8500' in content  # accuracy value
    
    def test_report_saving_json(self, sample_metrics, temp_dir):
        """Test saving JSON report to file."""
        reporter = EvaluationReporter()
        
        output_path = os.path.join(temp_dir, 'test_report.json')
        
        saved_path = reporter.save_report(
            sample_metrics,
            output_path,
            format_type='json'
        )
        
        # Check that file was created
        assert os.path.exists(saved_path)
        assert saved_path == output_path
        
        # Check file content
        with open(saved_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
        
        assert isinstance(content, dict)
        assert 'evaluation_metrics' in content
        assert content['evaluation_metrics'] == sample_metrics
    
    def test_invalid_format_type(self, sample_metrics, temp_dir):
        """Test error handling for invalid format types."""
        reporter = EvaluationReporter()
        
        output_path = os.path.join(temp_dir, 'test_report.xml')
        
        with pytest.raises(ValueError):
            reporter.save_report(
                sample_metrics,
                output_path,
                format_type='xml'  # Unsupported format
            )


class TestIntegrationScenarios:
    """Integration tests for complete evaluation workflows."""
    
    def test_complete_evaluation_workflow(self):
        """Test complete evaluation workflow from predictions to report."""
        # Generate synthetic evaluation data
        np.random.seed(42)
        n_samples = 1000
        
        # Create realistic predictions with some errors
        y_true = np.random.binomial(1, 0.5, n_samples)
        y_prob = np.where(y_true == 1, 
                         np.random.beta(3, 1, n_samples),  # Higher probs for positive class
                         np.random.beta(1, 3, n_samples))  # Lower probs for negative class
        
        # Add some noise to make it realistic
        noise = np.random.normal(0, 0.1, n_samples)
        y_prob = np.clip(y_prob + noise, 0, 1)
        
        y_pred = (y_prob >= 0.5).astype(int)
        
        # Calculate comprehensive metrics
        metrics = calculate_metrics(y_true, y_pred, y_prob, ['Negative', 'Positive'])
        
        # Verify metrics structure
        assert 'basic_metrics' in metrics
        assert 'confusion_matrix' in metrics
        assert 'per_class_metrics' in metrics
        
        # Verify reasonable metric values
        assert 0.7 <= metrics['basic_metrics']['accuracy'] <= 1.0
        assert 0.7 <= metrics['basic_metrics']['roc_auc'] <= 1.0
        
        # Test visualization (without showing plots)
        visualizer = EvaluationVisualizer()
        
        # Test confusion matrix plot
        cm = np.array(metrics['confusion_matrix'])
        fig_cm = visualizer.plot_confusion_matrix(cm, ['Negative', 'Positive'], show_plot=False)
        assert isinstance(fig_cm, plt.Figure)
        plt.close(fig_cm)
        
        # Test ROC curve plot
        if metrics['roc_curve']:
            fig_roc = visualizer.plot_roc_curve(
                metrics['roc_curve']['fpr'],
                metrics['roc_curve']['tpr'],
                metrics['basic_metrics']['roc_auc'],
                show_plot=False
            )
            assert isinstance(fig_roc, plt.Figure)
            plt.close(fig_roc)
        
        # Test report generation
        reporter = EvaluationReporter()
        text_report = reporter.generate_text_report(metrics)
        json_report = reporter.generate_json_report(metrics)
        
        assert isinstance(text_report, str)
        assert len(text_report) > 0
        assert isinstance(json_report, dict)
        assert 'evaluation_metrics' in json_report


if __name__ == '__main__':
    # Run tests if executed directly
    pytest.main([__file__, '-v'])