#!/usr/bin/env python3
"""
Continuous Improvement Pipeline for LSTM Sentiment Classifier.
This script provides automated model retraining, monitoring, and improvement capabilities.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import shutil
from typing import Dict, List, Optional

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from inference.inference_engine import create_inference_engine

class ModelPerformanceMonitor:
    """Monitor model performance and detect degradation."""
    
    def __init__(self, performance_log_path: str = "logs/performance_history.json"):
        self.performance_log_path = performance_log_path
        self.logger = logging.getLogger(__name__)
        
        # Create logs directory
        os.makedirs(os.path.dirname(performance_log_path), exist_ok=True)
        
        # Load existing performance history
        self.performance_history = self._load_performance_history()
    
    def _load_performance_history(self) -> List[Dict]:
        """Load performance history from file."""
        if os.path.exists(self.performance_log_path):
            try:
                with open(self.performance_log_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load performance history: {e}")
        
        return []
    
    def _save_performance_history(self):
        """Save performance history to file."""
        try:
            with open(self.performance_log_path, 'w') as f:
                json.dump(self.performance_history, f, indent=2)
        except Exception as e:
            self.logger.error(f"Could not save performance history: {e}")
    
    def log_performance(self, model_path: str, metrics: Dict[str, float], 
                       dataset_info: Dict[str, any] = None):
        """Log model performance metrics."""
        
        performance_entry = {
            'timestamp': datetime.now().isoformat(),
            'model_path': model_path,
            'metrics': metrics,
            'dataset_info': dataset_info or {}
        }
        
        self.performance_history.append(performance_entry)
        self._save_performance_history()
        
        self.logger.info(f"Performance logged for model: {model_path}")
        self.logger.info(f"Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
    
    def detect_performance_degradation(self, current_metrics: Dict[str, float], 
                                     threshold: float = 0.05) -> bool:
        """
        Detect if current performance has degraded compared to recent history.
        
        Args:
            current_metrics: Current model performance metrics
            threshold: Degradation threshold (default: 5% drop in accuracy)
            
        Returns:
            True if degradation detected, False otherwise
        """
        if len(self.performance_history) < 2:
            return False
        
        # Get recent performance (last 5 entries)
        recent_entries = self.performance_history[-5:]
        recent_accuracies = [entry['metrics'].get('accuracy', 0) for entry in recent_entries]
        
        if not recent_accuracies:
            return False
        
        avg_recent_accuracy = np.mean(recent_accuracies)
        current_accuracy = current_metrics.get('accuracy', 0)
        
        degradation = avg_recent_accuracy - current_accuracy
        
        if degradation > threshold:
            self.logger.warning(f"Performance degradation detected!")
            self.logger.warning(f"Recent average accuracy: {avg_recent_accuracy:.4f}")
            self.logger.warning(f"Current accuracy: {current_accuracy:.4f}")
            self.logger.warning(f"Degradation: {degradation:.4f} (threshold: {threshold:.4f})")
            return True
        
        return False
    
    def get_performance_trend(self, metric: str = 'accuracy', window: int = 10) -> Dict[str, float]:
        """
        Analyze performance trend for a specific metric.
        
        Args:
            metric: Metric to analyze
            window: Number of recent entries to consider
            
        Returns:
            Dictionary with trend analysis
        """
        if len(self.performance_history) < 2:
            return {'trend': 'insufficient_data'}
        
        recent_entries = self.performance_history[-window:]
        values = [entry['metrics'].get(metric, 0) for entry in recent_entries]
        
        if len(values) < 2:
            return {'trend': 'insufficient_data'}
        
        # Calculate trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        trend_analysis = {
            'trend': 'improving' if slope > 0.001 else 'declining' if slope < -0.001 else 'stable',
            'slope': slope,
            'current_value': values[-1],
            'average_value': np.mean(values),
            'min_value': np.min(values),
            'max_value': np.max(values),
            'std_value': np.std(values)
        }
        
        return trend_analysis


class DataDriftDetector:
    """Detect data drift in incoming predictions."""
    
    def __init__(self, baseline_stats_path: str = "logs/baseline_stats.json"):
        self.baseline_stats_path = baseline_stats_path
        self.logger = logging.getLogger(__name__)
        
        # Create logs directory
        os.makedirs(os.path.dirname(baseline_stats_path), exist_ok=True)
        
        self.baseline_stats = self._load_baseline_stats()
    
    def _load_baseline_stats(self) -> Dict:
        """Load baseline statistics."""
        if os.path.exists(self.baseline_stats_path):
            try:
                with open(self.baseline_stats_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load baseline stats: {e}")
        
        return {}
    
    def _save_baseline_stats(self):
        """Save baseline statistics."""
        try:
            with open(self.baseline_stats_path, 'w') as f:
                json.dump(self.baseline_stats, f, indent=2)
        except Exception as e:
            self.logger.error(f"Could not save baseline stats: {e}")
    
    def establish_baseline(self, texts: List[str], engine):
        """Establish baseline statistics from a reference dataset."""
        
        self.logger.info("Establishing baseline statistics...")
        
        # Get predictions for baseline
        results = engine.batch_predict_with_probabilities(texts)
        probabilities = [prob for _, prob, _ in results]
        confidences = [conf for _, _, conf in results]
        
        # Calculate statistics
        self.baseline_stats = {
            'timestamp': datetime.now().isoformat(),
            'sample_size': len(texts),
            'probability_stats': {
                'mean': float(np.mean(probabilities)),
                'std': float(np.std(probabilities)),
                'min': float(np.min(probabilities)),
                'max': float(np.max(probabilities)),
                'percentiles': {
                    '25': float(np.percentile(probabilities, 25)),
                    '50': float(np.percentile(probabilities, 50)),
                    '75': float(np.percentile(probabilities, 75))
                }
            },
            'confidence_stats': {
                'mean': float(np.mean(confidences)),
                'std': float(np.std(confidences)),
                'min': float(np.min(confidences)),
                'max': float(np.max(confidences))
            }
        }
        
        self._save_baseline_stats()
        self.logger.info("Baseline statistics established")
    
    def detect_drift(self, texts: List[str], engine, threshold: float = 0.1) -> Dict[str, any]:
        """
        Detect data drift by comparing current data to baseline.
        
        Args:
            texts: Current text samples
            engine: Inference engine
            threshold: Drift detection threshold
            
        Returns:
            Dictionary with drift analysis results
        """
        if not self.baseline_stats:
            return {'drift_detected': False, 'reason': 'no_baseline'}
        
        # Get current predictions
        results = engine.batch_predict_with_probabilities(texts)
        probabilities = [prob for _, prob, _ in results]
        confidences = [conf for _, _, conf in results]
        
        # Calculate current statistics
        current_stats = {
            'probability_mean': float(np.mean(probabilities)),
            'probability_std': float(np.std(probabilities)),
            'confidence_mean': float(np.mean(confidences)),
            'confidence_std': float(np.std(confidences))
        }
        
        # Compare with baseline
        baseline_prob_mean = self.baseline_stats['probability_stats']['mean']
        baseline_prob_std = self.baseline_stats['probability_stats']['std']
        baseline_conf_mean = self.baseline_stats['confidence_stats']['mean']
        
        # Calculate drift metrics
        prob_mean_drift = abs(current_stats['probability_mean'] - baseline_prob_mean)
        prob_std_drift = abs(current_stats['probability_std'] - baseline_prob_std)
        conf_mean_drift = abs(current_stats['confidence_mean'] - baseline_conf_mean)
        
        # Detect drift
        drift_detected = (
            prob_mean_drift > threshold or 
            prob_std_drift > threshold or 
            conf_mean_drift > threshold
        )
        
        drift_analysis = {
            'drift_detected': drift_detected,
            'timestamp': datetime.now().isoformat(),
            'sample_size': len(texts),
            'drift_metrics': {
                'probability_mean_drift': prob_mean_drift,
                'probability_std_drift': prob_std_drift,
                'confidence_mean_drift': conf_mean_drift
            },
            'current_stats': current_stats,
            'baseline_stats': self.baseline_stats,
            'threshold': threshold
        }
        
        if drift_detected:
            self.logger.warning("Data drift detected!")
            self.logger.warning(f"Probability mean drift: {prob_mean_drift:.4f}")
            self.logger.warning(f"Confidence mean drift: {conf_mean_drift:.4f}")
        
        return drift_analysis


class ContinuousImprovementPipeline:
    """Main pipeline for continuous model improvement."""
    
    def __init__(self, config_path: str = "configs/continuous_improvement.json"):
        self.config_path = config_path
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize components
        self.performance_monitor = ModelPerformanceMonitor()
        self.drift_detector = DataDriftDetector()
    
    def _load_config(self) -> Dict:
        """Load configuration for continuous improvement."""
        default_config = {
            'performance_degradation_threshold': 0.05,
            'drift_detection_threshold': 0.1,
            'min_samples_for_retraining': 1000,
            'retraining_schedule_days': 30,
            'model_backup_count': 5,
            'alert_email': None,
            'monitoring_enabled': True
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                self.logger.warning(f"Could not load config: {e}")
        
        return default_config
    
    def save_config(self):
        """Save current configuration."""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            self.logger.error(f"Could not save config: {e}")
    
    def backup_model(self, model_path: str, vocab_path: str):
        """Create backup of current model."""
        backup_dir = "models/backups"
        os.makedirs(backup_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Backup model
        backup_model_path = os.path.join(backup_dir, f"backup_model_{timestamp}.pth")
        backup_vocab_path = os.path.join(backup_dir, f"backup_vocab_{timestamp}.pth")
        
        shutil.copy2(model_path, backup_model_path)
        shutil.copy2(vocab_path, backup_vocab_path)
        
        self.logger.info(f"Model backed up to {backup_model_path}")
        
        # Clean up old backups
        self._cleanup_old_backups(backup_dir)
    
    def _cleanup_old_backups(self, backup_dir: str):
        """Remove old backup files beyond the configured limit."""
        backup_files = [f for f in os.listdir(backup_dir) if f.startswith('backup_model_')]
        backup_files.sort(reverse=True)  # Most recent first
        
        max_backups = self.config.get('model_backup_count', 5)
        
        for old_backup in backup_files[max_backups:]:
            old_path = os.path.join(backup_dir, old_backup)
            os.remove(old_path)
            
            # Also remove corresponding vocab file
            vocab_file = old_backup.replace('backup_model_', 'backup_vocab_')
            vocab_path = os.path.join(backup_dir, vocab_file)
            if os.path.exists(vocab_path):
                os.remove(vocab_path)
        
        if len(backup_files) > max_backups:
            self.logger.info(f"Cleaned up {len(backup_files) - max_backups} old backups")
    
    def should_retrain(self, current_metrics: Dict[str, float], 
                      drift_analysis: Dict[str, any] = None) -> Dict[str, any]:
        """
        Determine if model should be retrained based on performance and drift.
        
        Returns:
            Dictionary with retraining recommendation and reasons
        """
        reasons = []
        should_retrain = False
        
        # Check performance degradation
        if self.performance_monitor.detect_performance_degradation(
            current_metrics, 
            self.config['performance_degradation_threshold']
        ):
            reasons.append('performance_degradation')
            should_retrain = True
        
        # Check data drift
        if drift_analysis and drift_analysis.get('drift_detected', False):
            reasons.append('data_drift')
            should_retrain = True
        
        # Check scheduled retraining
        if self._is_scheduled_retraining_due():
            reasons.append('scheduled_retraining')
            should_retrain = True
        
        return {
            'should_retrain': should_retrain,
            'reasons': reasons,
            'timestamp': datetime.now().isoformat()
        }
    
    def _is_scheduled_retraining_due(self) -> bool:
        """Check if scheduled retraining is due."""
        if not self.performance_monitor.performance_history:
            return False
        
        last_entry = self.performance_monitor.performance_history[-1]
        last_timestamp = datetime.fromisoformat(last_entry['timestamp'])
        
        days_since_last = (datetime.now() - last_timestamp).days
        schedule_days = self.config.get('retraining_schedule_days', 30)
        
        return days_since_last >= schedule_days
    
    def generate_improvement_report(self, model_path: str, vocab_path: str, 
                                  test_data_path: str = None) -> Dict[str, any]:
        """Generate comprehensive improvement report."""
        
        self.logger.info("Generating continuous improvement report...")
        
        # Load model
        engine = create_inference_engine(model_path, vocab_path, device='cpu')
        
        # Load test data if available
        current_metrics = {}
        drift_analysis = {}
        
        if test_data_path and os.path.exists(test_data_path):
            test_df = pd.read_csv(test_data_path)
            
            # Evaluate current performance
            # (This would integrate with the comprehensive evaluation script)
            current_metrics = {'accuracy': 0.85}  # Placeholder
            
            # Check for drift
            texts = test_df['review' if 'review' in test_df.columns else 'text'].tolist()[:1000]
            drift_analysis = self.drift_detector.detect_drift(texts, engine)
        
        # Get retraining recommendation
        retraining_rec = self.should_retrain(current_metrics, drift_analysis)
        
        # Get performance trend
        trend_analysis = self.performance_monitor.get_performance_trend()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'model_path': model_path,
                'vocab_path': vocab_path
            },
            'current_metrics': current_metrics,
            'performance_trend': trend_analysis,
            'drift_analysis': drift_analysis,
            'retraining_recommendation': retraining_rec,
            'config': self.config
        }
        
        # Save report
        report_dir = "logs/improvement_reports"
        os.makedirs(report_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(report_dir, f"improvement_report_{timestamp}.json")
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Improvement report saved to {report_path}")
        
        return report


def main():
    """Main function for continuous improvement pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting continuous improvement pipeline...")
    
    # Initialize pipeline
    pipeline = ContinuousImprovementPipeline()
    
    # Find latest model
    model_files = [f for f in os.listdir('models') if f.startswith('improved_lstm_model') and f.endswith('.pth')]
    if not model_files:
        logger.error("No improved model found. Please train a model first.")
        return
    
    latest_model = sorted(model_files)[-1]
    model_path = os.path.join('models', latest_model)
    vocab_path = model_path.replace('.pth', '_vocabulary.pth')
    
    # Generate improvement report
    test_data_path = "data/imdb/test.csv"
    report = pipeline.generate_improvement_report(
        model_path, vocab_path, test_data_path
    )
    
    # Print summary
    print("\n" + "="*60)
    print("CONTINUOUS IMPROVEMENT REPORT SUMMARY")
    print("="*60)
    
    print(f"Model: {latest_model}")
    print(f"Performance Trend: {report['performance_trend'].get('trend', 'unknown')}")
    print(f"Drift Detected: {report['drift_analysis'].get('drift_detected', 'unknown')}")
    print(f"Retraining Recommended: {report['retraining_recommendation']['should_retrain']}")
    
    if report['retraining_recommendation']['should_retrain']:
        print(f"Reasons: {', '.join(report['retraining_recommendation']['reasons'])}")
    
    print("\nFor detailed analysis, check the generated report in logs/improvement_reports/")


if __name__ == "__main__":
    main()