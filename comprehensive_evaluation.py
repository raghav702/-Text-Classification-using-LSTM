#!/usr/bin/env python3
"""
Comprehensive evaluation script for LSTM Sentiment Classifier.
This script provides detailed performance metrics, visualizations, and analysis.
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import logging
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from inference.inference_engine import create_inference_engine

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_test_data():
    """Load test data for evaluation."""
    test_path = "data/imdb/test.csv"
    
    if not os.path.exists(test_path):
        print("Test data not found. Please ensure IMDB dataset is downloaded.")
        return None
    
    test_df = pd.read_csv(test_path)
    print(f"Loaded test data: {len(test_df)} samples")
    
    return test_df

def evaluate_model_performance(engine, test_df, logger):
    """Comprehensive model evaluation."""
    logger.info("Starting comprehensive model evaluation...")
    
    # Determine column names
    text_col = 'review' if 'review' in test_df.columns else 'text'
    label_col = 'sentiment' if 'sentiment' in test_df.columns else 'label'
    
    texts = test_df[text_col].tolist()
    true_labels = test_df[label_col].tolist()
    
    # Convert string labels to integers if needed
    if isinstance(true_labels[0], str):
        true_labels = [1 if label == 'positive' else 0 for label in true_labels]
    
    logger.info(f"Evaluating on {len(texts)} samples...")
    
    # Get predictions
    predictions = []
    probabilities = []
    
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_results = engine.batch_predict_with_probabilities(batch_texts)
        
        for sentiment, prob, confidence in batch_results:
            predictions.append(1 if sentiment == 'positive' else 0)
            probabilities.append(prob)
        
        if (i // batch_size + 1) % 10 == 0:
            logger.info(f"Processed {i + len(batch_texts)}/{len(texts)} samples")
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    auc_roc = roc_auc_score(true_labels, probabilities)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc_roc,
        'total_samples': len(texts)
    }
    
    logger.info("Evaluation completed!")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-Score: {f1:.4f}")
    logger.info(f"AUC-ROC: {auc_roc:.4f}")
    
    return metrics, true_labels, predictions, probabilities

def create_visualizations(true_labels, predictions, probabilities, output_dir):
    """Create comprehensive visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(true_labels, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. ROC Curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(true_labels, probabilities)
    auc_score = roc_auc_score(true_labels, probabilities)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Probability Distribution
    plt.figure(figsize=(12, 5))
    
    # Separate probabilities by true label
    pos_probs = [prob for i, prob in enumerate(probabilities) if true_labels[i] == 1]
    neg_probs = [prob for i, prob in enumerate(probabilities) if true_labels[i] == 0]
    
    plt.subplot(1, 2, 1)
    plt.hist(pos_probs, bins=50, alpha=0.7, color='green', label='True Positive')
    plt.hist(neg_probs, bins=50, alpha=0.7, color='red', label='True Negative')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.title('Probability Distribution by True Label')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Prediction Confidence
    plt.subplot(1, 2, 2)
    confidences = [abs(prob - 0.5) * 2 for prob in probabilities]
    plt.hist(confidences, bins=50, alpha=0.7, color='blue')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Model Confidence Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'probability_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")

def analyze_errors(engine, true_labels, predictions, probabilities, texts, output_dir):
    """Analyze prediction errors in detail."""
    
    # Find misclassified samples
    errors = []
    for i, (true_label, pred_label, prob, text) in enumerate(zip(true_labels, predictions, probabilities, texts)):
        if true_label != pred_label:
            errors.append({
                'index': i,
                'text': text[:200] + '...' if len(text) > 200 else text,  # Truncate long texts
                'true_label': 'positive' if true_label == 1 else 'negative',
                'predicted_label': 'positive' if pred_label == 1 else 'negative',
                'probability': prob,
                'confidence': abs(prob - 0.5) * 2
            })
    
    # Sort by confidence (most confident errors first)
    errors.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Save error analysis
    error_df = pd.DataFrame(errors)
    error_path = os.path.join(output_dir, 'error_analysis.csv')
    error_df.to_csv(error_path, index=False)
    
    print(f"Error analysis saved to {error_path}")
    print(f"Total errors: {len(errors)} out of {len(true_labels)} samples ({len(errors)/len(true_labels)*100:.2f}%)")
    
    # Show top 10 most confident errors
    print("\nTop 10 Most Confident Errors:")
    print("-" * 80)
    for i, error in enumerate(errors[:10]):
        print(f"{i+1}. True: {error['true_label']}, Predicted: {error['predicted_label']}")
        print(f"   Confidence: {error['confidence']:.4f}, Probability: {error['probability']:.4f}")
        print(f"   Text: {error['text']}")
        print()

def generate_report(metrics, model_info, output_dir):
    """Generate comprehensive evaluation report."""
    
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("LSTM Sentiment Classifier - Comprehensive Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Model Information:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Device: {model_info['device']}\n")
        f.write(f"Vocabulary Size: {model_info['vocab_info']['vocab_size']}\n")
        f.write(f"Max Sequence Length: {model_info['vocab_info']['max_length']}\n")
        f.write(f"Total Parameters: {model_info['model_parameters']['total_parameters']:,}\n")
        f.write(f"Trainable Parameters: {model_info['model_parameters']['trainable_parameters']:,}\n\n")
        
        f.write("Performance Metrics:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total Test Samples: {metrics['total_samples']:,}\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1-Score: {metrics['f1_score']:.4f}\n")
        f.write(f"AUC-ROC: {metrics['auc_roc']:.4f}\n\n")
        
        f.write("Performance Interpretation:\n")
        f.write("-" * 25 + "\n")
        
        if metrics['accuracy'] >= 0.90:
            f.write("✓ Excellent performance (≥90% accuracy)\n")
        elif metrics['accuracy'] >= 0.85:
            f.write("✓ Very good performance (85-90% accuracy)\n")
        elif metrics['accuracy'] >= 0.80:
            f.write("✓ Good performance (80-85% accuracy)\n")
        elif metrics['accuracy'] >= 0.70:
            f.write("⚠ Fair performance (70-80% accuracy)\n")
        else:
            f.write("✗ Poor performance (<70% accuracy)\n")
        
        if metrics['auc_roc'] >= 0.90:
            f.write("✓ Excellent discrimination ability (AUC ≥ 0.90)\n")
        elif metrics['auc_roc'] >= 0.80:
            f.write("✓ Good discrimination ability (AUC 0.80-0.90)\n")
        elif metrics['auc_roc'] >= 0.70:
            f.write("⚠ Fair discrimination ability (AUC 0.70-0.80)\n")
        else:
            f.write("✗ Poor discrimination ability (AUC < 0.70)\n")
        
        f.write(f"\nBalanced Performance: {'✓' if abs(metrics['precision'] - metrics['recall']) < 0.05 else '⚠'}\n")
        f.write(f"Precision-Recall Difference: {abs(metrics['precision'] - metrics['recall']):.4f}\n")
    
    print(f"Evaluation report saved to {report_path}")

def main():
    """Main evaluation function."""
    logger = setup_logging()
    
    # Model paths - use the latest improved model
    model_files = [f for f in os.listdir('models') if f.startswith('improved_lstm_model') and f.endswith('.pth')]
    if not model_files:
        print("No improved model found. Please train the model first using retrain_improved_model.py")
        return
    
    # Get the latest model
    latest_model = sorted(model_files)[-1]
    model_path = os.path.join('models', latest_model)
    vocab_path = model_path.replace('.pth', '_vocabulary.pth')
    
    if not os.path.exists(vocab_path):
        print(f"Vocabulary file not found: {vocab_path}")
        return
    
    print(f"Using model: {model_path}")
    print(f"Using vocabulary: {vocab_path}")
    
    # Load model
    logger.info("Loading model and vocabulary...")
    engine = create_inference_engine(model_path, vocab_path, device='cpu')
    model_info = engine.get_model_info()
    
    # Load test data
    test_df = load_test_data()
    if test_df is None:
        return
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"evaluation_results/comprehensive_eval_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate model
    metrics, true_labels, predictions, probabilities = evaluate_model_performance(
        engine, test_df, logger
    )
    
    # Create visualizations
    logger.info("Creating visualizations...")
    create_visualizations(true_labels, predictions, probabilities, output_dir)
    
    # Analyze errors
    logger.info("Analyzing prediction errors...")
    texts = test_df['review' if 'review' in test_df.columns else 'text'].tolist()
    analyze_errors(engine, true_labels, predictions, probabilities, texts, output_dir)
    
    # Generate report
    logger.info("Generating evaluation report...")
    generate_report(metrics, model_info, output_dir)
    
    logger.info(f"Comprehensive evaluation completed! Results saved to {output_dir}")
    
    # Test with sample texts
    logger.info("Testing with sample texts...")
    sample_texts = [
        "This movie is absolutely amazing and fantastic!",
        "This movie is terrible and awful!",
        "Great film with excellent acting",
        "Boring movie with bad plot",
        "I love this movie so much",
        "I hate this movie completely"
    ]
    
    print("\nSample Predictions:")
    print("-" * 50)
    for text in sample_texts:
        sentiment, probability, confidence = engine.predict_sentiment_with_probability(text)
        print(f"'{text}'")
        print(f"  → {sentiment} (prob: {probability:.4f}, conf: {confidence:.4f})")
        print()

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("evaluation_results", exist_ok=True)
    
    main()