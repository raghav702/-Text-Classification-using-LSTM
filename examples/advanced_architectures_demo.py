"""
Advanced Model Architectures Demo
Demonstrates attention mechanisms and ensemble techniques for sentiment classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.models.lstm_model import LSTMClassifier
from src.models.attention_lstm import AttentionLSTMClassifier
from src.models.ensemble import (
    ModelEnsemble, MajorityVoting, WeightedVoting, 
    ConfidenceBasedVoting, create_diverse_ensemble
)
from src.training.ensemble_trainer import EnsembleTrainer
from src.evaluation.ensemble_evaluator import EnsembleEvaluator
from src.utils.attention_visualization import AttentionVisualizer
from src.data.dataset import IMDBDataset
from src.data.text_preprocessor import TextPreprocessor


def create_sample_data(vocab_size=1000, seq_length=100, n_samples=1000):
    """Create sample data for demonstration."""
    # Generate random sequences
    sequences = torch.randint(1, vocab_size, (n_samples, seq_length))
    
    # Generate random labels (0 or 1)
    labels = torch.randint(0, 2, (n_samples,))
    
    # Generate random lengths
    lengths = torch.randint(50, seq_length + 1, (n_samples,))
    
    return sequences, labels, lengths


def demonstrate_attention_mechanism():
    """Demonstrate attention-enhanced LSTM model."""
    print("=" * 60)
    print("ATTENTION MECHANISM DEMONSTRATION")
    print("=" * 60)
    
    # Model configuration
    config = {
        'vocab_size': 1000,
        'embedding_dim': 128,
        'hidden_dim': 64,
        'attention_dim': 32,
        'n_layers': 2,
        'dropout': 0.3,
        'use_attention_pooling': True
    }
    
    # Create models
    print("Creating models...")
    lstm_model = LSTMClassifier(
        vocab_size=config['vocab_size'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        n_layers=config['n_layers'],
        dropout=config['dropout']
    )
    
    attention_lstm_model = AttentionLSTMClassifier(**config)
    
    # Create sample data
    print("Generating sample data...")
    sequences, labels, lengths = create_sample_data(
        vocab_size=config['vocab_size'], 
        n_samples=100
    )
    
    # Compare model outputs
    print("\nComparing model architectures...")
    
    with torch.no_grad():
        # Standard LSTM
        lstm_output = lstm_model(sequences[:5])
        lstm_probs = torch.sigmoid(lstm_output)
        
        # Attention LSTM
        attention_output, attention_info = attention_lstm_model(
            sequences[:5], return_attention=True
        )
        attention_probs = torch.sigmoid(attention_output)
        
        print(f"Standard LSTM predictions: {lstm_probs.squeeze()}")
        print(f"Attention LSTM predictions: {attention_probs.squeeze()}")
        
        # Analyze attention weights
        if attention_info['pooling_attention_weights'] is not None:
            attention_weights = attention_info['pooling_attention_weights'][0]  # First sample
            print(f"\nAttention weights shape: {attention_weights.shape}")
            print(f"Top 5 attended positions: {torch.topk(attention_weights, 5).indices}")
    
    # Model comparison
    print(f"\nModel Parameter Comparison:")
    lstm_params = sum(p.numel() for p in lstm_model.parameters())
    attention_params = sum(p.numel() for p in attention_lstm_model.parameters())
    
    print(f"Standard LSTM parameters: {lstm_params:,}")
    print(f"Attention LSTM parameters: {attention_params:,}")
    print(f"Parameter increase: {((attention_params - lstm_params) / lstm_params * 100):.1f}%")


def demonstrate_ensemble_framework():
    """Demonstrate ensemble model framework."""
    print("\n" + "=" * 60)
    print("ENSEMBLE FRAMEWORK DEMONSTRATION")
    print("=" * 60)
    
    # Create diverse model configurations
    base_config = {
        'vocab_size': 1000,
        'embedding_dim': 128,
        'hidden_dim': 64,
        'n_layers': 2,
        'dropout': 0.3,
        'batch_size': 32,
        'learning_rate': 0.001
    }
    
    # Initialize ensemble trainer
    print("Creating diverse model configurations...")
    ensemble_trainer = EnsembleTrainer(base_config)
    diverse_configs = ensemble_trainer.create_diverse_configs(n_models=3)
    
    # Display configurations
    for i, config in enumerate(diverse_configs):
        print(f"Model {i+1}: {config['model_type']}, "
              f"Hidden: {config['hidden_dim']}, "
              f"Layers: {config['n_layers']}, "
              f"LR: {config['learning_rate']}")
    
    # Create models manually for demonstration
    print("\nCreating ensemble models...")
    models = []
    model_names = []
    
    # Model 1: Standard LSTM
    model1 = LSTMClassifier(
        vocab_size=1000, embedding_dim=128, hidden_dim=64, n_layers=2
    )
    models.append(model1)
    model_names.append("LSTM_64_2")
    
    # Model 2: Larger LSTM
    model2 = LSTMClassifier(
        vocab_size=1000, embedding_dim=128, hidden_dim=96, n_layers=2
    )
    models.append(model2)
    model_names.append("LSTM_96_2")
    
    # Model 3: Attention LSTM
    model3 = AttentionLSTMClassifier(
        vocab_size=1000, embedding_dim=128, hidden_dim=64, 
        attention_dim=32, n_layers=2, use_attention_pooling=True
    )
    models.append(model3)
    model_names.append("AttentionLSTM_64_2")
    
    # Create different ensemble strategies
    print("\nTesting different ensemble strategies...")
    
    # 1. Majority Voting
    majority_ensemble = ModelEnsemble(
        models=models,
        strategy=MajorityVoting(),
        model_names=model_names
    )
    
    # 2. Weighted Voting
    weights = [0.3, 0.3, 0.4]  # Give more weight to attention model
    weighted_ensemble = ModelEnsemble(
        models=models,
        strategy=WeightedVoting(weights),
        model_names=model_names
    )
    
    # 3. Confidence-based Voting
    confidence_ensemble = ModelEnsemble(
        models=models,
        strategy=ConfidenceBasedVoting(confidence_threshold=0.1),
        model_names=model_names
    )
    
    # Test ensembles with sample data
    print("\nTesting ensemble predictions...")
    sequences, labels, lengths = create_sample_data(n_samples=10)
    
    with torch.no_grad():
        # Get predictions from each ensemble
        maj_preds, maj_conf = majority_ensemble.predict(sequences)
        weight_preds, weight_conf = weighted_ensemble.predict(sequences)
        conf_preds, conf_conf = confidence_ensemble.predict(sequences)
        
        print(f"Majority voting predictions: {maj_preds.squeeze()[:5]}")
        print(f"Weighted voting predictions: {weight_preds.squeeze()[:5]}")
        print(f"Confidence voting predictions: {conf_preds.squeeze()[:5]}")
        
        # Test uncertainty estimation
        unc_preds, unc_conf, uncertainty = majority_ensemble.predict_with_uncertainty(sequences)
        print(f"Uncertainty scores (first 5): {uncertainty.squeeze()[:5]}")


def demonstrate_attention_visualization():
    """Demonstrate attention visualization capabilities."""
    print("\n" + "=" * 60)
    print("ATTENTION VISUALIZATION DEMONSTRATION")
    print("=" * 60)
    
    # Create attention model
    model = AttentionLSTMClassifier(
        vocab_size=1000,
        embedding_dim=128,
        hidden_dim=64,
        attention_dim=32,
        n_layers=2,
        use_attention_pooling=True
    )
    
    # Create sample data with mock tokens
    sample_tokens = ['this', 'movie', 'was', 'absolutely', 'fantastic', 'and', 'amazing']
    sequence_length = len(sample_tokens)
    
    # Generate mock attention weights
    mock_attention_weights = torch.softmax(torch.randn(sequence_length), dim=0)
    
    print(f"Sample tokens: {sample_tokens}")
    print(f"Mock attention weights: {mock_attention_weights}")
    
    # Initialize visualizer
    visualizer = AttentionVisualizer()
    
    # Analyze attention patterns
    analysis = visualizer.analyze_attention_patterns(
        mock_attention_weights,
        sample_tokens,
        sentiment_label="positive"
    )
    
    print(f"\nAttention Analysis:")
    print(f"Most attended token: {analysis['max_attention_token']} "
          f"(weight: {analysis['max_attention_weight']:.3f})")
    print(f"Attention entropy: {analysis['attention_entropy']:.3f}")
    print(f"Attention concentration: {analysis['attention_concentration']:.3f}")
    
    print(f"\nTop attended tokens:")
    for token_info in analysis['top_attended_tokens']:
        print(f"  {token_info['token']}: {token_info['weight']:.3f}")


def demonstrate_ensemble_evaluation():
    """Demonstrate ensemble evaluation and comparison."""
    print("\n" + "=" * 60)
    print("ENSEMBLE EVALUATION DEMONSTRATION")
    print("=" * 60)
    
    # Create mock evaluation results for demonstration
    mock_results = {
        'Majority_Ensemble': {
            'accuracy': 0.87,
            'precision': 0.85,
            'recall': 0.89,
            'f1_score': 0.87,
            'roc_auc': 0.92,
            'calibration_error': 0.05,
            'uncertainty_analysis': {
                'mean_uncertainty': 0.12,
                'std_uncertainty': 0.08,
                'uncertainty_correct_correlation': -0.15
            },
            'ensemble_metrics': {
                'ensemble_improvement': {
                    'improvement': 0.03,
                    'ensemble_accuracy': 0.87,
                    'best_individual_accuracy': 0.84,
                    'individual_accuracies': [0.82, 0.84, 0.83]
                },
                'model_diversity': {
                    'mean_pairwise_disagreement': 0.15
                }
            },
            'individual_model_analysis': {
                'LSTM_64': {'accuracy': 0.82, 'f1_score': 0.81},
                'LSTM_96': {'accuracy': 0.84, 'f1_score': 0.83},
                'AttentionLSTM': {'accuracy': 0.83, 'f1_score': 0.82}
            },
            'roc_curve': {
                'fpr': [0.0, 0.1, 0.2, 0.3, 1.0],
                'tpr': [0.0, 0.7, 0.85, 0.95, 1.0]
            }
        },
        'Weighted_Ensemble': {
            'accuracy': 0.89,
            'precision': 0.87,
            'recall': 0.91,
            'f1_score': 0.89,
            'roc_auc': 0.94,
            'calibration_error': 0.04,
            'uncertainty_analysis': {
                'mean_uncertainty': 0.10,
                'std_uncertainty': 0.07,
                'uncertainty_correct_correlation': -0.18
            },
            'ensemble_metrics': {
                'ensemble_improvement': {
                    'improvement': 0.05,
                    'ensemble_accuracy': 0.89,
                    'best_individual_accuracy': 0.84,
                    'individual_accuracies': [0.82, 0.84, 0.83]
                },
                'model_diversity': {
                    'mean_pairwise_disagreement': 0.15
                }
            },
            'individual_model_analysis': {
                'LSTM_64': {'accuracy': 0.82, 'f1_score': 0.81},
                'LSTM_96': {'accuracy': 0.84, 'f1_score': 0.83},
                'AttentionLSTM': {'accuracy': 0.83, 'f1_score': 0.82}
            },
            'roc_curve': {
                'fpr': [0.0, 0.08, 0.15, 0.25, 1.0],
                'tpr': [0.0, 0.75, 0.88, 0.96, 1.0]
            }
        }
    }
    
    # Initialize evaluator
    evaluator = EnsembleEvaluator()
    
    # Create comparison
    print("Creating ensemble comparison...")
    comparison_df = evaluator.compare_ensembles(mock_results)
    print("\nEnsemble Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Identify best ensemble
    best_ensemble = comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Ensemble']
    best_accuracy = comparison_df['Accuracy'].max()
    
    print(f"\nBest performing ensemble: {best_ensemble} (Accuracy: {best_accuracy:.3f})")
    
    # Show ensemble improvement analysis
    print(f"\nEnsemble Improvement Analysis:")
    for ensemble_name, results in mock_results.items():
        improvement = results['ensemble_metrics']['ensemble_improvement']['improvement']
        print(f"{ensemble_name}: +{improvement:.3f} over best individual model")


def main():
    """Main demonstration function."""
    print("ADVANCED MODEL ARCHITECTURES DEMONSTRATION")
    print("=" * 80)
    print("This demo showcases attention mechanisms and ensemble techniques")
    print("for sentiment classification using LSTM models.")
    print("=" * 80)
    
    try:
        # Demonstrate attention mechanism
        demonstrate_attention_mechanism()
        
        # Demonstrate ensemble framework
        demonstrate_ensemble_framework()
        
        # Demonstrate attention visualization
        demonstrate_attention_visualization()
        
        # Demonstrate ensemble evaluation
        demonstrate_ensemble_evaluation()
        
        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nKey Features Demonstrated:")
        print("✓ Self-attention mechanism for LSTM models")
        print("✓ Attention-based pooling for sequence representation")
        print("✓ Multiple ensemble voting strategies")
        print("✓ Model diversity and ensemble improvement analysis")
        print("✓ Attention weight visualization and interpretation")
        print("✓ Comprehensive ensemble evaluation metrics")
        
        print("\nNext Steps:")
        print("- Train models on real IMDB data using the ensemble trainer")
        print("- Experiment with different attention mechanisms")
        print("- Optimize ensemble weights based on validation performance")
        print("- Use attention visualization for model interpretability")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()