"""
Models module for LSTM sentiment classifier.
"""

from src.models.lstm_model import LSTMClassifier
from src.models.attention_lstm import AttentionLSTMClassifier, SelfAttention, AttentionPooling
from src.models.ensemble import (
    ModelEnsemble, EnsembleStrategy, MajorityVoting, 
    WeightedVoting, ConfidenceBasedVoting, create_diverse_ensemble
)
from src.models.model_factory import ModelFactory, create_sentiment_classifier, analyze_embedding_quality
from src.models.advanced_model_factory import (
    AdvancedModelFactory, create_quick_lstm, create_quick_attention_lstm, create_quick_ensemble
)

__all__ = [
    'LSTMClassifier',
    'AttentionLSTMClassifier',
    'SelfAttention',
    'AttentionPooling',
    'ModelEnsemble',
    'EnsembleStrategy',
    'MajorityVoting',
    'WeightedVoting', 
    'ConfidenceBasedVoting',
    'create_diverse_ensemble',
    'ModelFactory', 
    'create_sentiment_classifier',
    'analyze_embedding_quality',
    'AdvancedModelFactory',
    'create_quick_lstm',
    'create_quick_attention_lstm',
    'create_quick_ensemble'
]