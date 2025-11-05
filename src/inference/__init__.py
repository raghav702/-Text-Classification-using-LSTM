"""
Inference module for LSTM Sentiment Classifier.

This module provides functionality for loading trained models and making
sentiment predictions on new text inputs.
"""

from .inference_engine import InferenceEngine, create_inference_engine

__all__ = ['InferenceEngine', 'create_inference_engine']