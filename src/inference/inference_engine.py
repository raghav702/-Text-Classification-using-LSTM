"""
Inference Engine for LSTM Sentiment Classifier

This module provides functionality for loading trained models and making
sentiment predictions on new text inputs.
"""

import os
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union, Optional
import logging

from models.lstm_model import LSTMClassifier
from data.text_preprocessor import TextPreprocessor


class InferenceEngine:
    """
    Inference engine for sentiment classification using trained LSTM models.
    
    Handles model loading, text preprocessing, and prediction generation
    with confidence scores and batch processing capabilities.
    """
    
    def __init__(self, model_path: str = None, vocab_path: str = None, device: str = None):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to trained model checkpoint
            vocab_path: Path to vocabulary file
            device: Device to run inference on ('cpu', 'cuda', or None for auto-detect)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.preprocessor = None
        self.model_config = None
        self.is_loaded = False
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Load model and vocabulary if paths provided
        if model_path and vocab_path:
            self.load_model(model_path, vocab_path)
    
    def load_model(self, model_path: str, vocab_path: str) -> bool:
        """
        Load trained model weights and vocabulary for inference.
        
        Args:
            model_path: Path to the trained model checkpoint file
            vocab_path: Path to the vocabulary file
            
        Returns:
            True if loading successful, False otherwise
            
        Raises:
            FileNotFoundError: If model or vocabulary files don't exist
            RuntimeError: If model loading fails
        """
        try:
            # Validate file paths
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            if not os.path.exists(vocab_path):
                raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
            
            self.logger.info(f"Loading model from {model_path}")
            self.logger.info(f"Loading vocabulary from {vocab_path}")
            
            # Load model checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract model configuration
            if 'model_config' in checkpoint:
                self.model_config = checkpoint['model_config']
            else:
                # Fallback to default config if not saved in checkpoint
                self.logger.warning("Model config not found in checkpoint, using default values")
                self.model_config = {
                    'vocab_size': 10000,
                    'embedding_dim': 300,
                    'hidden_dim': 128,
                    'output_dim': 1,
                    'n_layers': 2,
                    'dropout': 0.3,
                    'bidirectional': True
                }
            
            # Initialize model with loaded configuration
            # Filter out non-constructor parameters
            constructor_params = {
                'vocab_size', 'embedding_dim', 'hidden_dim', 'output_dim', 
                'n_layers', 'dropout', 'bidirectional', 'pad_idx'
            }
            filtered_config = {k: v for k, v in self.model_config.items() if k in constructor_params}
            self.model = LSTMClassifier(**filtered_config)
            
            # Load model weights
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Assume the checkpoint is just the state dict
                self.model.load_state_dict(checkpoint)
            
            # Move model to device and set to evaluation mode
            self.model.to(self.device)
            self.model.eval()
            
            # Load text preprocessor with vocabulary
            self.preprocessor = TextPreprocessor()
            self.preprocessor.load_vocabulary(vocab_path)
            
            self.is_loaded = True
            self.logger.info("Model and vocabulary loaded successfully")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            self.is_loaded = False
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    def _validate_loaded(self):
        """
        Validate that model and preprocessor are loaded.
        
        Raises:
            RuntimeError: If model or preprocessor not loaded
        """
        if not self.is_loaded or self.model is None or self.preprocessor is None:
            raise RuntimeError(
                "Model not loaded. Call load_model() first with valid model and vocabulary paths."
            )
    
    def _preprocess_text(self, text: str) -> torch.Tensor:
        """
        Preprocess single text input for inference.
        
        Args:
            text: Input text string
            
        Returns:
            Preprocessed tensor ready for model input
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        
        if not text.strip():
            raise ValueError("Input text cannot be empty")
        
        # Convert text to sequence
        sequence = self.preprocessor.text_to_sequence(text)
        
        # Apply padding (creates batch dimension)
        padded_sequence = self.preprocessor.pad_sequences([sequence])
        
        # Move to device
        return padded_sequence.to(self.device)
    
    def _preprocess_batch(self, texts: List[str]) -> torch.Tensor:
        """
        Preprocess batch of texts for inference.
        
        Args:
            texts: List of input text strings
            
        Returns:
            Preprocessed tensor batch ready for model input
        """
        if not isinstance(texts, list):
            raise ValueError("Input must be a list of strings")
        
        if not texts:
            raise ValueError("Input list cannot be empty")
        
        # Validate all inputs are strings
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValueError(f"All inputs must be strings. Item {i} is {type(text)}")
            if not text.strip():
                raise ValueError(f"Text at index {i} cannot be empty")
        
        # Convert texts to sequences
        sequences = [self.preprocessor.text_to_sequence(text) for text in texts]
        
        # Apply padding
        padded_sequences = self.preprocessor.pad_sequences(sequences)
        
        # Move to device
        return padded_sequences.to(self.device)
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model configuration and statistics
        """
        self._validate_loaded()
        
        info = {
            'model_config': self.model_config,
            'device': str(self.device),
            'vocab_info': self.preprocessor.get_vocab_info(),
            'model_parameters': self.model.get_model_info(),
            'is_loaded': self.is_loaded
        }
        
        return info
    
    def predict_sentiment(self, text: str, threshold: float = 0.5) -> Tuple[str, float]:
        """
        Predict sentiment for a single text input with confidence score.
        
        Args:
            text: Input text string to classify
            threshold: Decision threshold for binary classification (default: 0.5)
            
        Returns:
            Tuple of (sentiment_label, confidence_score)
            - sentiment_label: 'positive' or 'negative'
            - confidence_score: Float between 0 and 1
            
        Raises:
            RuntimeError: If model not loaded
            ValueError: If input validation fails
        """
        self._validate_loaded()
        
        try:
            # Preprocess input text
            input_tensor = self._preprocess_text(text)
            
            # Get model prediction
            with torch.no_grad():
                logits = self.model(input_tensor)
                probability = torch.sigmoid(logits).item()
            
            # Determine sentiment label based on threshold
            sentiment_label = 'positive' if probability >= threshold else 'negative'
            
            # Calculate confidence score (distance from 0.5)
            confidence_score = abs(probability - 0.5) * 2  # Scale to [0, 1]
            
            return sentiment_label, confidence_score
            
        except Exception as e:
            self.logger.error(f"Prediction failed for text: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def predict_sentiment_with_probability(self, text: str) -> Tuple[str, float, float]:
        """
        Predict sentiment with both raw probability and confidence score.
        
        Args:
            text: Input text string to classify
            
        Returns:
            Tuple of (sentiment_label, raw_probability, confidence_score)
            - sentiment_label: 'positive' or 'negative'
            - raw_probability: Raw sigmoid probability (0-1, where >0.5 is positive)
            - confidence_score: Confidence measure (0-1)
        """
        self._validate_loaded()
        
        try:
            # Preprocess input text
            input_tensor = self._preprocess_text(text)
            
            # Get model prediction
            with torch.no_grad():
                logits = self.model(input_tensor)
                probability = torch.sigmoid(logits).item()
            
            # Determine sentiment label
            sentiment_label = 'positive' if probability >= 0.5 else 'negative'
            
            # Calculate confidence score
            confidence_score = abs(probability - 0.5) * 2
            
            return sentiment_label, probability, confidence_score
            
        except Exception as e:
            self.logger.error(f"Prediction failed for text: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def batch_predict(self, texts: List[str], threshold: float = 0.5) -> List[Tuple[str, float]]:
        """
        Predict sentiment for multiple texts efficiently in batch.
        
        Args:
            texts: List of input text strings to classify
            threshold: Decision threshold for binary classification (default: 0.5)
            
        Returns:
            List of tuples (sentiment_label, confidence_score) for each input text
            
        Raises:
            RuntimeError: If model not loaded
            ValueError: If input validation fails
        """
        self._validate_loaded()
        
        if not texts:
            return []
        
        try:
            # Preprocess batch of texts
            input_tensor = self._preprocess_batch(texts)
            
            # Get model predictions
            with torch.no_grad():
                logits = self.model(input_tensor)
                probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()
            
            # Handle single input case (squeeze removes batch dimension)
            if len(texts) == 1:
                probabilities = [probabilities.item()]
            else:
                probabilities = probabilities.tolist()
            
            # Process predictions
            results = []
            for prob in probabilities:
                sentiment_label = 'positive' if prob >= threshold else 'negative'
                confidence_score = abs(prob - 0.5) * 2
                results.append((sentiment_label, confidence_score))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch prediction failed: {str(e)}")
            raise RuntimeError(f"Batch prediction failed: {str(e)}")
    
    def batch_predict_with_probabilities(self, texts: List[str]) -> List[Tuple[str, float, float]]:
        """
        Predict sentiment for multiple texts with raw probabilities and confidence scores.
        
        Args:
            texts: List of input text strings to classify
            
        Returns:
            List of tuples (sentiment_label, raw_probability, confidence_score) for each input
        """
        self._validate_loaded()
        
        if not texts:
            return []
        
        try:
            # Preprocess batch of texts
            input_tensor = self._preprocess_batch(texts)
            
            # Get model predictions
            with torch.no_grad():
                logits = self.model(input_tensor)
                probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()
            
            # Handle single input case
            if len(texts) == 1:
                probabilities = [probabilities.item()]
            else:
                probabilities = probabilities.tolist()
            
            # Process predictions
            results = []
            for prob in probabilities:
                sentiment_label = 'positive' if prob >= 0.5 else 'negative'
                confidence_score = abs(prob - 0.5) * 2
                results.append((sentiment_label, prob, confidence_score))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch prediction with probabilities failed: {str(e)}")
            raise RuntimeError(f"Batch prediction with probabilities failed: {str(e)}")
    
    def predict_with_threshold_analysis(self, text: str, thresholds: List[float] = None) -> Dict:
        """
        Analyze prediction across different confidence thresholds.
        
        Args:
            text: Input text string to classify
            thresholds: List of thresholds to test (default: [0.3, 0.4, 0.5, 0.6, 0.7])
            
        Returns:
            Dictionary with threshold analysis results
        """
        if thresholds is None:
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        
        self._validate_loaded()
        
        try:
            # Get raw prediction
            _, raw_probability, base_confidence = self.predict_sentiment_with_probability(text)
            
            # Analyze across thresholds
            threshold_results = {}
            for threshold in thresholds:
                sentiment = 'positive' if raw_probability >= threshold else 'negative'
                confidence = abs(raw_probability - threshold) / max(threshold, 1 - threshold)
                threshold_results[threshold] = {
                    'sentiment': sentiment,
                    'confidence': confidence
                }
            
            return {
                'text': text,
                'raw_probability': raw_probability,
                'base_confidence': base_confidence,
                'threshold_analysis': threshold_results,
                'recommended_threshold': 0.5  # Could be made adaptive based on model calibration
            }
            
        except Exception as e:
            self.logger.error(f"Threshold analysis failed: {str(e)}")
            raise RuntimeError(f"Threshold analysis failed: {str(e)}")
    
    def validate_input(self, text: Union[str, List[str]]) -> bool:
        """
        Validate input text(s) for prediction.
        
        Args:
            text: Single text string or list of text strings
            
        Returns:
            True if input is valid, False otherwise
        """
        try:
            if isinstance(text, str):
                return bool(text.strip())
            elif isinstance(text, list):
                return all(isinstance(t, str) and bool(t.strip()) for t in text)
            else:
                return False
        except Exception:
            return False
    
    def get_prediction_stats(self, texts: List[str]) -> Dict:
        """
        Get statistics about predictions for a batch of texts.
        
        Args:
            texts: List of input text strings
            
        Returns:
            Dictionary with prediction statistics
        """
        self._validate_loaded()
        
        if not texts:
            return {'error': 'No texts provided'}
        
        try:
            results = self.batch_predict_with_probabilities(texts)
            
            # Calculate statistics
            probabilities = [prob for _, prob, _ in results]
            confidences = [conf for _, _, conf in results]
            sentiments = [sent for sent, _, _ in results]
            
            positive_count = sum(1 for sent in sentiments if sent == 'positive')
            negative_count = len(sentiments) - positive_count
            
            stats = {
                'total_texts': len(texts),
                'positive_predictions': positive_count,
                'negative_predictions': negative_count,
                'positive_ratio': positive_count / len(texts),
                'average_probability': sum(probabilities) / len(probabilities),
                'average_confidence': sum(confidences) / len(confidences),
                'min_confidence': min(confidences),
                'max_confidence': max(confidences),
                'high_confidence_count': sum(1 for conf in confidences if conf > 0.7),
                'low_confidence_count': sum(1 for conf in confidences if conf < 0.3)
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Statistics calculation failed: {str(e)}")
            return {'error': f'Statistics calculation failed: {str(e)}'}


def create_inference_engine(model_path: str, vocab_path: str, device: str = None) -> InferenceEngine:
    """
    Factory function to create and initialize an inference engine.
    
    Args:
        model_path: Path to trained model checkpoint
        vocab_path: Path to vocabulary file
        device: Device to run inference on
        
    Returns:
        Initialized InferenceEngine instance
    """
    engine = InferenceEngine(device=device)
    engine.load_model(model_path, vocab_path)
    return engine