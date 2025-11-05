"""
Model Ensemble Framework
Implements ensemble techniques for combining multiple trained models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
from abc import ABC, abstractmethod
import pickle
import os
from pathlib import Path


class EnsembleStrategy(ABC):
    """
    Abstract base class for ensemble voting strategies.
    """
    
    @abstractmethod
    def combine_predictions(
        self, 
        predictions: List[torch.Tensor], 
        confidences: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Combine predictions from multiple models.
        
        Args:
            predictions: List of prediction tensors from different models
            confidences: Optional list of confidence scores
            
        Returns:
            Tuple of (combined_predictions, combined_confidences)
        """
        pass


class MajorityVoting(EnsembleStrategy):
    """
    Simple majority voting strategy for ensemble predictions.
    """
    
    def combine_predictions(
        self, 
        predictions: List[torch.Tensor], 
        confidences: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Combine predictions using majority voting.
        
        Args:
            predictions: List of binary prediction tensors
            confidences: Optional list of confidence scores (not used in majority voting)
            
        Returns:
            Tuple of (majority_predictions, vote_confidence)
        """
        # Stack predictions and compute majority vote
        stacked_predictions = torch.stack(predictions, dim=0)  # (n_models, batch_size, ...)
        
        # Compute majority vote
        vote_counts = stacked_predictions.sum(dim=0)  # Sum across models
        n_models = len(predictions)
        majority_predictions = (vote_counts > n_models / 2).float()
        
        # Compute confidence as proportion of models agreeing with majority
        vote_confidence = torch.maximum(vote_counts, n_models - vote_counts) / n_models
        
        return majority_predictions, vote_confidence


class WeightedVoting(EnsembleStrategy):
    """
    Weighted voting strategy based on model performance weights.
    """
    
    def __init__(self, weights: List[float]):
        """
        Initialize weighted voting with model weights.
        
        Args:
            weights: List of weights for each model (should sum to 1.0)
        """
        self.weights = torch.tensor(weights, dtype=torch.float32)
        if not torch.allclose(self.weights.sum(), torch.tensor(1.0), atol=1e-6):
            print(f"Warning: Weights sum to {self.weights.sum():.4f}, normalizing to 1.0")
            self.weights = self.weights / self.weights.sum()
    
    def combine_predictions(
        self, 
        predictions: List[torch.Tensor], 
        confidences: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Combine predictions using weighted voting.
        
        Args:
            predictions: List of prediction tensors
            confidences: Optional list of confidence scores
            
        Returns:
            Tuple of (weighted_predictions, weighted_confidences)
        """
        # Use confidences if available, otherwise use predictions
        if confidences is not None:
            vote_values = confidences
        else:
            vote_values = predictions
        
        # Stack and apply weights
        stacked_values = torch.stack(vote_values, dim=0)  # (n_models, batch_size, ...)
        weights = self.weights.view(-1, 1, 1)  # Reshape for broadcasting
        
        # Compute weighted average
        weighted_values = (stacked_values * weights).sum(dim=0)
        
        # Convert to binary predictions
        weighted_predictions = (weighted_values > 0.5).float()
        
        return weighted_predictions, weighted_values


class ConfidenceBasedVoting(EnsembleStrategy):
    """
    Confidence-based voting strategy that weights models by their prediction confidence.
    """
    
    def __init__(self, confidence_threshold: float = 0.1):
        """
        Initialize confidence-based voting.
        
        Args:
            confidence_threshold: Minimum confidence threshold for including a model's vote
        """
        self.confidence_threshold = confidence_threshold
    
    def combine_predictions(
        self, 
        predictions: List[torch.Tensor], 
        confidences: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Combine predictions using confidence-based weighting.
        
        Args:
            predictions: List of prediction tensors
            confidences: List of confidence scores (required for this strategy)
            
        Returns:
            Tuple of (confidence_weighted_predictions, combined_confidences)
        """
        if confidences is None:
            raise ValueError("Confidence scores are required for confidence-based voting")
        
        # Stack predictions and confidences
        stacked_predictions = torch.stack(predictions, dim=0)
        stacked_confidences = torch.stack(confidences, dim=0)
        
        # Apply confidence threshold mask
        confidence_mask = stacked_confidences > self.confidence_threshold
        
        # Compute confidence-weighted votes
        weighted_votes = stacked_predictions * stacked_confidences * confidence_mask.float()
        confidence_weights = stacked_confidences * confidence_mask.float()
        
        # Normalize by total confidence weights
        total_weights = confidence_weights.sum(dim=0)
        total_weights = torch.clamp(total_weights, min=1e-8)  # Avoid division by zero
        
        weighted_predictions_continuous = weighted_votes.sum(dim=0) / total_weights
        weighted_predictions = (weighted_predictions_continuous > 0.5).float()
        
        # Combined confidence is the average of contributing model confidences
        combined_confidences = (stacked_confidences * confidence_mask.float()).sum(dim=0) / torch.clamp(confidence_mask.sum(dim=0).float(), min=1)
        
        return weighted_predictions, combined_confidences


class ModelEnsemble:
    """
    Main ensemble class that combines multiple trained models using different voting strategies.
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        strategy: EnsembleStrategy,
        model_names: Optional[List[str]] = None,
        device: torch.device = None
    ):
        """
        Initialize model ensemble.
        
        Args:
            models: List of trained PyTorch models
            strategy: Ensemble voting strategy
            model_names: Optional names for the models
            device: Device to run inference on
        """
        self.models = models
        self.strategy = strategy
        self.model_names = model_names or [f"Model_{i}" for i in range(len(models))]
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move models to device and set to eval mode
        for model in self.models:
            model.to(self.device)
            model.eval()
        
        print(f"Initialized ensemble with {len(self.models)} models using {strategy.__class__.__name__}")
    
    def predict(
        self, 
        x: torch.Tensor, 
        lengths: Optional[torch.Tensor] = None,
        return_individual: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict]]:
        """
        Make ensemble predictions.
        
        Args:
            x: Input tensor
            lengths: Optional sequence lengths
            return_individual: Whether to return individual model predictions
            
        Returns:
            Tuple of (predictions, confidences) or (predictions, confidences, individual_results)
        """
        individual_predictions = []
        individual_confidences = []
        
        with torch.no_grad():
            for model in self.models:
                # Get model predictions
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(x, lengths)
                else:
                    logits = model(x, lengths)
                    probs = torch.sigmoid(logits)
                
                # Convert to binary predictions
                preds = (probs > 0.5).float()
                
                individual_predictions.append(preds)
                individual_confidences.append(probs)
        
        # Combine predictions using the ensemble strategy
        ensemble_predictions, ensemble_confidences = self.strategy.combine_predictions(
            individual_predictions, individual_confidences
        )
        
        if return_individual:
            individual_results = {
                'predictions': individual_predictions,
                'confidences': individual_confidences,
                'model_names': self.model_names
            }
            return ensemble_predictions, ensemble_confidences, individual_results
        
        return ensemble_predictions, ensemble_confidences
    
    def predict_with_uncertainty(
        self, 
        x: torch.Tensor, 
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Make predictions with uncertainty estimation based on model disagreement.
        
        Args:
            x: Input tensor
            lengths: Optional sequence lengths
            
        Returns:
            Tuple of (predictions, confidences, uncertainty_scores)
        """
        predictions, confidences, individual_results = self.predict(
            x, lengths, return_individual=True
        )
        
        # Calculate uncertainty as variance in individual model predictions
        individual_confidences = torch.stack(individual_results['confidences'], dim=0)
        uncertainty_scores = individual_confidences.var(dim=0)
        
        return predictions, confidences, uncertainty_scores
    
    def evaluate_ensemble(
        self, 
        dataloader, 
        criterion: nn.Module = None
    ) -> Dict[str, float]:
        """
        Evaluate ensemble performance on a dataset.
        
        Args:
            dataloader: DataLoader for evaluation data
            criterion: Loss function (optional)
            
        Returns:
            Dictionary with evaluation metrics
        """
        if criterion is None:
            criterion = nn.BCEWithLogitsLoss()
        
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, (data, targets, lengths) in enumerate(dataloader):
                data, targets = data.to(self.device), targets.to(self.device)
                if lengths is not None:
                    lengths = lengths.to(self.device)
                
                # Get ensemble predictions
                predictions, confidences = self.predict(data, lengths)
                
                # Calculate loss using confidences (probabilities)
                loss = criterion(torch.logit(confidences.clamp(1e-7, 1-1e-7)), targets.float())
                total_loss += loss.item()
                
                # Calculate accuracy
                correct_predictions += (predictions.squeeze() == targets.float()).sum().item()
                total_samples += targets.size(0)
                
                # Store for additional metrics
                all_predictions.extend(predictions.squeeze().cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate metrics
        accuracy = correct_predictions / total_samples
        avg_loss = total_loss / len(dataloader)
        
        # Calculate additional metrics
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # Precision, Recall, F1
        tp = np.sum((all_predictions == 1) & (all_targets == 1))
        fp = np.sum((all_predictions == 1) & (all_targets == 0))
        fn = np.sum((all_predictions == 0) & (all_targets == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def compare_individual_models(
        self, 
        dataloader,
        criterion: nn.Module = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare performance of individual models in the ensemble.
        
        Args:
            dataloader: DataLoader for evaluation data
            criterion: Loss function (optional)
            
        Returns:
            Dictionary with performance metrics for each model
        """
        if criterion is None:
            criterion = nn.BCEWithLogitsLoss()
        
        model_performances = {}
        
        for i, (model, model_name) in enumerate(zip(self.models, self.model_names)):
            total_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            
            with torch.no_grad():
                for batch_idx, (data, targets, lengths) in enumerate(dataloader):
                    data, targets = data.to(self.device), targets.to(self.device)
                    if lengths is not None:
                        lengths = lengths.to(self.device)
                    
                    # Get individual model predictions
                    if hasattr(model, 'predict_proba'):
                        probs = model.predict_proba(data, lengths)
                    else:
                        logits = model(data, lengths)
                        probs = torch.sigmoid(logits)
                    
                    predictions = (probs > 0.5).float()
                    
                    # Calculate loss
                    loss = criterion(torch.logit(probs.clamp(1e-7, 1-1e-7)), targets.float())
                    total_loss += loss.item()
                    
                    # Calculate accuracy
                    correct_predictions += (predictions.squeeze() == targets.float()).sum().item()
                    total_samples += targets.size(0)
            
            model_performances[model_name] = {
                'accuracy': correct_predictions / total_samples,
                'loss': total_loss / len(dataloader)
            }
        
        return model_performances
    
    def save_ensemble(self, save_path: str):
        """
        Save the ensemble configuration and model states.
        
        Args:
            save_path: Directory path to save ensemble
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save individual models
        model_paths = []
        for i, (model, model_name) in enumerate(zip(self.models, self.model_names)):
            model_path = save_path / f"{model_name}_model.pth"
            torch.save(model.state_dict(), model_path)
            model_paths.append(str(model_path))
        
        # Save ensemble configuration
        ensemble_config = {
            'model_paths': model_paths,
            'model_names': self.model_names,
            'strategy_class': self.strategy.__class__.__name__,
            'strategy_config': getattr(self.strategy, '__dict__', {})
        }
        
        config_path = save_path / 'ensemble_config.pkl'
        with open(config_path, 'wb') as f:
            pickle.dump(ensemble_config, f)
        
        print(f"Ensemble saved to: {save_path}")
    
    @classmethod
    def load_ensemble(
        cls,
        save_path: str,
        model_classes: List[type],
        model_configs: List[Dict],
        device: torch.device = None
    ):
        """
        Load a saved ensemble.
        
        Args:
            save_path: Directory path where ensemble was saved
            model_classes: List of model class types
            model_configs: List of model configuration dictionaries
            device: Device to load models on
            
        Returns:
            Loaded ModelEnsemble instance
        """
        save_path = Path(save_path)
        
        # Load ensemble configuration
        config_path = save_path / 'ensemble_config.pkl'
        with open(config_path, 'rb') as f:
            ensemble_config = pickle.load(f)
        
        # Recreate models
        models = []
        for i, (model_class, model_config, model_path) in enumerate(
            zip(model_classes, model_configs, ensemble_config['model_paths'])
        ):
            model = model_class(**model_config)
            model.load_state_dict(torch.load(model_path, map_location=device))
            models.append(model)
        
        # Recreate strategy
        strategy_class_name = ensemble_config['strategy_class']
        strategy_config = ensemble_config['strategy_config']
        
        if strategy_class_name == 'MajorityVoting':
            strategy = MajorityVoting()
        elif strategy_class_name == 'WeightedVoting':
            strategy = WeightedVoting(strategy_config['weights'])
        elif strategy_class_name == 'ConfidenceBasedVoting':
            strategy = ConfidenceBasedVoting(strategy_config.get('confidence_threshold', 0.1))
        else:
            raise ValueError(f"Unknown strategy class: {strategy_class_name}")
        
        # Create ensemble
        ensemble = cls(
            models=models,
            strategy=strategy,
            model_names=ensemble_config['model_names'],
            device=device
        )
        
        print(f"Ensemble loaded from: {save_path}")
        return ensemble


def create_diverse_ensemble(
    base_configs: List[Dict],
    model_classes: List[type] = None,
    strategy: EnsembleStrategy = None,
    device: torch.device = None
) -> ModelEnsemble:
    """
    Create a diverse ensemble with different model configurations.
    
    Args:
        base_configs: List of model configuration dictionaries
        model_classes: List of model classes (defaults to LSTMClassifier for all)
        strategy: Ensemble strategy (defaults to MajorityVoting)
        device: Device to create models on
        
    Returns:
        ModelEnsemble instance with diverse models
    """
    from src.models.lstm_model import LSTMClassifier
    from src.models.attention_lstm import AttentionLSTMClassifier
    
    if model_classes is None:
        model_classes = [LSTMClassifier] * len(base_configs)
    
    if strategy is None:
        strategy = MajorityVoting()
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create diverse models
    models = []
    model_names = []
    
    for i, (model_class, config) in enumerate(zip(model_classes, base_configs)):
        model = model_class(**config)
        models.append(model)
        
        # Generate descriptive model name
        model_type = model_class.__name__
        hidden_dim = config.get('hidden_dim', 'unknown')
        n_layers = config.get('n_layers', 'unknown')
        model_names.append(f"{model_type}_h{hidden_dim}_l{n_layers}")
    
    ensemble = ModelEnsemble(
        models=models,
        strategy=strategy,
        model_names=model_names,
        device=device
    )
    
    return ensemble