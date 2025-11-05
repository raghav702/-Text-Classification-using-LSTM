"""
Model interpretability tools for LSTM sentiment classifier.

This module provides tools for understanding model predictions including
LIME explanations, attention visualization, word importance analysis,
and adversarial example generation.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import logging
import warnings
warnings.filterwarnings('ignore')

# Optional imports for advanced interpretability
try:
    import lime
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logging.warning("LIME not available. Install with: pip install lime")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Install with: pip install shap")


class ModelInterpreter:
    """
    Comprehensive model interpretability toolkit.
    
    Provides various methods for understanding and explaining
    LSTM sentiment classifier predictions.
    """
    
    def __init__(self, model, preprocessor, device: str = 'cpu'):
        """
        Initialize model interpreter.
        
        Args:
            model: Trained LSTM model
            preprocessor: Text preprocessor instance
            device: Device to run computations on
        """
        self.model = model
        self.preprocessor = preprocessor
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize explainers
        self.lime_explainer = None
        self.shap_explainer = None
        
        if LIME_AVAILABLE:
            self._initialize_lime()
        
        if SHAP_AVAILABLE:
            self._initialize_shap()
    
    def explain_prediction(
        self,
        text: str,
        method: str = 'gradient',
        num_features: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Explain a single prediction using specified method.
        
        Args:
            text: Input text to explain
            method: Explanation method ('gradient', 'lime', 'shap', 'attention')
            num_features: Number of top features to return
            **kwargs: Additional arguments for specific methods
            
        Returns:
            Dictionary containing explanation results
        """
        try:
            if method == 'gradient':
                return self._explain_gradient_based(text, num_features, **kwargs)
            elif method == 'lime' and LIME_AVAILABLE:
                return self._explain_lime(text, num_features, **kwargs)
            elif method == 'shap' and SHAP_AVAILABLE:
                return self._explain_shap(text, num_features, **kwargs)
            elif method == 'attention':
                return self._explain_attention(text, num_features, **kwargs)
            else:
                available_methods = ['gradient', 'attention']
                if LIME_AVAILABLE:
                    available_methods.append('lime')
                if SHAP_AVAILABLE:
                    available_methods.append('shap')
                
                raise ValueError(f"Method '{method}' not available. Available methods: {available_methods}")
        
        except Exception as e:
            self.logger.error(f"Error explaining prediction: {e}")
            return {'error': str(e)}
    
    def _explain_gradient_based(self, text: str, num_features: int = 10, **kwargs) -> Dict[str, Any]:
        """Explain prediction using gradient-based attribution."""
        try:
            # Preprocess text
            tokens = self.preprocessor.tokenize(text)
            sequence = self.preprocessor.text_to_sequence(text, fit_vocabulary=False)
            
            if len(sequence) == 0:
                return {'error': 'Empty sequence after preprocessing'}
            
            # Convert to tensor and add batch dimension
            input_tensor = torch.tensor([sequence], dtype=torch.long).to(self.device)
            input_tensor.requires_grad_(True)
            
            # Forward pass
            self.model.zero_grad()
            output = self.model(input_tensor)
            prediction = torch.sigmoid(output).item()
            
            # Backward pass to get gradients
            output.backward()
            
            # Get gradients with respect to input embeddings
            gradients = input_tensor.grad
            
            if gradients is None:
                return {'error': 'No gradients computed'}
            
            # Calculate attribution scores
            gradients = gradients.squeeze().cpu().numpy()
            attribution_scores = np.abs(gradients)
            
            # Map back to tokens
            word_attributions = []
            for i, (token, score) in enumerate(zip(tokens[:len(attribution_scores)], attribution_scores)):
                if token not in ['<PAD>', '<UNK>']:
                    word_attributions.append({
                        'word': token,
                        'attribution': float(score),
                        'position': i
                    })
            
            # Sort by attribution score
            word_attributions.sort(key=lambda x: x['attribution'], reverse=True)
            
            return {
                'method': 'gradient',
                'prediction': float(prediction),
                'predicted_class': 'Positive' if prediction >= 0.5 else 'Negative',
                'confidence': float(max(prediction, 1 - prediction)),
                'word_attributions': word_attributions[:num_features],
                'text': text,
                'tokens': tokens
            }
        
        except Exception as e:
            self.logger.error(f"Error in gradient-based explanation: {e}")
            return {'error': str(e)}
    
    def _explain_lime(self, text: str, num_features: int = 10, **kwargs) -> Dict[str, Any]:
        """Explain prediction using LIME."""
        if not LIME_AVAILABLE:
            return {'error': 'LIME not available'}
        
        try:
            # Create prediction function for LIME
            def predict_fn(texts):
                predictions = []
                for t in texts:
                    try:
                        sequence = self.preprocessor.text_to_sequence(t, fit_vocabulary=False)
                        if len(sequence) == 0:
                            predictions.append([0.5, 0.5])
                            continue
                        
                        input_tensor = torch.tensor([sequence], dtype=torch.long).to(self.device)
                        
                        with torch.no_grad():
                            output = self.model(input_tensor)
                            prob = torch.sigmoid(output).item()
                            predictions.append([1 - prob, prob])
                    except:
                        predictions.append([0.5, 0.5])
                
                return np.array(predictions)
            
            # Generate explanation
            explanation = self.lime_explainer.explain_instance(
                text,
                predict_fn,
                num_features=num_features,
                **kwargs
            )
            
            # Extract feature importance
            feature_importance = explanation.as_list()
            
            # Get prediction
            prediction_probs = predict_fn([text])[0]
            prediction = prediction_probs[1]
            
            return {
                'method': 'lime',
                'prediction': float(prediction),
                'predicted_class': 'Positive' if prediction >= 0.5 else 'Negative',
                'confidence': float(max(prediction, 1 - prediction)),
                'word_attributions': [
                    {
                        'word': word,
                        'attribution': float(importance),
                        'position': -1  # LIME doesn't preserve position
                    }
                    for word, importance in feature_importance
                ],
                'text': text,
                'lime_explanation': explanation
            }
        
        except Exception as e:
            self.logger.error(f"Error in LIME explanation: {e}")
            return {'error': str(e)}
    
    def _explain_shap(self, text: str, num_features: int = 10, **kwargs) -> Dict[str, Any]:
        """Explain prediction using SHAP."""
        if not SHAP_AVAILABLE:
            return {'error': 'SHAP not available'}
        
        try:
            # Create prediction function for SHAP
            def predict_fn(texts):
                predictions = []
                for t in texts:
                    try:
                        sequence = self.preprocessor.text_to_sequence(t, fit_vocabulary=False)
                        if len(sequence) == 0:
                            predictions.append(0.5)
                            continue
                        
                        input_tensor = torch.tensor([sequence], dtype=torch.long).to(self.device)
                        
                        with torch.no_grad():
                            output = self.model(input_tensor)
                            prob = torch.sigmoid(output).item()
                            predictions.append(prob)
                    except:
                        predictions.append(0.5)
                
                return np.array(predictions)
            
            # Use SHAP's text explainer
            explainer = shap.Explainer(predict_fn, self.shap_explainer)
            shap_values = explainer([text])
            
            # Extract feature importance
            tokens = text.split()
            word_attributions = []
            
            if len(shap_values.values[0]) == len(tokens):
                for i, (token, shap_val) in enumerate(zip(tokens, shap_values.values[0])):
                    word_attributions.append({
                        'word': token,
                        'attribution': float(shap_val),
                        'position': i
                    })
            
            # Sort by absolute attribution
            word_attributions.sort(key=lambda x: abs(x['attribution']), reverse=True)
            
            # Get prediction
            prediction = predict_fn([text])[0]
            
            return {
                'method': 'shap',
                'prediction': float(prediction),
                'predicted_class': 'Positive' if prediction >= 0.5 else 'Negative',
                'confidence': float(max(prediction, 1 - prediction)),
                'word_attributions': word_attributions[:num_features],
                'text': text,
                'shap_values': shap_values
            }
        
        except Exception as e:
            self.logger.error(f"Error in SHAP explanation: {e}")
            return {'error': str(e)}
    
    def _explain_attention(self, text: str, num_features: int = 10, **kwargs) -> Dict[str, Any]:
        """Explain prediction using attention weights (if model has attention)."""
        try:
            # Check if model has attention mechanism
            if not hasattr(self.model, 'attention') and not hasattr(self.model, 'get_attention_weights'):
                return {'error': 'Model does not have attention mechanism'}
            
            # Preprocess text
            tokens = self.preprocessor.tokenize(text)
            sequence = self.preprocessor.text_to_sequence(text, fit_vocabulary=False)
            
            if len(sequence) == 0:
                return {'error': 'Empty sequence after preprocessing'}
            
            # Convert to tensor
            input_tensor = torch.tensor([sequence], dtype=torch.long).to(self.device)
            
            # Forward pass with attention
            with torch.no_grad():
                if hasattr(self.model, 'get_attention_weights'):
                    output, attention_weights = self.model.get_attention_weights(input_tensor)
                else:
                    output = self.model(input_tensor)
                    attention_weights = self.model.attention_weights if hasattr(self.model, 'attention_weights') else None
                
                prediction = torch.sigmoid(output).item()
            
            if attention_weights is None:
                return {'error': 'No attention weights available'}
            
            # Process attention weights
            attention_weights = attention_weights.squeeze().cpu().numpy()
            
            # Map to tokens
            word_attributions = []
            for i, (token, weight) in enumerate(zip(tokens[:len(attention_weights)], attention_weights)):
                if token not in ['<PAD>', '<UNK>']:
                    word_attributions.append({
                        'word': token,
                        'attribution': float(weight),
                        'position': i
                    })
            
            # Sort by attention weight
            word_attributions.sort(key=lambda x: x['attribution'], reverse=True)
            
            return {
                'method': 'attention',
                'prediction': float(prediction),
                'predicted_class': 'Positive' if prediction >= 0.5 else 'Negative',
                'confidence': float(max(prediction, 1 - prediction)),
                'word_attributions': word_attributions[:num_features],
                'text': text,
                'tokens': tokens,
                'attention_weights': attention_weights.tolist()
            }
        
        except Exception as e:
            self.logger.error(f"Error in attention explanation: {e}")
            return {'error': str(e)}
    
    def analyze_word_importance(
        self,
        texts: List[str],
        labels: List[int] = None,
        method: str = 'gradient',
        top_k: int = 20
    ) -> Dict[str, Any]:
        """
        Analyze word importance across multiple texts.
        
        Args:
            texts: List of texts to analyze
            labels: True labels (optional)
            method: Attribution method to use
            top_k: Number of top words to return
            
        Returns:
            Dictionary containing word importance analysis
        """
        try:
            word_importance = defaultdict(list)
            class_word_importance = {0: defaultdict(list), 1: defaultdict(list)}
            
            for i, text in enumerate(texts):
                explanation = self.explain_prediction(text, method=method)
                
                if 'error' in explanation:
                    continue
                
                predicted_class = 1 if explanation['prediction'] >= 0.5 else 0
                true_class = labels[i] if labels and i < len(labels) else predicted_class
                
                # Aggregate word attributions
                for word_attr in explanation.get('word_attributions', []):
                    word = word_attr['word']
                    attribution = word_attr['attribution']
                    
                    word_importance[word].append(attribution)
                    class_word_importance[true_class][word].append(attribution)
            
            # Calculate statistics
            word_stats = {}
            for word, attributions in word_importance.items():
                word_stats[word] = {
                    'mean_attribution': float(np.mean(attributions)),
                    'std_attribution': float(np.std(attributions)),
                    'frequency': len(attributions),
                    'max_attribution': float(np.max(attributions)),
                    'min_attribution': float(np.min(attributions))
                }
            
            # Sort by mean attribution
            sorted_words = sorted(word_stats.items(), key=lambda x: abs(x[1]['mean_attribution']), reverse=True)
            
            # Class-specific analysis
            class_specific_words = {}
            for class_label in [0, 1]:
                class_words = {}
                for word, attributions in class_word_importance[class_label].items():
                    if len(attributions) >= 2:  # Minimum frequency threshold
                        class_words[word] = {
                            'mean_attribution': float(np.mean(attributions)),
                            'frequency': len(attributions)
                        }
                
                # Sort by mean attribution for this class
                sorted_class_words = sorted(class_words.items(), key=lambda x: abs(x[1]['mean_attribution']), reverse=True)
                class_specific_words[class_label] = sorted_class_words[:top_k]
            
            return {
                'method': method,
                'total_texts_analyzed': len(texts),
                'total_unique_words': len(word_stats),
                'top_words_overall': sorted_words[:top_k],
                'class_specific_words': {
                    'negative': class_specific_words[0],
                    'positive': class_specific_words[1]
                },
                'word_statistics': word_stats
            }
        
        except Exception as e:
            self.logger.error(f"Error in word importance analysis: {e}")
            return {'error': str(e)}
    
    def generate_adversarial_examples(
        self,
        text: str,
        target_class: int = None,
        method: str = 'word_substitution',
        max_perturbations: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate adversarial examples for robustness testing.
        
        Args:
            text: Original text
            target_class: Target class for adversarial attack (None for untargeted)
            method: Attack method ('word_substitution', 'word_deletion', 'word_insertion')
            max_perturbations: Maximum number of word changes
            
        Returns:
            Dictionary containing adversarial examples
        """
        try:
            # Get original prediction
            original_sequence = self.preprocessor.text_to_sequence(text, fit_vocabulary=False)
            if len(original_sequence) == 0:
                return {'error': 'Empty sequence after preprocessing'}
            
            original_tensor = torch.tensor([original_sequence], dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                original_output = self.model(original_tensor)
                original_prob = torch.sigmoid(original_output).item()
                original_class = 1 if original_prob >= 0.5 else 0
            
            if method == 'word_substitution':
                return self._generate_word_substitution_adversarial(
                    text, original_class, target_class, max_perturbations, **kwargs
                )
            elif method == 'word_deletion':
                return self._generate_word_deletion_adversarial(
                    text, original_class, target_class, max_perturbations, **kwargs
                )
            elif method == 'word_insertion':
                return self._generate_word_insertion_adversarial(
                    text, original_class, target_class, max_perturbations, **kwargs
                )
            else:
                raise ValueError(f"Unknown adversarial method: {method}")
        
        except Exception as e:
            self.logger.error(f"Error generating adversarial examples: {e}")
            return {'error': str(e)}
    
    def _generate_word_substitution_adversarial(
        self,
        text: str,
        original_class: int,
        target_class: int = None,
        max_perturbations: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate adversarial examples by substituting words."""
        try:
            tokens = text.split()
            adversarial_examples = []
            
            # Simple word substitution strategy
            substitutions = {
                # Positive to negative substitutions
                'good': ['bad', 'terrible', 'awful'],
                'great': ['terrible', 'horrible', 'awful'],
                'excellent': ['terrible', 'horrible', 'bad'],
                'amazing': ['terrible', 'awful', 'horrible'],
                'wonderful': ['terrible', 'awful', 'bad'],
                'fantastic': ['terrible', 'horrible', 'awful'],
                'love': ['hate', 'dislike', 'despise'],
                'like': ['dislike', 'hate'],
                'best': ['worst', 'terrible'],
                'perfect': ['terrible', 'awful'],
                
                # Negative to positive substitutions
                'bad': ['good', 'great', 'excellent'],
                'terrible': ['great', 'excellent', 'amazing'],
                'awful': ['great', 'excellent', 'wonderful'],
                'horrible': ['great', 'excellent', 'amazing'],
                'hate': ['love', 'like', 'enjoy'],
                'dislike': ['like', 'love', 'enjoy'],
                'worst': ['best', 'great', 'excellent'],
                'boring': ['exciting', 'interesting', 'engaging']
            }
            
            for i, token in enumerate(tokens):
                token_lower = token.lower().strip('.,!?;:')
                
                if token_lower in substitutions:
                    for substitute in substitutions[token_lower]:
                        # Create modified text
                        modified_tokens = tokens.copy()
                        modified_tokens[i] = substitute
                        modified_text = ' '.join(modified_tokens)
                        
                        # Test the modified text
                        try:
                            sequence = self.preprocessor.text_to_sequence(modified_text, fit_vocabulary=False)
                            if len(sequence) == 0:
                                continue
                            
                            input_tensor = torch.tensor([sequence], dtype=torch.long).to(self.device)
                            
                            with torch.no_grad():
                                output = self.model(input_tensor)
                                prob = torch.sigmoid(output).item()
                                pred_class = 1 if prob >= 0.5 else 0
                            
                            # Check if this is a successful adversarial example
                            if target_class is None:
                                # Untargeted attack - just need to change prediction
                                success = pred_class != original_class
                            else:
                                # Targeted attack - need to match target class
                                success = pred_class == target_class
                            
                            adversarial_examples.append({
                                'modified_text': modified_text,
                                'original_word': token,
                                'substituted_word': substitute,
                                'position': i,
                                'prediction': float(prob),
                                'predicted_class': pred_class,
                                'success': success,
                                'confidence_change': float(abs(prob - (0.5 if original_class == 0 else 0.5)))
                            })
                            
                            if len(adversarial_examples) >= max_perturbations:
                                break
                        
                        except Exception:
                            continue
                
                if len(adversarial_examples) >= max_perturbations:
                    break
            
            # Sort by success and confidence change
            adversarial_examples.sort(key=lambda x: (x['success'], x['confidence_change']), reverse=True)
            
            return {
                'method': 'word_substitution',
                'original_text': text,
                'original_class': original_class,
                'target_class': target_class,
                'adversarial_examples': adversarial_examples,
                'successful_attacks': sum(1 for ex in adversarial_examples if ex['success']),
                'total_attempts': len(adversarial_examples)
            }
        
        except Exception as e:
            self.logger.error(f"Error in word substitution adversarial generation: {e}")
            return {'error': str(e)}
    
    def _generate_word_deletion_adversarial(
        self,
        text: str,
        original_class: int,
        target_class: int = None,
        max_perturbations: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate adversarial examples by deleting words."""
        try:
            tokens = text.split()
            adversarial_examples = []
            
            for i in range(len(tokens)):
                # Create modified text by removing word at position i
                modified_tokens = tokens[:i] + tokens[i+1:]
                modified_text = ' '.join(modified_tokens)
                
                if not modified_text.strip():
                    continue
                
                try:
                    sequence = self.preprocessor.text_to_sequence(modified_text, fit_vocabulary=False)
                    if len(sequence) == 0:
                        continue
                    
                    input_tensor = torch.tensor([sequence], dtype=torch.long).to(self.device)
                    
                    with torch.no_grad():
                        output = self.model(input_tensor)
                        prob = torch.sigmoid(output).item()
                        pred_class = 1 if prob >= 0.5 else 0
                    
                    # Check if this is a successful adversarial example
                    if target_class is None:
                        success = pred_class != original_class
                    else:
                        success = pred_class == target_class
                    
                    adversarial_examples.append({
                        'modified_text': modified_text,
                        'deleted_word': tokens[i],
                        'position': i,
                        'prediction': float(prob),
                        'predicted_class': pred_class,
                        'success': success
                    })
                
                except Exception:
                    continue
            
            # Sort by success
            adversarial_examples.sort(key=lambda x: x['success'], reverse=True)
            
            return {
                'method': 'word_deletion',
                'original_text': text,
                'original_class': original_class,
                'target_class': target_class,
                'adversarial_examples': adversarial_examples[:max_perturbations],
                'successful_attacks': sum(1 for ex in adversarial_examples if ex['success']),
                'total_attempts': len(adversarial_examples)
            }
        
        except Exception as e:
            self.logger.error(f"Error in word deletion adversarial generation: {e}")
            return {'error': str(e)}
    
    def _generate_word_insertion_adversarial(
        self,
        text: str,
        original_class: int,
        target_class: int = None,
        max_perturbations: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate adversarial examples by inserting words."""
        try:
            tokens = text.split()
            adversarial_examples = []
            
            # Words to insert based on target sentiment
            if target_class == 0 or (target_class is None and original_class == 1):
                # Insert negative words
                insert_words = ['not', 'never', 'terrible', 'awful', 'bad', 'horrible']
            else:
                # Insert positive words
                insert_words = ['very', 'really', 'extremely', 'absolutely', 'great', 'excellent']
            
            for insert_word in insert_words:
                for i in range(len(tokens) + 1):
                    # Insert word at position i
                    modified_tokens = tokens[:i] + [insert_word] + tokens[i:]
                    modified_text = ' '.join(modified_tokens)
                    
                    try:
                        sequence = self.preprocessor.text_to_sequence(modified_text, fit_vocabulary=False)
                        if len(sequence) == 0:
                            continue
                        
                        input_tensor = torch.tensor([sequence], dtype=torch.long).to(self.device)
                        
                        with torch.no_grad():
                            output = self.model(input_tensor)
                            prob = torch.sigmoid(output).item()
                            pred_class = 1 if prob >= 0.5 else 0
                        
                        # Check if this is a successful adversarial example
                        if target_class is None:
                            success = pred_class != original_class
                        else:
                            success = pred_class == target_class
                        
                        adversarial_examples.append({
                            'modified_text': modified_text,
                            'inserted_word': insert_word,
                            'position': i,
                            'prediction': float(prob),
                            'predicted_class': pred_class,
                            'success': success
                        })
                        
                        if len(adversarial_examples) >= max_perturbations * 2:
                            break
                    
                    except Exception:
                        continue
                
                if len(adversarial_examples) >= max_perturbations * 2:
                    break
            
            # Sort by success
            adversarial_examples.sort(key=lambda x: x['success'], reverse=True)
            
            return {
                'method': 'word_insertion',
                'original_text': text,
                'original_class': original_class,
                'target_class': target_class,
                'adversarial_examples': adversarial_examples[:max_perturbations],
                'successful_attacks': sum(1 for ex in adversarial_examples if ex['success']),
                'total_attempts': len(adversarial_examples)
            }
        
        except Exception as e:
            self.logger.error(f"Error in word insertion adversarial generation: {e}")
            return {'error': str(e)}
    
    def visualize_explanation(
        self,
        explanation: Dict[str, Any],
        save_path: str = None,
        show_plot: bool = True,
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Visualize explanation results.
        
        Args:
            explanation: Explanation dictionary from explain_prediction
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure object
        """
        try:
            if 'error' in explanation:
                raise ValueError(f"Cannot visualize explanation with error: {explanation['error']}")
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
            
            # Extract data
            word_attributions = explanation.get('word_attributions', [])
            if not word_attributions:
                raise ValueError("No word attributions found in explanation")
            
            words = [attr['word'] for attr in word_attributions]
            attributions = [attr['attribution'] for attr in word_attributions]
            
            # Plot 1: Bar chart of word attributions
            colors = ['red' if attr < 0 else 'green' for attr in attributions]
            bars = ax1.barh(range(len(words)), attributions, color=colors, alpha=0.7)
            ax1.set_yticks(range(len(words)))
            ax1.set_yticklabels(words)
            ax1.set_xlabel('Attribution Score')
            ax1.set_title(f'Word Attributions ({explanation.get("method", "Unknown")} method)')
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, (bar, attr) in enumerate(zip(bars, attributions)):
                ax1.text(attr + (0.01 if attr >= 0 else -0.01), i, f'{attr:.3f}', 
                        va='center', ha='left' if attr >= 0 else 'right')
            
            # Plot 2: Text with highlighted words
            ax2.axis('off')
            text = explanation.get('text', '')
            
            # Create highlighted text visualization
            text_parts = []
            tokens = text.split()
            
            # Create a mapping of words to their attributions
            word_to_attr = {attr['word']: attr['attribution'] for attr in word_attributions}
            
            for token in tokens:
                # Clean token for matching
                clean_token = token.lower().strip('.,!?;:')
                attribution = word_to_attr.get(clean_token, 0)
                
                # Determine color based on attribution
                if attribution > 0:
                    color = 'lightgreen'
                    alpha = min(0.8, abs(attribution) * 2)
                elif attribution < 0:
                    color = 'lightcoral'
                    alpha = min(0.8, abs(attribution) * 2)
                else:
                    color = 'lightgray'
                    alpha = 0.3
                
                text_parts.append((token, color, alpha))
            
            # Display highlighted text
            x_pos = 0.05
            y_pos = 0.7
            line_height = 0.15
            
            for token, color, alpha in text_parts:
                # Check if we need to wrap to next line
                if x_pos > 0.9:
                    x_pos = 0.05
                    y_pos -= line_height
                
                # Add background rectangle
                bbox = dict(boxstyle="round,pad=0.3", facecolor=color, alpha=alpha)
                ax2.text(x_pos, y_pos, token, transform=ax2.transAxes, 
                        bbox=bbox, fontsize=10, va='center')
                
                # Update x position
                x_pos += len(token) * 0.012 + 0.02
            
            # Add prediction info
            pred_text = f"Prediction: {explanation.get('predicted_class', 'Unknown')} "
            pred_text += f"(Confidence: {explanation.get('confidence', 0):.3f})"
            ax2.text(0.05, 0.3, pred_text, transform=ax2.transAxes, 
                    fontsize=12, fontweight='bold')
            
            # Add legend
            legend_elements = [
                plt.Rectangle((0, 0), 1, 1, facecolor='lightgreen', alpha=0.7, label='Positive Attribution'),
                plt.Rectangle((0, 0), 1, 1, facecolor='lightcoral', alpha=0.7, label='Negative Attribution')
            ]
            ax2.legend(handles=legend_elements, loc='lower right')
            
            ax2.set_title('Text with Attribution Highlighting')
            
            plt.tight_layout()
            
            # Save if requested
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Explanation visualization saved to {save_path}")
            
            # Show if requested
            if show_plot:
                plt.show()
            
            return fig
        
        except Exception as e:
            self.logger.error(f"Error visualizing explanation: {e}")
            # Return empty figure on error
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', transform=ax.transAxes)
            return fig
    
    def _initialize_lime(self):
        """Initialize LIME explainer."""
        try:
            self.lime_explainer = LimeTextExplainer(
                class_names=['Negative', 'Positive'],
                mode='classification'
            )
        except Exception as e:
            self.logger.error(f"Error initializing LIME: {e}")
    
    def _initialize_shap(self):
        """Initialize SHAP explainer."""
        try:
            # Create a simple background dataset for SHAP
            background_texts = [
                "This is a neutral sentence.",
                "The movie was okay.",
                "It was an average film.",
                "The story was simple."
            ]
            self.shap_explainer = background_texts
        except Exception as e:
            self.logger.error(f"Error initializing SHAP: {e}")


# Convenience functions
def explain_prediction(
    model,
    preprocessor,
    text: str,
    method: str = 'gradient',
    device: str = 'cpu',
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to explain a single prediction.
    
    Args:
        model: Trained model
        preprocessor: Text preprocessor
        text: Input text
        method: Explanation method
        device: Device to use
        **kwargs: Additional arguments
        
    Returns:
        Explanation dictionary
    """
    interpreter = ModelInterpreter(model, preprocessor, device)
    return interpreter.explain_prediction(text, method, **kwargs)


def analyze_word_importance(
    model,
    preprocessor,
    texts: List[str],
    labels: List[int] = None,
    method: str = 'gradient',
    device: str = 'cpu',
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to analyze word importance across texts.
    
    Args:
        model: Trained model
        preprocessor: Text preprocessor
        texts: List of texts
        labels: True labels (optional)
        method: Attribution method
        device: Device to use
        **kwargs: Additional arguments
        
    Returns:
        Word importance analysis
    """
    interpreter = ModelInterpreter(model, preprocessor, device)
    return interpreter.analyze_word_importance(texts, labels, method, **kwargs)


def generate_adversarial_examples(
    model,
    preprocessor,
    text: str,
    device: str = 'cpu',
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to generate adversarial examples.
    
    Args:
        model: Trained model
        preprocessor: Text preprocessor
        text: Input text
        device: Device to use
        **kwargs: Additional arguments
        
    Returns:
        Adversarial examples
    """
    interpreter = ModelInterpreter(model, preprocessor, device)
    return interpreter.generate_adversarial_examples(text, **kwargs)