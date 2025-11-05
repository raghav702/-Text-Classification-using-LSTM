"""
Attention Visualization Utilities
Tools for visualizing and interpreting attention weights from attention-enhanced models
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import pandas as pd


class AttentionVisualizer:
    """
    Utility class for visualizing attention weights and model interpretability.
    """
    
    def __init__(self, tokenizer=None):
        """
        Initialize attention visualizer.
        
        Args:
            tokenizer: Optional tokenizer for converting indices back to tokens
        """
        self.tokenizer = tokenizer
        
    def plot_attention_heatmap(
        self,
        attention_weights: torch.Tensor,
        tokens: List[str],
        title: str = "Attention Weights",
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ):
        """
        Plot attention weights as a heatmap.
        
        Args:
            attention_weights: Attention weights tensor of shape (seq_len, seq_len) or (seq_len,)
            tokens: List of tokens corresponding to the sequence
            title: Title for the plot
            figsize: Figure size tuple
            save_path: Optional path to save the plot
        """
        # Convert to numpy if tensor
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()
        
        plt.figure(figsize=figsize)
        
        if attention_weights.ndim == 2:
            # Self-attention heatmap (seq_len x seq_len)
            sns.heatmap(
                attention_weights,
                xticklabels=tokens,
                yticklabels=tokens,
                cmap='Blues',
                annot=False,
                cbar=True,
                square=True
            )
            plt.xlabel('Key Tokens')
            plt.ylabel('Query Tokens')
        else:
            # Pooling attention (seq_len,)
            attention_df = pd.DataFrame({
                'Token': tokens,
                'Attention Weight': attention_weights
            })
            
            plt.barh(range(len(tokens)), attention_weights, color='skyblue')
            plt.yticks(range(len(tokens)), tokens)
            plt.xlabel('Attention Weight')
            plt.ylabel('Tokens')
            plt.gca().invert_yaxis()
        
        plt.title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_attention_distribution(
        self,
        attention_weights: torch.Tensor,
        tokens: List[str],
        top_k: int = 10,
        title: str = "Top Attended Tokens",
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ):
        """
        Plot distribution of attention weights focusing on top-k tokens.
        
        Args:
            attention_weights: Attention weights tensor of shape (seq_len,)
            tokens: List of tokens corresponding to the sequence
            top_k: Number of top tokens to highlight
            title: Title for the plot
            figsize: Figure size tuple
            save_path: Optional path to save the plot
        """
        # Convert to numpy if tensor
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()
        
        # Get top-k indices
        top_indices = np.argsort(attention_weights)[-top_k:][::-1]
        
        plt.figure(figsize=figsize)
        
        # Create bar plot
        colors = ['red' if i in top_indices else 'lightblue' for i in range(len(tokens))]
        bars = plt.bar(range(len(tokens)), attention_weights, color=colors)
        
        # Highlight top-k tokens
        for i, idx in enumerate(top_indices):
            plt.annotate(
                f'{tokens[idx]}\n{attention_weights[idx]:.3f}',
                xy=(idx, attention_weights[idx]),
                xytext=(0, 10),
                textcoords='offset points',
                ha='center',
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7)
            )
        
        plt.xlabel('Token Position')
        plt.ylabel('Attention Weight')
        plt.title(title)
        plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def analyze_attention_patterns(
        self,
        attention_weights: torch.Tensor,
        tokens: List[str],
        sentiment_label: str = None
    ) -> Dict:
        """
        Analyze attention patterns and extract insights.
        
        Args:
            attention_weights: Attention weights tensor
            tokens: List of tokens
            sentiment_label: Optional sentiment label for analysis
            
        Returns:
            Dictionary with attention analysis results
        """
        # Convert to numpy if tensor
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()
        
        # Handle different attention weight shapes
        if attention_weights.ndim == 2:
            # For self-attention, use average attention received by each token
            pooling_weights = attention_weights.mean(axis=0)
        else:
            # For pooling attention, use directly
            pooling_weights = attention_weights
        
        # Calculate statistics
        max_attention_idx = np.argmax(pooling_weights)
        min_attention_idx = np.argmin(pooling_weights)
        
        # Get top-k most attended tokens
        top_k = min(5, len(tokens))
        top_indices = np.argsort(pooling_weights)[-top_k:][::-1]
        
        analysis = {
            'max_attention_token': tokens[max_attention_idx],
            'max_attention_weight': float(pooling_weights[max_attention_idx]),
            'min_attention_token': tokens[min_attention_idx],
            'min_attention_weight': float(pooling_weights[min_attention_idx]),
            'attention_entropy': float(-np.sum(pooling_weights * np.log(pooling_weights + 1e-8))),
            'attention_concentration': float(np.max(pooling_weights) / np.mean(pooling_weights)),
            'top_attended_tokens': [
                {
                    'token': tokens[idx],
                    'weight': float(pooling_weights[idx]),
                    'position': int(idx)
                }
                for idx in top_indices
            ],
            'sentiment_label': sentiment_label
        }
        
        return analysis
    
    def compare_attention_across_samples(
        self,
        attention_data: List[Dict],
        save_path: Optional[str] = None
    ):
        """
        Compare attention patterns across multiple samples.
        
        Args:
            attention_data: List of dictionaries containing attention info for each sample
            save_path: Optional path to save the comparison plot
        """
        fig, axes = plt.subplots(len(attention_data), 1, figsize=(12, 4 * len(attention_data)))
        
        if len(attention_data) == 1:
            axes = [axes]
        
        for i, data in enumerate(attention_data):
            attention_weights = data['attention_weights']
            tokens = data['tokens']
            label = data.get('label', f'Sample {i+1}')
            
            # Convert to numpy if tensor
            if isinstance(attention_weights, torch.Tensor):
                attention_weights = attention_weights.detach().cpu().numpy()
            
            # Handle different attention shapes
            if attention_weights.ndim == 2:
                attention_weights = attention_weights.mean(axis=0)
            
            # Plot attention distribution
            axes[i].bar(range(len(tokens)), attention_weights, color='skyblue', alpha=0.7)
            axes[i].set_title(f'{label} - Attention Distribution')
            axes[i].set_xlabel('Token Position')
            axes[i].set_ylabel('Attention Weight')
            axes[i].set_xticks(range(len(tokens)))
            axes[i].set_xticklabels(tokens, rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_attention_summary_report(
        self,
        attention_analyses: List[Dict],
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create a summary report of attention patterns across multiple samples.
        
        Args:
            attention_analyses: List of attention analysis dictionaries
            output_path: Optional path to save the report as CSV
            
        Returns:
            DataFrame with attention summary statistics
        """
        summary_data = []
        
        for i, analysis in enumerate(attention_analyses):
            summary_data.append({
                'sample_id': i,
                'sentiment_label': analysis.get('sentiment_label', 'unknown'),
                'max_attention_token': analysis['max_attention_token'],
                'max_attention_weight': analysis['max_attention_weight'],
                'attention_entropy': analysis['attention_entropy'],
                'attention_concentration': analysis['attention_concentration'],
                'top_token_1': analysis['top_attended_tokens'][0]['token'],
                'top_weight_1': analysis['top_attended_tokens'][0]['weight'],
                'top_token_2': analysis['top_attended_tokens'][1]['token'] if len(analysis['top_attended_tokens']) > 1 else '',
                'top_weight_2': analysis['top_attended_tokens'][1]['weight'] if len(analysis['top_attended_tokens']) > 1 else 0.0
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        if output_path:
            summary_df.to_csv(output_path, index=False)
            print(f"Attention summary report saved to: {output_path}")
        
        return summary_df
    
    def visualize_attention_evolution(
        self,
        attention_weights_sequence: List[torch.Tensor],
        tokens: List[str],
        epoch_labels: List[str] = None,
        title: str = "Attention Evolution During Training",
        figsize: Tuple[int, int] = (15, 10),
        save_path: Optional[str] = None
    ):
        """
        Visualize how attention patterns evolve during training.
        
        Args:
            attention_weights_sequence: List of attention weight tensors from different epochs
            tokens: List of tokens
            epoch_labels: Optional labels for each epoch
            title: Title for the plot
            figsize: Figure size tuple
            save_path: Optional path to save the plot
        """
        n_epochs = len(attention_weights_sequence)
        
        if epoch_labels is None:
            epoch_labels = [f'Epoch {i+1}' for i in range(n_epochs)]
        
        fig, axes = plt.subplots(1, n_epochs, figsize=figsize)
        
        if n_epochs == 1:
            axes = [axes]
        
        for i, (attention_weights, epoch_label) in enumerate(zip(attention_weights_sequence, epoch_labels)):
            # Convert to numpy if tensor
            if isinstance(attention_weights, torch.Tensor):
                attention_weights = attention_weights.detach().cpu().numpy()
            
            # Handle different attention shapes
            if attention_weights.ndim == 2:
                attention_weights = attention_weights.mean(axis=0)
            
            # Plot attention distribution
            axes[i].bar(range(len(tokens)), attention_weights, color='skyblue', alpha=0.7)
            axes[i].set_title(epoch_label)
            axes[i].set_xlabel('Token Position')
            axes[i].set_ylabel('Attention Weight')
            axes[i].set_xticks(range(len(tokens)))
            axes[i].set_xticklabels(tokens, rotation=45, ha='right')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def extract_attention_insights(
    model,
    text_samples: List[str],
    tokenizer,
    device: torch.device,
    max_length: int = 500
) -> List[Dict]:
    """
    Extract attention insights from a batch of text samples.
    
    Args:
        model: Attention-enhanced model
        text_samples: List of text samples to analyze
        tokenizer: Tokenizer for preprocessing
        device: Device to run inference on
        max_length: Maximum sequence length
        
    Returns:
        List of dictionaries containing attention analysis for each sample
    """
    model.eval()
    insights = []
    
    with torch.no_grad():
        for text in text_samples:
            # Preprocess text
            tokens = tokenizer.tokenize(text)[:max_length]
            token_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)]).to(device)
            
            # Get predictions with attention
            predictions, probabilities, attention_info = model.predict_with_attention(token_ids)
            
            # Extract attention weights
            pooling_attention = attention_info.get('pooling_attention_weights')
            if pooling_attention is not None:
                pooling_attention = pooling_attention.squeeze(0)  # Remove batch dimension
            
            # Create insight dictionary
            insight = {
                'text': text,
                'tokens': tokens,
                'prediction': float(predictions.squeeze()),
                'probability': float(probabilities.squeeze()),
                'attention_weights': pooling_attention,
                'self_attention_weights': attention_info.get('self_attention_weights')
            }
            
            insights.append(insight)
    
    return insights