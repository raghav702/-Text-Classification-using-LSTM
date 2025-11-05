"""
LSTM Sentiment Classifier with Attention Mechanism
Implements self-attention mechanism for better context understanding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math


class SelfAttention(nn.Module):
    """
    Self-attention mechanism for sequence modeling.
    Computes attention weights to focus on important words in the sequence.
    """
    
    def __init__(self, hidden_dim: int, attention_dim: int = 64):
        """
        Initialize self-attention layer.
        
        Args:
            hidden_dim: Hidden dimension of the input sequences
            attention_dim: Dimension of attention computation
        """
        super(SelfAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        
        # Linear layers for attention computation
        self.query = nn.Linear(hidden_dim, attention_dim)
        self.key = nn.Linear(hidden_dim, attention_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Scaling factor for attention scores
        self.scale = math.sqrt(attention_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through self-attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            mask: Optional mask tensor of shape (batch_size, seq_len)
            
        Returns:
            Tuple of (attended_output, attention_weights)
        """
        batch_size, seq_len, hidden_dim = x.size()
        
        # Compute query, key, value
        Q = self.query(x)  # (batch_size, seq_len, attention_dim)
        K = self.key(x)    # (batch_size, seq_len, attention_dim)
        V = self.value(x)  # (batch_size, seq_len, hidden_dim)
        
        # Compute attention scores
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # (batch_size, seq_len, seq_len)
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask to match attention scores shape
            mask = mask.unsqueeze(1).expand(-1, seq_len, -1)
            scores.masked_fill_(mask == 0, -1e9)
        
        # Compute attention weights
        attention_weights = F.softmax(scores, dim=-1)  # (batch_size, seq_len, seq_len)
        
        # Apply attention to values
        attended = torch.bmm(attention_weights, V)  # (batch_size, seq_len, hidden_dim)
        
        # Output projection
        output = self.output_proj(attended)
        
        return output, attention_weights


class AttentionPooling(nn.Module):
    """
    Attention-based pooling for sequence representation.
    Computes a weighted average of sequence elements based on attention scores.
    """
    
    def __init__(self, hidden_dim: int):
        """
        Initialize attention pooling layer.
        
        Args:
            hidden_dim: Hidden dimension of input sequences
        """
        super(AttentionPooling, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # Attention computation layers
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through attention pooling.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            mask: Optional mask tensor of shape (batch_size, seq_len)
            
        Returns:
            Tuple of (pooled_output, attention_weights)
        """
        # Compute attention scores
        attention_scores = self.attention(x).squeeze(-1)  # (batch_size, seq_len)
        
        # Apply mask if provided
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        
        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, seq_len)
        
        # Apply attention to input
        pooled_output = torch.bmm(
            attention_weights.unsqueeze(1), x
        ).squeeze(1)  # (batch_size, hidden_dim)
        
        return pooled_output, attention_weights


class AttentionLSTMClassifier(nn.Module):
    """
    LSTM-based sentiment classifier enhanced with self-attention mechanism.
    
    Architecture:
    - Embedding layer (with optional GloVe initialization)
    - Bidirectional LSTM layers
    - Self-attention mechanism
    - Attention-based pooling
    - Fully connected layers with dropout
    - Binary classification output
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_dim: int = 128,
        attention_dim: int = 64,
        output_dim: int = 1,
        n_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        pad_idx: int = 0,
        use_attention_pooling: bool = True
    ):
        """
        Initialize the attention-enhanced LSTM classifier.
        
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Hidden dimension of LSTM layers
            attention_dim: Dimension of attention computation
            output_dim: Output dimension (1 for binary classification)
            n_layers: Number of LSTM layers
            dropout: Dropout rate for regularization
            bidirectional: Whether to use bidirectional LSTM
            pad_idx: Index of padding token in vocabulary
            use_attention_pooling: Whether to use attention-based pooling
        """
        super(AttentionLSTMClassifier, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.pad_idx = pad_idx
        self.use_attention_pooling = use_attention_pooling
        
        # Calculate LSTM output dimension based on bidirectionality
        self.lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx
        )
        
        # Embedding dropout
        self.embedding_dropout = nn.Dropout(dropout)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Self-attention mechanism
        self.self_attention = SelfAttention(self.lstm_output_dim, attention_dim)
        
        # Attention-based pooling (optional)
        if use_attention_pooling:
            self.attention_pooling = AttentionPooling(self.lstm_output_dim)
            fc_input_dim = self.lstm_output_dim
        else:
            self.attention_pooling = None
            fc_input_dim = self.lstm_output_dim
        
        # Fully connected layers for classification
        self.fc1 = nn.Linear(fc_input_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)
        
        # Dropout for fully connected layers
        self.fc_dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden and cell states for LSTM.
        
        Args:
            batch_size: Size of the current batch
            device: Device to place tensors on
            
        Returns:
            Tuple of (hidden_state, cell_state)
        """
        num_directions = 2 if self.bidirectional else 1
        
        hidden = torch.zeros(
            self.n_layers * num_directions,
            batch_size,
            self.hidden_dim,
            device=device
        )
        
        cell = torch.zeros(
            self.n_layers * num_directions,
            batch_size,
            self.hidden_dim,
            device=device
        )
        
        return hidden, cell
    
    def create_padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create padding mask for attention computation.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            
        Returns:
            Mask tensor of shape (batch_size, seq_len)
        """
        return (x != self.pad_idx).float()
    
    def load_pretrained_embeddings(
        self, 
        pretrained_embeddings: torch.Tensor, 
        freeze_embeddings: bool = False,
        fine_tune_embeddings: bool = True
    ):
        """
        Load pre-trained embeddings (e.g., GloVe) into the embedding layer.
        
        Args:
            pretrained_embeddings: Pre-trained embedding matrix of shape (vocab_size, embedding_dim)
            freeze_embeddings: Whether to freeze embedding weights during training
            fine_tune_embeddings: Whether to allow fine-tuning of embeddings
        """
        if pretrained_embeddings.shape != (self.vocab_size, self.embedding_dim):
            raise ValueError(
                f"Pretrained embeddings shape {pretrained_embeddings.shape} "
                f"doesn't match expected shape ({self.vocab_size}, {self.embedding_dim})"
            )
        
        # Load the embeddings
        self.embedding.weight.data.copy_(pretrained_embeddings)
        
        # Zero out padding token embedding
        if self.pad_idx is not None:
            self.embedding.weight.data[self.pad_idx].fill_(0)
        
        # Set embedding layer training behavior
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False
            print("Embedding weights frozen (will not be updated during training)")
        elif fine_tune_embeddings:
            self.embedding.weight.requires_grad = True
            print("Embedding weights will be fine-tuned during training")
        
        print(f"Loaded pre-trained embeddings: {pretrained_embeddings.shape}")
    
    def _init_weights(self):
        """Initialize model weights using Xavier uniform initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name or 'attention' in name or 'fc' in name:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None, return_attention: bool = False) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            lengths: Optional tensor of actual sequence lengths for packed sequences
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor of shape (batch_size, output_dim) or tuple with attention weights
        """
        batch_size = x.size(0)
        device = x.device
        
        # Create padding mask
        padding_mask = self.create_padding_mask(x)
        
        # Embedding layer
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        embedded = self.embedding_dropout(embedded)
        
        # Initialize hidden states
        hidden, cell = self.init_hidden(batch_size, device)
        
        # LSTM forward pass
        if lengths is not None:
            # Pack padded sequences for efficiency
            packed_embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, (hidden, cell) = self.lstm(packed_embedded, (hidden, cell))
            lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True
            )
        else:
            lstm_output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        
        # Self-attention mechanism
        attended_output, self_attention_weights = self.self_attention(lstm_output, padding_mask)
        
        # Combine LSTM output with attention (residual connection)
        combined_output = lstm_output + attended_output
        
        # Pooling strategy
        if self.use_attention_pooling:
            # Attention-based pooling
            pooled_output, pooling_attention_weights = self.attention_pooling(combined_output, padding_mask)
        else:
            # Use the last hidden state from both directions
            if self.bidirectional:
                forward_hidden = hidden[-2]
                backward_hidden = hidden[-1]
                pooled_output = torch.cat([forward_hidden, backward_hidden], dim=1)
            else:
                pooled_output = hidden[-1]
            pooling_attention_weights = None
        
        # Fully connected layers
        fc_input = self.fc_dropout(pooled_output)
        fc1_output = F.relu(self.fc1(fc_input))
        fc1_output = self.fc_dropout(fc1_output)
        
        # Final output layer
        output = self.fc2(fc1_output)
        
        if return_attention:
            attention_info = {
                'self_attention_weights': self_attention_weights,
                'pooling_attention_weights': pooling_attention_weights
            }
            return output, attention_info
        
        return output
    
    def predict_proba(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get prediction probabilities using sigmoid activation.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            lengths: Optional tensor of actual sequence lengths
            
        Returns:
            Probability tensor of shape (batch_size, output_dim)
        """
        with torch.no_grad():
            logits = self.forward(x, lengths)
            probabilities = torch.sigmoid(logits)
        return probabilities
    
    def predict_with_attention(
        self, 
        x: torch.Tensor, 
        lengths: Optional[torch.Tensor] = None, 
        threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Get predictions with attention weights for interpretability.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            lengths: Optional tensor of actual sequence lengths
            threshold: Decision threshold for binary classification
            
        Returns:
            Tuple of (predictions, probabilities, attention_info)
        """
        with torch.no_grad():
            logits, attention_info = self.forward(x, lengths, return_attention=True)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= threshold).float()
        
        return predictions, probabilities, attention_info
    
    def get_model_info(self) -> dict:
        """
        Get model architecture information.
        
        Returns:
            Dictionary containing model configuration
        """
        return {
            'model_type': 'AttentionLSTMClassifier',
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'attention_dim': self.attention_dim,
            'output_dim': self.output_dim,
            'n_layers': self.n_layers,
            'dropout': self.dropout,
            'bidirectional': self.bidirectional,
            'use_attention_pooling': self.use_attention_pooling,
            'lstm_output_dim': self.lstm_output_dim,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }