"""
LSTM Sentiment Classifier Model
Implements a bidirectional LSTM neural network for binary sentiment classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class LSTMClassifier(nn.Module):
    """
    LSTM-based sentiment classifier with embedding layer and fully connected head.
    
    Architecture:
    - Embedding layer (with optional GloVe initialization)
    - Bidirectional LSTM layers
    - Fully connected layers with dropout
    - Binary classification output
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_dim: int = 128,
        output_dim: int = 1,
        n_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        pad_idx: int = 0
    ):
        """
        Initialize the LSTM classifier.
        
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Hidden dimension of LSTM layers
            output_dim: Output dimension (1 for binary classification)
            n_layers: Number of LSTM layers
            dropout: Dropout rate for regularization
            bidirectional: Whether to use bidirectional LSTM
            pad_idx: Index of padding token in vocabulary
        """
        super(LSTMClassifier, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.pad_idx = pad_idx
        
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
        
        # Fully connected layers for classification
        self.fc1 = nn.Linear(self.lstm_output_dim, 64)
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
    
    def freeze_embeddings(self):
        """Freeze embedding layer weights."""
        self.embedding.weight.requires_grad = False
        print("Embedding layer frozen")
    
    def unfreeze_embeddings(self):
        """Unfreeze embedding layer weights for fine-tuning."""
        self.embedding.weight.requires_grad = True
        print("Embedding layer unfrozen for fine-tuning")
    
    def get_embedding_statistics(self) -> Dict[str, any]:
        """
        Get statistics about the embedding layer.
        
        Returns:
            Dictionary with embedding layer statistics
        """
        embedding_weights = self.embedding.weight.data
        
        stats = {
            'embedding_shape': tuple(embedding_weights.shape),
            'embedding_mean': float(embedding_weights.mean()),
            'embedding_std': float(embedding_weights.std()),
            'embedding_min': float(embedding_weights.min()),
            'embedding_max': float(embedding_weights.max()),
            'frozen': not self.embedding.weight.requires_grad,
            'zero_embeddings': int((embedding_weights.norm(dim=1) == 0).sum()),
            'pad_idx': self.pad_idx
        }
        
        return stats
    
    def _init_weights(self):
        """Initialize model weights using Xavier uniform initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    # Initialize LSTM weights
                    nn.init.xavier_uniform_(param)
                elif 'fc' in name:
                    # Initialize fully connected weights
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                # Initialize biases to zero
                nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            lengths: Optional tensor of actual sequence lengths for packed sequences
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        batch_size = x.size(0)
        device = x.device
        
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
        
        # Use the last hidden state from both directions
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            # hidden shape: (n_layers * 2, batch_size, hidden_dim)
            forward_hidden = hidden[-2]  # Last layer forward direction
            backward_hidden = hidden[-1]  # Last layer backward direction
            final_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        else:
            # Use the last layer's hidden state
            final_hidden = hidden[-1]  # (batch_size, hidden_dim)
        
        # Fully connected layers
        fc_input = self.fc_dropout(final_hidden)
        fc1_output = F.relu(self.fc1(fc_input))
        fc1_output = self.fc_dropout(fc1_output)
        
        # Final output layer (no activation - will be applied in loss function)
        output = self.fc2(fc1_output)  # (batch_size, output_dim)
        
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
    
    def predict(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None, threshold: float = 0.5) -> torch.Tensor:
        """
        Get binary predictions.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            lengths: Optional tensor of actual sequence lengths
            threshold: Decision threshold for binary classification
            
        Returns:
            Binary prediction tensor of shape (batch_size, output_dim)
        """
        probabilities = self.predict_proba(x, lengths)
        predictions = (probabilities >= threshold).float()
        return predictions
    
    def get_model_info(self) -> dict:
        """
        Get model architecture information.
        
        Returns:
            Dictionary containing model configuration
        """
        return {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'n_layers': self.n_layers,
            'dropout': self.dropout,
            'bidirectional': self.bidirectional,
            'lstm_output_dim': self.lstm_output_dim,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }