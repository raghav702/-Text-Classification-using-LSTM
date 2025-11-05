"""
Text preprocessing module for LSTM sentiment classifier.

This module provides the TextPreprocessor class that handles tokenization,
vocabulary building, and sequence conversion for text data.
"""

import re
import torch
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional


class TextPreprocessor:
    """
    Text preprocessing class for sentiment analysis.
    
    Handles tokenization, vocabulary building, sequence conversion,
    padding, and truncation for text data.
    """
    
    def __init__(self, max_vocab_size: int = 10000, min_freq: int = 2, max_length: int = 500):
        """
        Initialize the TextPreprocessor.
        
        Args:
            max_vocab_size: Maximum vocabulary size
            min_freq: Minimum frequency threshold for vocabulary inclusion
            max_length: Maximum sequence length for padding/truncation
        """
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.max_length = max_length
        
        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.START_TOKEN = '<START>'
        self.END_TOKEN = '<END>'
        
        # Vocabulary mappings
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0
        
        # Initialize with special tokens
        self._init_special_tokens()
    
    def _init_special_tokens(self):
        """Initialize special tokens in vocabulary."""
        special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.START_TOKEN, self.END_TOKEN]
        
        for i, token in enumerate(special_tokens):
            self.word_to_idx[token] = i
            self.idx_to_word[i] = token
        
        self.vocab_size = len(special_tokens)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into individual tokens.
        
        Uses regex-based approach to split text while preserving punctuation
        and handling contractions appropriately.
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens
        """
        if not isinstance(text, str):
            return []
        
        # Convert to lowercase
        text = text.lower().strip()
        
        # Handle contractions and punctuation
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can't", "cannot", text)
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"'re", " are", text)
        text = re.sub(r"'ve", " have", text)
        text = re.sub(r"'ll", " will", text)
        text = re.sub(r"'d", " would", text)
        text = re.sub(r"'m", " am", text)
        
        # Split on whitespace and punctuation, keeping punctuation as separate tokens
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
        
        # Filter out empty tokens
        tokens = [token for token in tokens if token.strip()]
        
        return tokens
    
    def build_vocabulary(self, texts: List[str]) -> Dict[str, int]:
        """
        Build vocabulary from a list of texts.
        
        Creates word-to-index mapping based on frequency thresholds
        and vocabulary size limits.
        
        Args:
            texts: List of text strings
            
        Returns:
            Dictionary mapping words to their frequencies
        """
        # Count word frequencies
        word_freq = Counter()
        
        for text in texts:
            tokens = self.tokenize(text)
            word_freq.update(tokens)
        
        # Filter by minimum frequency and sort by frequency
        filtered_words = {word: freq for word, freq in word_freq.items() 
                         if freq >= self.min_freq}
        
        # Sort by frequency (descending) and take top words
        sorted_words = sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)
        
        # Limit vocabulary size (accounting for special tokens already added)
        max_new_words = self.max_vocab_size - self.vocab_size
        top_words = sorted_words[:max_new_words]
        
        # Add words to vocabulary
        for word, freq in top_words:
            if word not in self.word_to_idx:
                idx = self.vocab_size
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
                self.vocab_size += 1
        
        return dict(top_words)  
  
    def text_to_sequence(self, text: str) -> torch.Tensor:
        """
        Convert text to numerical sequence using vocabulary mapping.
        
        Args:
            text: Input text string
            
        Returns:
            PyTorch tensor of token indices
        """
        tokens = self.tokenize(text)
        
        # Convert tokens to indices, using UNK for out-of-vocabulary words
        unk_idx = self.word_to_idx[self.UNK_TOKEN]
        indices = [self.word_to_idx.get(token, unk_idx) for token in tokens]
        
        return torch.tensor(indices, dtype=torch.long)
    
    def pad_sequences(self, sequences: List[torch.Tensor], max_length: Optional[int] = None) -> torch.Tensor:
        """
        Apply padding to ensure uniform sequence lengths.
        
        Args:
            sequences: List of tensor sequences
            max_length: Maximum length for padding (uses self.max_length if None)
            
        Returns:
            Padded tensor of shape (batch_size, max_length)
        """
        if not sequences:
            return torch.empty(0, dtype=torch.long)
        
        if max_length is None:
            max_length = self.max_length
        
        pad_idx = self.word_to_idx[self.PAD_TOKEN]
        batch_size = len(sequences)
        
        # Initialize padded tensor
        padded = torch.full((batch_size, max_length), pad_idx, dtype=torch.long)
        
        for i, seq in enumerate(sequences):
            # Truncate if sequence is too long
            seq_len = min(len(seq), max_length)
            padded[i, :seq_len] = seq[:seq_len]
        
        return padded
    
    def truncate_sequence(self, sequence: torch.Tensor, max_length: Optional[int] = None) -> torch.Tensor:
        """
        Truncate sequence to maximum length.
        
        Args:
            sequence: Input tensor sequence
            max_length: Maximum length (uses self.max_length if None)
            
        Returns:
            Truncated tensor
        """
        if max_length is None:
            max_length = self.max_length
        
        if len(sequence) <= max_length:
            return sequence
        
        return sequence[:max_length]
    
    def preprocess_texts(self, texts: List[str], fit_vocabulary: bool = False) -> torch.Tensor:
        """
        Complete preprocessing pipeline for a batch of texts.
        
        Args:
            texts: List of text strings
            fit_vocabulary: Whether to build vocabulary from these texts
            
        Returns:
            Padded tensor of sequences
        """
        if fit_vocabulary:
            self.build_vocabulary(texts)
        
        # Convert texts to sequences
        sequences = [self.text_to_sequence(text) for text in texts]
        
        # Apply padding
        padded_sequences = self.pad_sequences(sequences)
        
        return padded_sequences
    
    def get_vocab_info(self) -> Dict[str, any]:
        """
        Get vocabulary information.
        
        Returns:
            Dictionary with vocabulary statistics
        """
        return {
            'vocab_size': self.vocab_size,
            'max_vocab_size': self.max_vocab_size,
            'min_freq': self.min_freq,
            'max_length': self.max_length,
            'pad_token': self.PAD_TOKEN,
            'unk_token': self.UNK_TOKEN,
            'pad_idx': self.word_to_idx.get(self.PAD_TOKEN, 0),
            'unk_idx': self.word_to_idx.get(self.UNK_TOKEN, 1)
        }
    
    def save_vocabulary(self, filepath: str):
        """
        Save vocabulary to file.
        
        Args:
            filepath: Path to save vocabulary
        """
        vocab_data = {
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'vocab_size': self.vocab_size,
            'max_vocab_size': self.max_vocab_size,
            'min_freq': self.min_freq,
            'max_length': self.max_length
        }
        
        torch.save(vocab_data, filepath)
    
    def load_vocabulary(self, filepath: str):
        """
        Load vocabulary from file.
        
        Args:
            filepath: Path to load vocabulary from
        """
        vocab_data = torch.load(filepath)
        
        self.word_to_idx = vocab_data['word_to_idx']
        self.idx_to_word = vocab_data['idx_to_word']
        self.vocab_size = vocab_data['vocab_size']
        self.max_vocab_size = vocab_data['max_vocab_size']
        self.min_freq = vocab_data['min_freq']
        self.max_length = vocab_data['max_length']