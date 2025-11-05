"""
Unit tests for text preprocessing components.

Tests tokenization accuracy, vocabulary building, and padding/truncation behavior
as specified in requirements 2.1, 2.2, and 2.3.
"""

import unittest
import torch
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.text_preprocessor import TextPreprocessor


class TestTextPreprocessor(unittest.TestCase):
    """Test cases for TextPreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.preprocessor = TextPreprocessor(max_vocab_size=100, min_freq=1, max_length=10)
    
    def test_tokenization_accuracy(self):
        """Test tokenization accuracy on sample texts."""
        # Test basic tokenization
        text = "This is a simple test."
        expected_tokens = ["this", "is", "a", "simple", "test", "."]
        actual_tokens = self.preprocessor.tokenize(text)
        self.assertEqual(actual_tokens, expected_tokens)
        
        # Test contractions handling
        text = "I can't believe it won't work."
        expected_tokens = ["i", "cannot", "believe", "it", "will", "not", "work", "."]
        actual_tokens = self.preprocessor.tokenize(text)
        self.assertEqual(actual_tokens, expected_tokens)
        
        # Test punctuation separation
        text = "Hello, world! How are you?"
        expected_tokens = ["hello", ",", "world", "!", "how", "are", "you", "?"]
        actual_tokens = self.preprocessor.tokenize(text)
        self.assertEqual(actual_tokens, expected_tokens)
        
        # Test empty and None inputs
        self.assertEqual(self.preprocessor.tokenize(""), [])
        self.assertEqual(self.preprocessor.tokenize("   "), [])
        self.assertEqual(self.preprocessor.tokenize(None), [])
        
        # Test mixed case and whitespace
        text = "  UPPER lower   MiXeD  "
        expected_tokens = ["upper", "lower", "mixed"]
        actual_tokens = self.preprocessor.tokenize(text)
        self.assertEqual(actual_tokens, expected_tokens)
    
    def test_vocabulary_building_with_known_frequencies(self):
        """Test vocabulary building with known word frequencies."""
        # Sample texts with known word frequencies
        texts = [
            "the cat sat on the mat",
            "the dog ran in the park", 
            "a cat and a dog played",
            "the park was beautiful"
        ]
        
        # Build vocabulary
        word_freq = self.preprocessor.build_vocabulary(texts)
        
        # Check that high-frequency words are included
        self.assertIn("the", self.preprocessor.word_to_idx)
        self.assertIn("cat", self.preprocessor.word_to_idx)
        self.assertIn("park", self.preprocessor.word_to_idx)
        
        # Verify frequency counts in returned dictionary
        # "the" appears 5 times total: 2 in first text, 2 in second, 1 in fourth
        self.assertEqual(word_freq["the"], 5)
        self.assertEqual(word_freq["cat"], 2)
        self.assertEqual(word_freq["park"], 2)
        
        # Check vocabulary size includes special tokens
        self.assertGreaterEqual(self.preprocessor.vocab_size, 4)  # At least special tokens
        
        # Verify special tokens are present
        self.assertIn("<PAD>", self.preprocessor.word_to_idx)
        self.assertIn("<UNK>", self.preprocessor.word_to_idx)
        self.assertIn("<START>", self.preprocessor.word_to_idx)
        self.assertIn("<END>", self.preprocessor.word_to_idx)
    
    def test_vocabulary_frequency_threshold(self):
        """Test vocabulary building respects minimum frequency threshold."""
        # Create preprocessor with higher minimum frequency
        preprocessor = TextPreprocessor(max_vocab_size=100, min_freq=2, max_length=10)
        
        texts = [
            "common word appears twice",
            "common word appears again", 
            "rare single occurrence"
        ]
        
        word_freq = preprocessor.build_vocabulary(texts)
        
        # Words appearing twice should be included
        self.assertIn("common", preprocessor.word_to_idx)
        self.assertIn("word", preprocessor.word_to_idx)
        self.assertIn("appears", preprocessor.word_to_idx)
        
        # Words appearing once should be excluded
        self.assertNotIn("rare", preprocessor.word_to_idx)
        self.assertNotIn("single", preprocessor.word_to_idx)
        self.assertNotIn("occurrence", preprocessor.word_to_idx)
    
    def test_vocabulary_size_limit(self):
        """Test vocabulary building respects maximum vocabulary size."""
        # Create preprocessor with small vocabulary limit
        preprocessor = TextPreprocessor(max_vocab_size=8, min_freq=1, max_length=10)
        
        texts = [
            "word1 word2 word3 word4 word5",
            "word6 word7 word8 word9 word10"
        ]
        
        preprocessor.build_vocabulary(texts)
        
        # Should not exceed max vocabulary size
        self.assertLessEqual(preprocessor.vocab_size, 8)
        
        # Should include special tokens (4) plus some regular words
        self.assertEqual(preprocessor.vocab_size, 8)
    
    def test_text_to_sequence_conversion(self):
        """Test text to sequence conversion using vocabulary mapping."""
        # Build vocabulary first
        texts = ["hello world", "hello there"]
        self.preprocessor.build_vocabulary(texts)
        
        # Test conversion
        text = "hello world"
        sequence = self.preprocessor.text_to_sequence(text)
        
        # Should return tensor
        self.assertIsInstance(sequence, torch.Tensor)
        self.assertEqual(sequence.dtype, torch.long)
        
        # Should have correct length
        self.assertEqual(len(sequence), 2)  # "hello" and "world"
        
        # Test unknown word handling
        text_with_unk = "hello unknown"
        sequence_unk = self.preprocessor.text_to_sequence(text_with_unk)
        
        # Unknown word should map to UNK token
        unk_idx = self.preprocessor.word_to_idx["<UNK>"]
        self.assertEqual(sequence_unk[1].item(), unk_idx)
    
    def test_padding_behavior(self):
        """Test padding behavior for sequences of different lengths."""
        # Create sequences of different lengths
        seq1 = torch.tensor([1, 2, 3])
        seq2 = torch.tensor([4, 5])
        seq3 = torch.tensor([6, 7, 8, 9, 10])
        
        sequences = [seq1, seq2, seq3]
        
        # Test padding to default max_length (10)
        padded = self.preprocessor.pad_sequences(sequences)
        
        # Check output shape
        self.assertEqual(padded.shape, (3, 10))
        
        # Check padding values
        pad_idx = self.preprocessor.word_to_idx["<PAD>"]
        
        # First sequence: [1, 2, 3, pad, pad, pad, pad, pad, pad, pad]
        expected_seq1 = torch.tensor([1, 2, 3] + [pad_idx] * 7)
        self.assertTrue(torch.equal(padded[0], expected_seq1))
        
        # Second sequence: [4, 5, pad, pad, pad, pad, pad, pad, pad, pad]
        expected_seq2 = torch.tensor([4, 5] + [pad_idx] * 8)
        self.assertTrue(torch.equal(padded[1], expected_seq2))
        
        # Third sequence: [6, 7, 8, 9, 10, pad, pad, pad, pad, pad]
        expected_seq3 = torch.tensor([6, 7, 8, 9, 10] + [pad_idx] * 5)
        self.assertTrue(torch.equal(padded[2], expected_seq3))
    
    def test_padding_with_custom_length(self):
        """Test padding with custom maximum length."""
        seq1 = torch.tensor([1, 2, 3])
        seq2 = torch.tensor([4, 5])
        
        sequences = [seq1, seq2]
        
        # Test padding to custom length
        padded = self.preprocessor.pad_sequences(sequences, max_length=5)
        
        # Check output shape
        self.assertEqual(padded.shape, (2, 5))
        
        # Check padding values
        pad_idx = self.preprocessor.word_to_idx["<PAD>"]
        expected_seq1 = torch.tensor([1, 2, 3, pad_idx, pad_idx])
        expected_seq2 = torch.tensor([4, 5, pad_idx, pad_idx, pad_idx])
        
        self.assertTrue(torch.equal(padded[0], expected_seq1))
        self.assertTrue(torch.equal(padded[1], expected_seq2))
    
    def test_truncation_behavior(self):
        """Test truncation behavior for overly long sequences."""
        # Create sequences longer than max_length
        long_seq1 = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        long_seq2 = torch.tensor([13, 14, 15, 16, 17, 18, 19, 20])
        
        sequences = [long_seq1, long_seq2]
        
        # Test padding with truncation (max_length = 10)
        padded = self.preprocessor.pad_sequences(sequences)
        
        # Check output shape
        self.assertEqual(padded.shape, (2, 10))
        
        # First sequence should be truncated to first 10 elements
        expected_seq1 = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.assertTrue(torch.equal(padded[0], expected_seq1))
        
        # Second sequence should be padded (length < max_length)
        pad_idx = self.preprocessor.word_to_idx["<PAD>"]
        expected_seq2 = torch.tensor([13, 14, 15, 16, 17, 18, 19, 20, pad_idx, pad_idx])
        self.assertTrue(torch.equal(padded[1], expected_seq2))
    
    def test_truncate_sequence_method(self):
        """Test the truncate_sequence method specifically."""
        # Test sequence longer than max_length
        long_seq = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        truncated = self.preprocessor.truncate_sequence(long_seq)
        
        # Should be truncated to max_length (10)
        expected = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.assertTrue(torch.equal(truncated, expected))
        
        # Test sequence shorter than max_length
        short_seq = torch.tensor([1, 2, 3])
        truncated_short = self.preprocessor.truncate_sequence(short_seq)
        
        # Should remain unchanged
        self.assertTrue(torch.equal(truncated_short, short_seq))
        
        # Test with custom max_length
        custom_truncated = self.preprocessor.truncate_sequence(long_seq, max_length=5)
        expected_custom = torch.tensor([1, 2, 3, 4, 5])
        self.assertTrue(torch.equal(custom_truncated, expected_custom))
    
    def test_empty_sequences_handling(self):
        """Test handling of empty sequences."""
        # Test empty list
        empty_padded = self.preprocessor.pad_sequences([])
        self.assertEqual(empty_padded.shape, (0,))
        
        # Test list with empty tensor
        empty_seq = torch.tensor([], dtype=torch.long)
        sequences = [empty_seq]
        padded = self.preprocessor.pad_sequences(sequences)
        
        # Should create padded sequence of all PAD tokens
        pad_idx = self.preprocessor.word_to_idx["<PAD>"]
        expected = torch.full((1, 10), pad_idx, dtype=torch.long)
        self.assertTrue(torch.equal(padded, expected))
    
    def test_special_tokens_initialization(self):
        """Test that special tokens are properly initialized."""
        # Check that all special tokens exist
        special_tokens = ["<PAD>", "<UNK>", "<START>", "<END>"]
        
        for token in special_tokens:
            self.assertIn(token, self.preprocessor.word_to_idx)
            idx = self.preprocessor.word_to_idx[token]
            self.assertEqual(self.preprocessor.idx_to_word[idx], token)
        
        # Check that special tokens have expected indices (0, 1, 2, 3)
        self.assertEqual(self.preprocessor.word_to_idx["<PAD>"], 0)
        self.assertEqual(self.preprocessor.word_to_idx["<UNK>"], 1)
        self.assertEqual(self.preprocessor.word_to_idx["<START>"], 2)
        self.assertEqual(self.preprocessor.word_to_idx["<END>"], 3)
    
    def test_preprocess_texts_pipeline(self):
        """Test the complete preprocessing pipeline."""
        texts = [
            "This is a positive review!",
            "This movie was terrible.",
            "Great film, loved it!"
        ]
        
        # Test with vocabulary fitting
        result = self.preprocessor.preprocess_texts(texts, fit_vocabulary=True)
        
        # Should return padded tensor
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape[0], 3)  # 3 texts
        self.assertEqual(result.shape[1], 10)  # max_length
        
        # Test without vocabulary fitting (should work with existing vocab)
        new_texts = ["This is another review"]
        result2 = self.preprocessor.preprocess_texts(new_texts, fit_vocabulary=False)
        
        self.assertIsInstance(result2, torch.Tensor)
        self.assertEqual(result2.shape[0], 1)
        self.assertEqual(result2.shape[1], 10)


if __name__ == '__main__':
    unittest.main()