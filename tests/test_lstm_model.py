"""
Unit tests for LSTM model architecture.

Tests model initialization, forward pass output shapes and value ranges,
and gradient flow through all layers as specified in requirements 3.1, 3.2, 3.3, 3.4, 3.5.
"""

import unittest
import torch
import torch.nn as nn
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.lstm_model import LSTMClassifier


class TestLSTMClassifier(unittest.TestCase):
    """Test cases for LSTMClassifier model architecture."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.default_config = {
            'vocab_size': 1000,
            'embedding_dim': 100,
            'hidden_dim': 64,
            'output_dim': 1,
            'n_layers': 2,
            'dropout': 0.3,
            'bidirectional': True,
            'pad_idx': 0
        }
        self.model = LSTMClassifier(**self.default_config)
        
        # Sample input data
        self.batch_size = 4
        self.seq_length = 10
        self.sample_input = torch.randint(0, 1000, (self.batch_size, self.seq_length))
    
    def test_model_initialization_default_parameters(self):
        """Test model initialization with default hyperparameters."""
        model = LSTMClassifier(vocab_size=1000)
        
        # Check default parameter values
        self.assertEqual(model.vocab_size, 1000)
        self.assertEqual(model.embedding_dim, 300)
        self.assertEqual(model.hidden_dim, 128)
        self.assertEqual(model.output_dim, 1)
        self.assertEqual(model.n_layers, 2)
        self.assertEqual(model.dropout, 0.3)
        self.assertTrue(model.bidirectional)
        self.assertEqual(model.pad_idx, 0)
        
        # Check calculated dimensions
        self.assertEqual(model.lstm_output_dim, 256)  # 128 * 2 for bidirectional
        
        # Check layer initialization
        self.assertIsInstance(model.embedding, nn.Embedding)
        self.assertIsInstance(model.lstm, nn.LSTM)
        self.assertIsInstance(model.fc1, nn.Linear)
        self.assertIsInstance(model.fc2, nn.Linear)
    
    def test_model_initialization_custom_parameters(self):
        """Test model initialization with different hyperparameters."""
        configs = [
            # Small model
            {
                'vocab_size': 500,
                'embedding_dim': 50,
                'hidden_dim': 32,
                'output_dim': 1,
                'n_layers': 1,
                'dropout': 0.1,
                'bidirectional': False
            },
            # Large model
            {
                'vocab_size': 5000,
                'embedding_dim': 300,
                'hidden_dim': 256,
                'output_dim': 1,
                'n_layers': 3,
                'dropout': 0.5,
                'bidirectional': True
            },
            # Multi-class model
            {
                'vocab_size': 2000,
                'embedding_dim': 200,
                'hidden_dim': 100,
                'output_dim': 5,
                'n_layers': 2,
                'dropout': 0.2,
                'bidirectional': True
            }
        ]
        
        for config in configs:
            with self.subTest(config=config):
                model = LSTMClassifier(**config)
                
                # Check all parameters are set correctly
                self.assertEqual(model.vocab_size, config['vocab_size'])
                self.assertEqual(model.embedding_dim, config['embedding_dim'])
                self.assertEqual(model.hidden_dim, config['hidden_dim'])
                self.assertEqual(model.output_dim, config['output_dim'])
                self.assertEqual(model.n_layers, config['n_layers'])
                self.assertEqual(model.dropout, config['dropout'])
                self.assertEqual(model.bidirectional, config['bidirectional'])
                
                # Check embedding layer dimensions
                self.assertEqual(model.embedding.num_embeddings, config['vocab_size'])
                self.assertEqual(model.embedding.embedding_dim, config['embedding_dim'])
                
                # Check LSTM layer configuration
                self.assertEqual(model.lstm.input_size, config['embedding_dim'])
                self.assertEqual(model.lstm.hidden_size, config['hidden_dim'])
                self.assertEqual(model.lstm.num_layers, config['n_layers'])
                self.assertEqual(model.lstm.bidirectional, config['bidirectional'])
                
                # Check fully connected layer dimensions
                expected_lstm_output = config['hidden_dim'] * (2 if config['bidirectional'] else 1)
                self.assertEqual(model.fc1.in_features, expected_lstm_output)
                self.assertEqual(model.fc1.out_features, 64)
                self.assertEqual(model.fc2.in_features, 64)
                self.assertEqual(model.fc2.out_features, config['output_dim'])
    
    def test_forward_pass_output_shapes(self):
        """Test forward pass output shapes with different input configurations."""
        test_cases = [
            # Different batch sizes
            {'batch_size': 1, 'seq_length': 10},
            {'batch_size': 8, 'seq_length': 10},
            {'batch_size': 32, 'seq_length': 10},
            
            # Different sequence lengths
            {'batch_size': 4, 'seq_length': 5},
            {'batch_size': 4, 'seq_length': 20},
            {'batch_size': 4, 'seq_length': 100},
        ]
        
        for case in test_cases:
            with self.subTest(case=case):
                input_tensor = torch.randint(0, 1000, (case['batch_size'], case['seq_length']))
                
                # Forward pass
                output = self.model(input_tensor)
                
                # Check output shape
                expected_shape = (case['batch_size'], self.default_config['output_dim'])
                self.assertEqual(output.shape, expected_shape)
                
                # Check output tensor properties
                self.assertEqual(output.dtype, torch.float32)
                self.assertFalse(torch.isnan(output).any())
                self.assertFalse(torch.isinf(output).any())
    
    def test_forward_pass_value_ranges(self):
        """Test forward pass output value ranges are reasonable."""
        # Test with different models
        models = [
            LSTMClassifier(vocab_size=1000, dropout=0.0),  # No dropout
            LSTMClassifier(vocab_size=1000, dropout=0.5),  # High dropout
            LSTMClassifier(vocab_size=1000, bidirectional=False),  # Unidirectional
        ]
        
        for i, model in enumerate(models):
            with self.subTest(model_idx=i):
                model.eval()  # Set to evaluation mode
                
                with torch.no_grad():
                    output = model(self.sample_input)
                    
                    # Check output is finite
                    self.assertTrue(torch.isfinite(output).all())
                    
                    # For binary classification, raw logits should be reasonable
                    # (not extremely large or small)
                    self.assertTrue(torch.abs(output).max() < 50.0)
                    
                    # Check that different inputs produce different outputs
                    different_input = torch.randint(0, 1000, self.sample_input.shape)
                    different_output = model(different_input)
                    
                    # Outputs should be different (with high probability)
                    self.assertFalse(torch.allclose(output, different_output, atol=1e-6))
    
    def test_predict_proba_output_ranges(self):
        """Test predict_proba method outputs valid probabilities."""
        self.model.eval()
        
        with torch.no_grad():
            probabilities = self.model.predict_proba(self.sample_input)
            
            # Check shape
            expected_shape = (self.batch_size, self.default_config['output_dim'])
            self.assertEqual(probabilities.shape, expected_shape)
            
            # Check probability ranges [0, 1]
            self.assertTrue((probabilities >= 0.0).all())
            self.assertTrue((probabilities <= 1.0).all())
            
            # Check no NaN or inf values
            self.assertFalse(torch.isnan(probabilities).any())
            self.assertFalse(torch.isinf(probabilities).any())
    
    def test_predict_binary_output(self):
        """Test predict method outputs binary predictions."""
        self.model.eval()
        
        with torch.no_grad():
            predictions = self.model.predict(self.sample_input)
            
            # Check shape
            expected_shape = (self.batch_size, self.default_config['output_dim'])
            self.assertEqual(predictions.shape, expected_shape)
            
            # Check binary values (0 or 1)
            unique_values = torch.unique(predictions)
            self.assertTrue(len(unique_values) <= 2)
            self.assertTrue(all(val in [0.0, 1.0] for val in unique_values.tolist()))
    
    def test_gradient_flow_through_all_layers(self):
        """Test gradient flow through all layers during backpropagation."""
        self.model.train()
        
        # Create dummy target
        target = torch.randint(0, 2, (self.batch_size, 1)).float()
        
        # Forward pass
        output = self.model(self.sample_input)
        
        # Calculate loss
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check that all parameters have gradients
        for name, param in self.model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(param.grad, f"No gradient for parameter: {name}")
                self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradient for parameter: {name}")
                self.assertFalse(torch.isinf(param.grad).any(), f"Inf gradient for parameter: {name}")
                
                # Check that gradients are not all zeros (indicating gradient flow)
                if param.requires_grad:
                    grad_norm = param.grad.norm().item()
                    self.assertGreater(grad_norm, 0.0, f"Zero gradient for parameter: {name}")
    
    def test_gradient_flow_with_different_loss_functions(self):
        """Test gradient flow with different loss functions."""
        loss_functions = [
            nn.BCEWithLogitsLoss(),
            nn.MSELoss(),
        ]
        
        for loss_fn in loss_functions:
            with self.subTest(loss_function=type(loss_fn).__name__):
                # Fresh model for each test
                model = LSTMClassifier(**self.default_config)
                model.train()
                
                # Create appropriate target based on loss function
                if isinstance(loss_fn, nn.BCEWithLogitsLoss):
                    target = torch.randint(0, 2, (self.batch_size, 1)).float()
                else:  # MSELoss
                    target = torch.randn(self.batch_size, 1)
                
                # Forward and backward pass
                output = model(self.sample_input)
                loss = loss_fn(output, target)
                loss.backward()
                
                # Check gradient flow
                gradient_norms = []
                for param in model.parameters():
                    if param.requires_grad and param.grad is not None:
                        gradient_norms.append(param.grad.norm().item())
                
                # At least some gradients should be non-zero
                self.assertTrue(any(norm > 0 for norm in gradient_norms))
    
    def test_hidden_state_initialization(self):
        """Test hidden state initialization for different configurations."""
        test_configs = [
            {'n_layers': 1, 'bidirectional': False},
            {'n_layers': 2, 'bidirectional': False},
            {'n_layers': 1, 'bidirectional': True},
            {'n_layers': 3, 'bidirectional': True},
        ]
        
        for config in test_configs:
            with self.subTest(config=config):
                model = LSTMClassifier(
                    vocab_size=1000,
                    hidden_dim=64,
                    n_layers=config['n_layers'],
                    bidirectional=config['bidirectional']
                )
                
                batch_size = 8
                device = torch.device('cpu')
                
                hidden, cell = model.init_hidden(batch_size, device)
                
                # Calculate expected dimensions
                num_directions = 2 if config['bidirectional'] else 1
                expected_layers = config['n_layers'] * num_directions
                
                # Check shapes
                expected_shape = (expected_layers, batch_size, 64)
                self.assertEqual(hidden.shape, expected_shape)
                self.assertEqual(cell.shape, expected_shape)
                
                # Check initialization (should be zeros)
                self.assertTrue(torch.allclose(hidden, torch.zeros_like(hidden)))
                self.assertTrue(torch.allclose(cell, torch.zeros_like(cell)))
                
                # Check device placement
                self.assertEqual(hidden.device, device)
                self.assertEqual(cell.device, device)
    
    def test_pretrained_embeddings_loading(self):
        """Test loading pre-trained embeddings."""
        vocab_size = 1000
        embedding_dim = 100
        model = LSTMClassifier(vocab_size=vocab_size, embedding_dim=embedding_dim)
        
        # Create fake pre-trained embeddings
        pretrained_embeddings = torch.randn(vocab_size, embedding_dim)
        original_embeddings = pretrained_embeddings.clone()
        
        # Load embeddings
        model.load_pretrained_embeddings(pretrained_embeddings)
        
        # Check that non-padding embeddings were loaded correctly
        for i in range(vocab_size):
            if i != model.pad_idx:
                self.assertTrue(torch.allclose(
                    model.embedding.weight.data[i],
                    original_embeddings[i],
                    atol=1e-6
                ))
        
        # Check that padding token embedding is zeroed
        if model.pad_idx is not None:
            expected_pad_embedding = torch.zeros(embedding_dim)
            self.assertTrue(torch.allclose(
                model.embedding.weight.data[model.pad_idx],
                expected_pad_embedding
            ))
    
    def test_pretrained_embeddings_shape_validation(self):
        """Test that loading embeddings with wrong shape raises error."""
        model = LSTMClassifier(vocab_size=1000, embedding_dim=100)
        
        # Wrong vocabulary size
        wrong_vocab_embeddings = torch.randn(500, 100)
        with self.assertRaises(ValueError):
            model.load_pretrained_embeddings(wrong_vocab_embeddings)
        
        # Wrong embedding dimension
        wrong_dim_embeddings = torch.randn(1000, 200)
        with self.assertRaises(ValueError):
            model.load_pretrained_embeddings(wrong_dim_embeddings)
    
    def test_model_info_method(self):
        """Test get_model_info method returns correct information."""
        info = self.model.get_model_info()
        
        # Check that all expected keys are present
        expected_keys = [
            'vocab_size', 'embedding_dim', 'hidden_dim', 'output_dim',
            'n_layers', 'dropout', 'bidirectional', 'lstm_output_dim',
            'total_parameters', 'trainable_parameters'
        ]
        
        for key in expected_keys:
            self.assertIn(key, info)
        
        # Check values match model configuration
        self.assertEqual(info['vocab_size'], self.default_config['vocab_size'])
        self.assertEqual(info['embedding_dim'], self.default_config['embedding_dim'])
        self.assertEqual(info['hidden_dim'], self.default_config['hidden_dim'])
        self.assertEqual(info['output_dim'], self.default_config['output_dim'])
        self.assertEqual(info['n_layers'], self.default_config['n_layers'])
        self.assertEqual(info['dropout'], self.default_config['dropout'])
        self.assertEqual(info['bidirectional'], self.default_config['bidirectional'])
        
        # Check parameter counts are positive integers
        self.assertIsInstance(info['total_parameters'], int)
        self.assertIsInstance(info['trainable_parameters'], int)
        self.assertGreater(info['total_parameters'], 0)
        self.assertGreater(info['trainable_parameters'], 0)
        self.assertEqual(info['total_parameters'], info['trainable_parameters'])  # All params should be trainable
    
    def test_model_modes_training_vs_eval(self):
        """Test model behavior in training vs evaluation modes."""
        # Test in training mode
        self.model.train()
        self.assertTrue(self.model.training)
        
        output_train_1 = self.model(self.sample_input)
        output_train_2 = self.model(self.sample_input)
        
        # In training mode with dropout, outputs should be different
        self.assertFalse(torch.allclose(output_train_1, output_train_2, atol=1e-6))
        
        # Test in evaluation mode
        self.model.eval()
        self.assertFalse(self.model.training)
        
        output_eval_1 = self.model(self.sample_input)
        output_eval_2 = self.model(self.sample_input)
        
        # In evaluation mode, outputs should be identical
        self.assertTrue(torch.allclose(output_eval_1, output_eval_2))
    
    def test_weight_initialization(self):
        """Test that model weights are properly initialized."""
        model = LSTMClassifier(vocab_size=1000, embedding_dim=100, hidden_dim=64)
        
        # Check that weights are not all zeros or ones
        for name, param in model.named_parameters():
            if 'weight' in name:
                # Weights should not be all zeros
                self.assertFalse(torch.allclose(param, torch.zeros_like(param)))
                
                # Weights should have reasonable variance (not all the same value)
                if param.numel() > 1:
                    self.assertGreater(param.var().item(), 1e-6)
            
            elif 'bias' in name:
                # Biases should be initialized to zero
                self.assertTrue(torch.allclose(param, torch.zeros_like(param)))


if __name__ == '__main__':
    unittest.main()