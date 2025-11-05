"""
Integration tests for the training pipeline.

These tests verify the complete training workflow including
model training, checkpointing, and early stopping functionality.
"""

import os
import tempfile
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytest
import numpy as np

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.lstm_model import LSTMClassifier
from training.trainer import Trainer, create_trainer
from training.checkpoint_manager import CheckpointManager, EarlyStopping
from training.glove_loader import GloVeLoader, initialize_model_with_glove
from data.text_preprocessor import TextPreprocessor


class TestTrainingIntegration:
    """Integration tests for training pipeline."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        # Create synthetic data
        vocab_size = 1000
        seq_length = 50
        batch_size = 32
        num_samples = 200
        
        # Generate random sequences
        X = torch.randint(1, vocab_size, (num_samples, seq_length))
        # Generate binary labels
        y = torch.randint(0, 2, (num_samples,))
        
        # Create datasets
        dataset = TensorDataset(X, y)
        
        # Split into train/val
        train_size = int(0.8 * num_samples)
        val_size = num_samples - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, vocab_size
    
    @pytest.fixture
    def sample_model(self, sample_data):
        """Create sample LSTM model."""
        _, _, vocab_size = sample_data
        
        model = LSTMClassifier(
            vocab_size=vocab_size,
            embedding_dim=100,
            hidden_dim=64,
            output_dim=1,
            n_layers=2,
            dropout=0.3,
            bidirectional=True
        )
        
        return model
    
    def test_basic_training_workflow(self, sample_data, sample_model, temp_dir):
        """Test basic training workflow with synthetic data."""
        train_loader, val_loader, _ = sample_data
        
        # Create trainer
        trainer = create_trainer(
            model=sample_model,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=0.01,
            checkpoint_dir=temp_dir
        )
        
        # Train for a few epochs
        history = trainer.train(epochs=3, save_best=True)
        
        # Verify training completed
        assert trainer.current_epoch == 3
        assert len(history['train_losses']) == 3
        assert len(history['val_losses']) == 3
        assert len(history['val_accuracies']) == 3
        
        # Verify checkpoints were created
        checkpoint_files = [f for f in os.listdir(temp_dir) if f.endswith('.pth')]
        assert len(checkpoint_files) > 0
    
    def test_model_convergence_synthetic_data(self, temp_dir):
        """Test model convergence on perfectly separable synthetic data."""
        # Create perfectly separable data
        vocab_size = 100
        seq_length = 10
        num_samples = 200
        
        # Create data where sequences starting with token 1 are positive,
        # sequences starting with token 2 are negative
        X_pos = torch.ones(num_samples // 2, seq_length, dtype=torch.long)
        X_neg = torch.full((num_samples // 2, seq_length), 2, dtype=torch.long)
        
        X = torch.cat([X_pos, X_neg], dim=0)
        y = torch.cat([
            torch.ones(num_samples // 2),
            torch.zeros(num_samples // 2)
        ])
        
        # Shuffle data
        indices = torch.randperm(num_samples)
        X, y = X[indices], y[indices]
        
        # Create data loaders
        dataset = TensorDataset(X, y)
        train_size = int(0.8 * num_samples)
        val_size = num_samples - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        # Create simple model
        model = LSTMClassifier(
            vocab_size=vocab_size,
            embedding_dim=32,
            hidden_dim=32,
            output_dim=1,
            n_layers=1,
            dropout=0.1,
            bidirectional=False
        )
        
        # Create trainer
        trainer = create_trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=0.01,
            checkpoint_dir=temp_dir
        )
        
        # Train until convergence
        history = trainer.train(epochs=20, early_stopping_patience=5)
        
        # Verify convergence (should achieve high accuracy on this simple task)
        final_accuracy = history['val_accuracies'][-1]
        assert final_accuracy > 80.0, f"Model should converge to >80% accuracy, got {final_accuracy}%"
        
        # Verify loss decreased
        initial_loss = history['train_losses'][0]
        final_loss = history['train_losses'][-1]
        assert final_loss < initial_loss, "Training loss should decrease"
    
    def test_checkpoint_saving_and_loading(self, sample_data, sample_model, temp_dir):
        """Test checkpoint saving and loading functionality."""
        train_loader, val_loader, _ = sample_data
        
        # Create trainer
        trainer = create_trainer(
            model=sample_model,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_dir=temp_dir
        )
        
        # Train for a few epochs
        trainer.train(epochs=2, save_best=True, save_every=1)
        
        # Get model state before loading
        original_state = sample_model.state_dict()
        
        # Find checkpoint file
        checkpoint_files = [f for f in os.listdir(temp_dir) if f.startswith('checkpoint_')]
        assert len(checkpoint_files) > 0
        
        checkpoint_path = os.path.join(temp_dir, checkpoint_files[0])
        
        # Create new model and trainer
        new_model = LSTMClassifier(
            vocab_size=sample_model.vocab_size,
            embedding_dim=sample_model.embedding_dim,
            hidden_dim=sample_model.hidden_dim,
            output_dim=sample_model.output_dim,
            n_layers=sample_model.n_layers,
            dropout=sample_model.dropout,
            bidirectional=sample_model.bidirectional
        )
        
        new_trainer = create_trainer(
            model=new_model,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_dir=temp_dir
        )
        
        # Load checkpoint
        checkpoint_info = new_trainer.load_checkpoint(checkpoint_path)
        
        # Verify checkpoint was loaded correctly
        assert checkpoint_info['epoch'] > 0
        assert 'model_state_dict' in checkpoint_info
        
        # Verify model states are different (new model should have loaded weights)
        loaded_state = new_model.state_dict()
        
        # At least one parameter should be different from original random initialization
        params_different = False
        for key in original_state:
            if not torch.equal(original_state[key], loaded_state[key]):
                params_different = True
                break
        
        assert params_different, "Loaded model should have different weights than random initialization"
    
    def test_early_stopping_functionality(self, temp_dir):
        """Test early stopping with non-improving validation loss."""
        # Create data that won't improve (random labels)
        vocab_size = 100
        seq_length = 20
        num_samples = 200
        
        X = torch.randint(1, vocab_size, (num_samples, seq_length))
        y = torch.randint(0, 2, (num_samples,))  # Random labels - won't improve
        
        dataset = TensorDataset(X, y)
        train_size = int(0.8 * num_samples)
        val_size = num_samples - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        # Create model
        model = LSTMClassifier(
            vocab_size=vocab_size,
            embedding_dim=32,
            hidden_dim=32,
            output_dim=1,
            n_layers=1,
            dropout=0.1
        )
        
        # Create trainer
        trainer = create_trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_dir=temp_dir
        )
        
        # Train with early stopping
        history = trainer.train(epochs=20, early_stopping_patience=3)
        
        # Should stop early due to no improvement
        assert history['total_epochs'] < 20, "Training should stop early due to no improvement"
        assert history['total_epochs'] >= 3, "Should train for at least patience epochs"
    
    def test_checkpoint_manager_integration(self, sample_data, sample_model, temp_dir):
        """Test integration with checkpoint manager."""
        train_loader, val_loader, _ = sample_data
        
        # Create checkpoint manager
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=temp_dir,
            max_checkpoints=3,
            save_best_n=2
        )
        
        # Create optimizer for checkpoint manager
        optimizer = torch.optim.Adam(sample_model.parameters(), lr=0.001)
        
        # Simulate training with checkpoint manager
        for epoch in range(5):
            # Simulate training metrics
            train_loss = 1.0 - epoch * 0.1  # Decreasing loss
            val_loss = 0.8 - epoch * 0.05   # Decreasing validation loss
            val_accuracy = 50.0 + epoch * 5.0  # Increasing accuracy
            
            is_best = epoch >= 2  # Last few epochs are "best"
            
            checkpoint_path = checkpoint_manager.save_checkpoint(
                model=sample_model,
                optimizer=optimizer,
                epoch=epoch + 1,
                train_loss=train_loss,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
                is_best=is_best
            )
            
            assert os.path.exists(checkpoint_path)
        
        # Verify checkpoint management
        checkpoints = checkpoint_manager.list_checkpoints()
        assert len(checkpoints) > 0
        
        # Should have limited number of regular checkpoints
        regular_checkpoints = [c for c in checkpoints if c['type'] == 'regular']
        assert len(regular_checkpoints) <= 3
        
        # Should have best model checkpoints
        best_checkpoints = [c for c in checkpoints if c['type'] == 'best']
        assert len(best_checkpoints) > 0
        
        # Test loading best checkpoint
        best_checkpoint_path = checkpoint_manager.load_best_checkpoint('loss')
        assert best_checkpoint_path is not None
        assert os.path.exists(best_checkpoint_path)
    
    def test_training_with_scheduler(self, sample_data, sample_model, temp_dir):
        """Test training with learning rate scheduler."""
        train_loader, val_loader, _ = sample_data
        
        # Create trainer with scheduler
        trainer = create_trainer(
            model=sample_model,
            train_loader=train_loader,
            val_loader=val_loader,
            scheduler_type='plateau',
            checkpoint_dir=temp_dir
        )
        
        # Train for several epochs
        history = trainer.train(epochs=5)
        
        # Verify learning rates were recorded
        assert len(history['learning_rates']) == 5
        
        # Learning rates should be recorded for each epoch
        for lr in history['learning_rates']:
            assert isinstance(lr, float)
            assert lr > 0
    
    def test_training_summary_generation(self, sample_data, sample_model, temp_dir):
        """Test training summary generation."""
        train_loader, val_loader, _ = sample_data
        
        trainer = create_trainer(
            model=sample_model,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_dir=temp_dir
        )
        
        # Train model
        trainer.train(epochs=3)
        
        # Get training summary
        summary = trainer.get_training_summary()
        
        # Verify summary structure
        assert 'training_status' in summary
        assert 'performance_metrics' in summary
        assert 'training_progress' in summary
        assert 'model_info' in summary
        assert 'training_config' in summary
        
        # Verify specific fields
        assert summary['training_status']['epochs_completed'] == 3
        assert summary['performance_metrics']['best_val_loss'] is not None
        assert summary['model_info']['total_parameters'] > 0


class TestGloVeIntegration:
    """Tests for GloVe embedding integration."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_preprocessor(self):
        """Create sample text preprocessor with vocabulary."""
        preprocessor = TextPreprocessor(max_vocab_size=1000, min_freq=1)
        
        # Build vocabulary from sample texts
        sample_texts = [
            "this is a positive review",
            "this is a negative review",
            "great movie excellent acting",
            "terrible film bad plot",
            "good story nice characters"
        ]
        
        preprocessor.build_vocabulary(sample_texts)
        return preprocessor
    
    def test_glove_loader_mock_functionality(self, temp_dir, sample_preprocessor):
        """Test GloVe loader with mock embeddings (since we can't download in tests)."""
        # Create mock GloVe file
        glove_file = os.path.join(temp_dir, "mock_glove.txt")
        
        # Create mock embeddings for some vocabulary words
        mock_embeddings = {
            'this': [0.1, 0.2, 0.3],
            'is': [0.4, 0.5, 0.6],
            'good': [0.7, 0.8, 0.9],
            'bad': [-0.1, -0.2, -0.3]
        }
        
        with open(glove_file, 'w') as f:
            for word, vector in mock_embeddings.items():
                f.write(f"{word} {' '.join(map(str, vector))}\n")
        
        # Test loading
        glove_loader = GloVeLoader(cache_dir=temp_dir)
        embeddings, dim = glove_loader.load_glove_embeddings(glove_file)
        
        assert dim == 3
        assert len(embeddings) == 4
        assert 'this' in embeddings
        assert np.allclose(embeddings['this'], [0.1, 0.2, 0.3])
        
        # Test embedding matrix creation
        embedding_matrix, stats = glove_loader.create_embedding_matrix(
            sample_preprocessor, embeddings, dim
        )
        
        assert embedding_matrix.shape[0] == sample_preprocessor.vocab_size
        assert embedding_matrix.shape[1] == dim
        assert stats['found_in_glove'] > 0
        assert stats['coverage_ratio'] > 0


if __name__ == '__main__':
    # Run tests if executed directly
    pytest.main([__file__, '-v'])