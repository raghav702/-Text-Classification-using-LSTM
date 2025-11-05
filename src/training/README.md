# LSTM Sentiment Classifier - Training Pipeline

This module provides a comprehensive training pipeline for the LSTM sentiment classifier with advanced features including GloVe embeddings, checkpointing, early stopping, and training monitoring.

## Features

- **Complete Training Pipeline**: Full training workflow with validation and monitoring
- **GloVe Integration**: Automatic download and integration of pre-trained embeddings
- **Advanced Checkpointing**: Automatic checkpoint management with best model tracking
- **Early Stopping**: Configurable early stopping with multiple criteria
- **Learning Rate Scheduling**: Multiple scheduler options (plateau, step, cosine)
- **Gradient Clipping**: Prevents exploding gradients during training
- **Comprehensive Logging**: Detailed training progress and metrics logging
- **Integration Tests**: Full test suite for training pipeline validation

## Quick Start

### Basic Training

```python
from src.training import create_trainer
from src.models.lstm_model import LSTMClassifier

# Create model and data loaders (see data module)
model = LSTMClassifier(vocab_size=10000, embedding_dim=300)
# train_loader, val_loader = ... (from data module)

# Create trainer with default settings
trainer = create_trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader
)

# Train the model
history = trainer.train(epochs=20, early_stopping_patience=5)
```

### Training with GloVe Embeddings

```python
from src.training import initialize_model_with_glove

# Initialize model with GloVe embeddings
embedding_stats = initialize_model_with_glove(
    model=model,
    preprocessor=preprocessor,
    corpus='6B',
    dimension='300d',
    freeze_embeddings=False
)

# Then train as usual
trainer = create_trainer(model, train_loader, val_loader)
history = trainer.train(epochs=20)
```

### Command-Line Training

```bash
# Basic training
python src/training/train_model.py --data-dir data/imdb --epochs 20

# Training with GloVe embeddings
python src/training/train_model.py \
    --data-dir data/imdb \
    --use-glove \
    --glove-corpus 6B \
    --glove-dim 300d \
    --epochs 20 \
    --early-stopping-patience 5

# Advanced configuration
python src/training/train_model.py \
    --data-dir data/imdb \
    --batch-size 64 \
    --learning-rate 0.001 \
    --hidden-dim 128 \
    --n-layers 2 \
    --dropout 0.3 \
    --scheduler plateau \
    --gradient-clip 1.0 \
    --output-dir models \
    --checkpoint-dir checkpoints
```

## Module Components

### 1. Trainer (`trainer.py`)

The core training engine with comprehensive functionality:

```python
from src.training import Trainer, create_trainer

# Manual trainer creation
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    scheduler=scheduler,
    gradient_clip=1.0,
    checkpoint_dir="checkpoints"
)

# Factory function (recommended)
trainer = create_trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    learning_rate=0.001,
    scheduler_type='plateau'
)
```

#### Key Methods

- `train(epochs, early_stopping_patience, save_best, save_every)`: Main training loop
- `train_epoch()`: Single epoch training
- `evaluate_model(data_loader)`: Model evaluation
- `save_checkpoint(filename, is_best, metrics)`: Save training state
- `load_checkpoint(path, load_optimizer)`: Load training state
- `resume_training(checkpoint_path, additional_epochs)`: Resume from checkpoint
- `get_training_summary()`: Get comprehensive training statistics

### 2. GloVe Loader (`glove_loader.py`)

Handles GloVe embedding download, loading, and integration:

```python
from src.training import GloVeLoader, initialize_model_with_glove

# Manual GloVe handling
glove_loader = GloVeLoader(cache_dir="data/glove")
glove_path = glove_loader.download_glove('6B', '300d')
embeddings, dim = glove_loader.load_glove_embeddings(glove_path)

# Convenience function (recommended)
stats = initialize_model_with_glove(
    model=model,
    preprocessor=preprocessor,
    corpus='6B',
    dimension='300d'
)
```

#### Supported GloVe Variants

- **6B**: 6 billion tokens, 400K vocab (50d, 100d, 200d, 300d)
- **42B**: 42 billion tokens, 1.9M vocab (300d)
- **840B**: 840 billion tokens, 2.2M vocab (300d)

### 3. Checkpoint Manager (`checkpoint_manager.py`)

Advanced checkpoint management with automatic cleanup:

```python
from src.training import CheckpointManager, create_checkpoint_manager

# Create checkpoint manager
checkpoint_manager = create_checkpoint_manager(
    checkpoint_dir="checkpoints",
    max_checkpoints=5,
    save_best_n=3
)

# Save checkpoint
checkpoint_manager.save_checkpoint(
    model=model,
    optimizer=optimizer,
    epoch=epoch,
    train_loss=train_loss,
    val_loss=val_loss,
    val_accuracy=val_accuracy,
    is_best=True
)

# Load best checkpoint
best_path = checkpoint_manager.load_best_checkpoint('loss')
```

#### Features

- **Automatic Cleanup**: Keeps only N most recent checkpoints
- **Best Model Tracking**: Maintains separate best models by loss/accuracy
- **Registry Management**: JSON registry of best models
- **Checkpoint Information**: Detailed checkpoint metadata

### 4. Early Stopping (`checkpoint_manager.py`)

Configurable early stopping with multiple criteria:

```python
from src.training import EarlyStopping, create_early_stopping

# Create early stopping
early_stopping = create_early_stopping(
    patience=7,
    min_delta=0.001,
    monitor='val_loss',
    restore_best_weights=True
)

# Use in training loop
for epoch in range(epochs):
    # ... training code ...
    
    if early_stopping(val_loss, model, epoch):
        print(f"Early stopping at epoch {epoch}")
        break
```

## Training Configuration

### Model Configuration

```python
model_config = {
    'vocab_size': 10000,
    'embedding_dim': 300,
    'hidden_dim': 128,
    'output_dim': 1,
    'n_layers': 2,
    'dropout': 0.3,
    'bidirectional': True
}
```

### Training Configuration

```python
training_config = {
    'batch_size': 64,
    'learning_rate': 0.001,
    'epochs': 20,
    'weight_decay': 1e-5,
    'gradient_clip': 1.0,
    'early_stopping_patience': 5,
    'validation_split': 0.2
}
```

### Scheduler Options

```python
# Plateau scheduler (recommended)
scheduler_type = 'plateau'  # Reduces LR when validation loss plateaus

# Step scheduler
scheduler_type = 'step'     # Reduces LR every N epochs

# Cosine annealing
scheduler_type = 'cosine'   # Cosine annealing schedule

# No scheduler
scheduler_type = None       # Fixed learning rate
```

## Training Monitoring

### Training History

The trainer returns comprehensive training history:

```python
history = trainer.train(epochs=20)

# Available metrics
print(f"Training losses: {history['train_losses']}")
print(f"Validation losses: {history['val_losses']}")
print(f"Validation accuracies: {history['val_accuracies']}")
print(f"Learning rates: {history['learning_rates']}")
print(f"Best validation loss: {history['best_val_loss']}")
print(f"Total training time: {history['total_time']}")
```

### Training Summary

```python
summary = trainer.get_training_summary()

# Training status
print(f"Epochs completed: {summary['training_status']['epochs_completed']}")
print(f"Best epoch: {summary['training_status']['best_epoch']}")

# Performance metrics
print(f"Best validation accuracy: {summary['performance_metrics']['best_val_accuracy']}")
print(f"Final training loss: {summary['performance_metrics']['final_train_loss']}")

# Model information
print(f"Total parameters: {summary['model_info']['total_parameters']}")
```

## Integration Tests

Run the comprehensive test suite:

```bash
# Run all training tests
python -m pytest tests/test_training_integration.py -v

# Run specific test categories
python -m pytest tests/test_training_integration.py::TestTrainingIntegration -v
python -m pytest tests/test_training_integration.py::TestGloVeIntegration -v
```

### Test Coverage

- **Basic Training Workflow**: End-to-end training with synthetic data
- **Model Convergence**: Validation on separable synthetic data
- **Checkpoint Management**: Save/load functionality
- **Early Stopping**: Non-improving validation loss scenarios
- **GloVe Integration**: Embedding initialization and alignment
- **Scheduler Integration**: Learning rate scheduling
- **Training Summary**: Metrics and statistics generation

## Performance Considerations

### Memory Optimization

- **Gradient Accumulation**: For large effective batch sizes
- **Mixed Precision**: Use `torch.cuda.amp` for faster training
- **DataLoader Workers**: Optimize `num_workers` for your system
- **Pin Memory**: Enable for GPU training

### Training Speed

- **Batch Size**: Larger batches for better GPU utilization
- **Sequence Length**: Shorter sequences for faster training
- **Model Size**: Balance between capacity and speed
- **Checkpointing Frequency**: Balance between safety and speed

### Best Practices

1. **Start Small**: Begin with smaller models and datasets
2. **Monitor Overfitting**: Watch validation vs training metrics
3. **Learning Rate**: Use learning rate scheduling
4. **Regularization**: Apply dropout and weight decay
5. **Early Stopping**: Prevent overfitting with patience
6. **Checkpointing**: Save progress regularly
7. **Reproducibility**: Set random seeds for consistent results

## Error Handling

The training pipeline includes comprehensive error handling:

- **Data Loading Errors**: Validation and informative messages
- **Model Initialization**: Configuration validation
- **Training Failures**: Graceful handling with cleanup
- **Checkpoint Corruption**: Validation and recovery
- **Memory Issues**: Clear error messages and suggestions
- **Device Errors**: Automatic fallback and warnings

## File Structure

```
src/training/
├── __init__.py              # Module exports
├── trainer.py              # Core training engine
├── glove_loader.py         # GloVe embedding integration
├── checkpoint_manager.py   # Advanced checkpoint management
├── train_model.py          # Complete training script
└── README.md               # This file

tests/
└── test_training_integration.py  # Integration tests
```

## Dependencies

- PyTorch >= 1.9.0
- NumPy
- tqdm (for progress bars)
- urllib (for GloVe download)
- pytest (for testing)

## Examples

See `examples/` directory for complete training examples and notebooks demonstrating various training configurations and use cases.