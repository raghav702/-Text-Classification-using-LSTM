# LSTM Sentiment Classifier

A comprehensive implementation of an LSTM-based sentiment classifier for movie reviews using PyTorch. This project provides a complete pipeline for training, evaluating, and deploying sentiment analysis models with state-of-the-art features.

## Features

- **Complete LSTM Implementation**: Bidirectional LSTM with attention mechanisms
- **GloVe Integration**: Pre-trained word embeddings for improved performance
- **Comprehensive Training Pipeline**: Advanced training with early stopping, checkpointing, and scheduling
- **Professional Evaluation**: Detailed metrics, visualizations, and reporting
- **Production-Ready Inference**: Fast, scalable prediction engine
- **CLI Interface**: User-friendly command-line tools
- **Configuration Management**: Flexible YAML/JSON configuration system
- **Extensive Testing**: Comprehensive test suite for all components

## Quick Start

### 1. Setup Project

```bash
# Clone and setup
git clone <repository-url>
cd lstm-sentiment-classifier

# Install dependencies
pip install -r requirements.txt

# Setup project structure
python lstm_sentiment.py setup
```

### 2. Download Data

Download the IMDB movie review dataset and place it in `data/imdb/`:
- `data/imdb/train/pos/` - Positive training reviews
- `data/imdb/train/neg/` - Negative training reviews
- `data/imdb/test/pos/` - Positive test reviews
- `data/imdb/test/neg/` - Negative test reviews

### 3. Train Model

```bash
# Quick training (5 epochs, small model)
python lstm_sentiment.py train --config configs/examples/quick_training.yaml

# Production training (50 epochs, large model with GloVe)
python lstm_sentiment.py train --config configs/examples/production_training.yaml

# Custom training
python lstm_sentiment.py train --data-dir data/imdb --epochs 20 --batch-size 64
```

### 4. Make Predictions

```bash
# Single text prediction
python lstm_sentiment.py predict -m models/model.pth -v models/vocab.pth -t "This movie was amazing!"

# Interactive mode
python lstm_sentiment.py predict -m models/model.pth -v models/vocab.pth --interactive

# Batch prediction from file
python lstm_sentiment.py predict -m models/model.pth -v models/vocab.pth -f reviews.txt
```

### 5. Evaluate Model

```bash
# Comprehensive evaluation
python lstm_sentiment.py evaluate -m models/model.pth -v models/vocab.pth -d data/imdb

# Compare multiple models
python lstm_sentiment.py evaluate --model-paths model1.pth model2.pth --model-names "Model 1" "Model 2" -v vocab.pth -d data/imdb
```

## Architecture

### Model Components

- **Text Preprocessor**: Tokenization, vocabulary building, sequence padding
- **LSTM Model**: Bidirectional LSTM with embedding layer and classification head
- **Training Pipeline**: Advanced training with GloVe embeddings and optimization
- **Inference Engine**: Fast prediction with confidence scoring
- **Evaluation Suite**: Comprehensive metrics and visualization tools

### Key Features

- **Bidirectional LSTM**: Processes sequences in both directions for better context
- **GloVe Embeddings**: Pre-trained word vectors (6B, 42B, 840B tokens)
- **Attention Mechanism**: Optional attention for improved performance
- **Dropout Regularization**: Prevents overfitting during training
- **Gradient Clipping**: Prevents exploding gradients
- **Early Stopping**: Automatic training termination on convergence
- **Learning Rate Scheduling**: Adaptive learning rate adjustment

## Project Structure

```
lstm-sentiment-classifier/
├── src/                          # Source code
│   ├── data/                     # Data loading and preprocessing
│   ├── models/                   # Model architectures
│   ├── training/                 # Training pipeline
│   ├── inference/                # Inference engine
│   └── evaluation/               # Evaluation tools
├── tests/                        # Test suite
├── configs/                      # Configuration files
│   ├── examples/                 # Example configurations
│   ├── training_config.yaml     # Default training config
│   ├── inference_config.yaml    # Default inference config
│   └── evaluation_config.yaml   # Default evaluation config
├── data/                         # Data directory
│   ├── imdb/                     # IMDB dataset
│   └── glove/                    # GloVe embeddings cache
├── models/                       # Trained models
├── checkpoints/                  # Training checkpoints
├── logs/                         # Training logs
├── evaluation_results/           # Evaluation outputs
├── train.py                      # Main training script
├── predict.py                    # Main inference script
├── evaluate.py                   # Main evaluation script
├── lstm_sentiment.py             # Unified CLI interface
└── README.md                     # This file
```

## Configuration

The project uses YAML configuration files for easy parameter management:

### Training Configuration (`configs/training_config.yaml`)

```yaml
# Data Configuration
data_dir: "data/imdb"
batch_size: 64
max_vocab_size: 10000
validation_split: 0.2

# Model Architecture
embedding_dim: 300
hidden_dim: 128
n_layers: 2
dropout: 0.3
bidirectional: true

# Training Parameters
epochs: 20
learning_rate: 0.001
scheduler: "plateau"
early_stopping_patience: 5

# GloVe Embeddings
use_glove: true
glove_corpus: "6B"
glove_dim: "300d"
```

### Create Custom Configurations

```bash
# Create new configuration from template
python lstm_sentiment.py config --create training --output my_config.yaml

# Validate configuration
python lstm_sentiment.py config --validate my_config.yaml

# View configuration
python lstm_sentiment.py config --show my_config.yaml
```

## Command-Line Interface

### Unified CLI (`lstm_sentiment.py`)

The main CLI provides access to all functionality:

```bash
# Setup project
python lstm_sentiment.py setup

# Train model
python lstm_sentiment.py train --config configs/training_config.yaml

# Make predictions
python lstm_sentiment.py predict -m model.pth -v vocab.pth -t "Great movie!"

# Evaluate model
python lstm_sentiment.py evaluate -m model.pth -v vocab.pth -d data/imdb

# Configuration management
python lstm_sentiment.py config --create training
```

### Individual Scripts

Each component has its own script for advanced usage:

- `train.py` - Comprehensive training with all options
- `predict.py` - Flexible inference with multiple input modes
- `evaluate.py` - Detailed evaluation with visualization and reporting

## Training

### Basic Training

```bash
python train.py --data-dir data/imdb --epochs 20 --batch-size 64
```

### Advanced Training

```bash
python train.py \
    --config configs/production_training.yaml \
    --data-dir data/imdb \
    --epochs 50 \
    --use-glove \
    --glove-corpus 6B \
    --glove-dim 300d \
    --scheduler plateau \
    --early-stopping-patience 10 \
    --output-dir models/production
```

### Training Features

- **Automatic GloVe Download**: Downloads and caches GloVe embeddings
- **Vocabulary Alignment**: Intelligent mapping between dataset and GloVe vocabularies
- **Checkpointing**: Automatic model saving with best model tracking
- **Progress Monitoring**: Detailed logging and metrics tracking
- **Resume Training**: Continue from saved checkpoints
- **Configuration Saving**: Complete training configuration preservation

## Inference

### Single Text Prediction

```bash
python predict.py -m model.pth -v vocab.pth -t "This movie was fantastic!"
```

### Batch Prediction

```bash
# From file
python predict.py -m model.pth -v vocab.pth -f reviews.txt -o results.json

# With probabilities and statistics
python predict.py -m model.pth -v vocab.pth -f reviews.txt --show-probability --show-stats
```

### Interactive Mode

```bash
python predict.py -m model.pth -v vocab.pth --interactive
```

### Inference Features

- **Fast Processing**: Optimized batch processing for large datasets
- **Confidence Scoring**: Confidence measures for prediction reliability
- **Threshold Analysis**: Performance analysis across decision thresholds
- **Multiple Output Formats**: JSON, CSV, and console output
- **Input Validation**: Robust error handling and validation

## Evaluation

### Comprehensive Evaluation

```bash
python evaluate.py --model-path model.pth --vocab-path vocab.pth --data-dir data/imdb
```

### Model Comparison

```bash
python evaluate.py \
    --model-paths model1.pth model2.pth model3.pth \
    --model-names "Baseline" "With GloVe" "Large Model" \
    --vocab-path vocab.pth \
    --data-dir data/imdb \
    --output-dir comparison_results
```

### Evaluation Features

- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score, ROC AUC, PR AUC
- **Confusion Matrix**: Detailed confusion matrix analysis and visualization
- **ROC and PR Curves**: Receiver Operating Characteristic and Precision-Recall curves
- **Training History**: Loss and accuracy curves over training epochs
- **Threshold Analysis**: Performance across different decision thresholds
- **Automated Reporting**: Text and JSON evaluation reports
- **Visualization Dashboard**: Professional plots and charts

## API Usage

### Training API

```python
from src.training import create_trainer
from src.models.lstm_model import LSTMClassifier

# Create model
model = LSTMClassifier(vocab_size=10000, embedding_dim=300)

# Create trainer
trainer = create_trainer(model, train_loader, val_loader)

# Train model
history = trainer.train(epochs=20, early_stopping_patience=5)
```

### Inference API

```python
from src.inference import create_inference_engine

# Create inference engine
engine = create_inference_engine('model.pth', 'vocab.pth')

# Single prediction
sentiment, confidence = engine.predict_sentiment("Great movie!")

# Batch prediction
results = engine.batch_predict(["Good film", "Bad movie", "Amazing!"])
```

### Evaluation API

```python
from src.evaluation import calculate_metrics, plot_confusion_matrix

# Calculate comprehensive metrics
metrics = calculate_metrics(y_true, y_pred, y_prob)

# Create visualizations
plot_confusion_matrix(metrics['confusion_matrix'], save_path='cm.png')
```

## Performance

### Model Performance

- **Accuracy**: 85-90% on IMDB test set
- **Training Time**: 2-4 hours on GPU for full dataset
- **Inference Speed**: 1000+ predictions per second
- **Memory Usage**: 2-4 GB GPU memory for training

### Optimization Features

- **GPU Acceleration**: Automatic CUDA utilization
- **Batch Processing**: Efficient batch inference
- **Memory Management**: Optimized memory usage
- **Gradient Accumulation**: Support for large effective batch sizes
- **Mixed Precision**: Optional FP16 training for speed

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_training_integration.py -v
python -m pytest tests/test_evaluation.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Test Coverage

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow testing
- **Performance Tests**: Speed and memory benchmarks
- **Error Handling**: Edge case and error condition testing

## Dependencies

### Core Dependencies

- **PyTorch**: Deep learning framework
- **NumPy**: Numerical computing
- **scikit-learn**: Machine learning utilities
- **Matplotlib**: Plotting and visualization
- **Seaborn**: Statistical visualization
- **tqdm**: Progress bars
- **PyYAML**: YAML configuration files

### Optional Dependencies

- **Pandas**: Data manipulation (for CSV output)
- **Jupyter**: Interactive notebooks
- **TensorBoard**: Training visualization
- **pytest**: Testing framework

### Installation

```bash
# Install core dependencies
pip install torch numpy scikit-learn matplotlib seaborn tqdm pyyaml

# Install optional dependencies
pip install pandas jupyter tensorboard pytest

# Or install from requirements file
pip install -r requirements.txt
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd lstm-sentiment-classifier

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/ -v
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **IMDB Dataset**: Large Movie Review Dataset by Maas et al.
- **GloVe Embeddings**: Global Vectors for Word Representation by Pennington et al.
- **PyTorch**: Deep learning framework by Facebook AI Research
- **scikit-learn**: Machine learning library by the scikit-learn developers

## Citation

If you use this project in your research, please cite:

```bibtex
@software{lstm_sentiment_classifier,
  title={LSTM Sentiment Classifier},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/lstm-sentiment-classifier}
}
```