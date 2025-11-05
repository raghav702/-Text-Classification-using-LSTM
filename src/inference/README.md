# LSTM Sentiment Classifier - Inference Engine

This module provides a complete inference engine for making sentiment predictions using trained LSTM models.

## Features

- **Model Loading**: Load trained LSTM models and vocabularies
- **Single Predictions**: Classify individual text inputs with confidence scores
- **Batch Processing**: Efficiently process multiple texts at once
- **Confidence Analysis**: Get detailed confidence metrics and threshold analysis
- **Command-Line Interface**: Easy-to-use CLI for quick predictions
- **Error Handling**: Robust input validation and error management

## Quick Start

### Basic Usage

```python
from src.inference import InferenceEngine

# Initialize and load model
engine = InferenceEngine()
engine.load_model('path/to/model.pth', 'path/to/vocab.pth')

# Single prediction
sentiment, confidence = engine.predict_sentiment("This movie was great!")
print(f"Sentiment: {sentiment}, Confidence: {confidence:.4f}")

# Batch prediction
texts = ["Great film!", "Terrible movie.", "It was okay."]
results = engine.batch_predict(texts)
for text, (sentiment, confidence) in zip(texts, results):
    print(f"{text} -> {sentiment} ({confidence:.4f})")
```

### Using the Factory Function

```python
from src.inference import create_inference_engine

# Create and initialize in one step
engine = create_inference_engine('model.pth', 'vocab.pth')
sentiment, confidence = engine.predict_sentiment("Amazing movie!")
```

## API Reference

### InferenceEngine Class

#### Core Methods

- `load_model(model_path, vocab_path)`: Load trained model and vocabulary
- `predict_sentiment(text, threshold=0.5)`: Single text prediction
- `batch_predict(texts, threshold=0.5)`: Batch text prediction
- `predict_sentiment_with_probability(text)`: Get raw probabilities
- `batch_predict_with_probabilities(texts)`: Batch with probabilities

#### Analysis Methods

- `predict_with_threshold_analysis(text, thresholds)`: Analyze across thresholds
- `get_prediction_stats(texts)`: Get batch statistics
- `get_model_info()`: Get model configuration and info

#### Utility Methods

- `validate_input(text)`: Validate input format
- `_preprocess_text(text)`: Preprocess single text
- `_preprocess_batch(texts)`: Preprocess batch of texts

### Return Formats

#### Single Prediction
```python
sentiment, confidence = engine.predict_sentiment(text)
# sentiment: 'positive' or 'negative'
# confidence: float [0, 1] - distance from decision boundary
```

#### With Probability
```python
sentiment, probability, confidence = engine.predict_sentiment_with_probability(text)
# sentiment: 'positive' or 'negative'
# probability: float [0, 1] - raw sigmoid output
# confidence: float [0, 1] - distance from 0.5
```

#### Batch Prediction
```python
results = engine.batch_predict(texts)
# results: List[Tuple[str, float]] - [(sentiment, confidence), ...]
```

## Command-Line Interface

The module includes a comprehensive CLI for easy predictions:

```bash
# Single text prediction
python src/inference/predict.py -m model.pth -v vocab.pth -t "Great movie!"

# Batch prediction from file
python src/inference/predict.py -m model.pth -v vocab.pth -f reviews.txt

# Interactive mode
python src/inference/predict.py -m model.pth -v vocab.pth --interactive

# With custom threshold and probabilities
python src/inference/predict.py -m model.pth -v vocab.pth -t "Good film" --threshold 0.6 --show-probability
```

### CLI Options

- `-m, --model`: Path to model checkpoint (required)
- `-v, --vocab`: Path to vocabulary file (required)
- `-t, --text`: Single text to classify
- `-f, --file`: File with texts (one per line)
- `--interactive`: Interactive mode
- `--threshold`: Decision threshold (default: 0.5)
- `--show-probability`: Show raw probabilities
- `--device`: Device to use (cpu/cuda)
- `--verbose`: Enable verbose logging

## Error Handling

The inference engine includes comprehensive error handling:

- **File Validation**: Checks for model and vocabulary file existence
- **Input Validation**: Validates text inputs and formats
- **Model Loading**: Handles checkpoint format variations
- **Device Management**: Automatic device detection and tensor placement
- **Graceful Failures**: Informative error messages for debugging

## Performance Considerations

- **Batch Processing**: Use batch methods for multiple texts (more efficient)
- **Device Selection**: Use GPU when available for faster inference
- **Memory Management**: Large batches are automatically handled
- **Preprocessing Cache**: Vocabulary is loaded once and reused

## Integration Examples

### Web API Integration
```python
from flask import Flask, request, jsonify
from src.inference import create_inference_engine

app = Flask(__name__)
engine = create_inference_engine('model.pth', 'vocab.pth')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['text']
    sentiment, confidence = engine.predict_sentiment(text)
    return jsonify({
        'sentiment': sentiment,
        'confidence': confidence
    })
```

### Streaming Processing
```python
def process_stream(text_stream):
    batch = []
    for text in text_stream:
        batch.append(text)
        if len(batch) >= 32:  # Process in batches of 32
            results = engine.batch_predict(batch)
            yield from zip(batch, results)
            batch = []
    
    # Process remaining texts
    if batch:
        results = engine.batch_predict(batch)
        yield from zip(batch, results)
```

## Requirements

- PyTorch >= 1.9.0
- NumPy
- Trained LSTM model checkpoint
- Vocabulary file from training

## File Structure

```
src/inference/
├── __init__.py              # Module exports
├── inference_engine.py      # Main inference engine
├── predict.py              # CLI interface
└── README.md               # This file

examples/
└── inference_example.py    # Usage examples
```