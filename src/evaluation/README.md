# LSTM Sentiment Classifier - Evaluation Module

This module provides comprehensive evaluation functionality for the LSTM sentiment classifier, including metrics calculation, visualization, and reporting tools.

## Features

- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score, specificity, ROC AUC, PR AUC
- **Confusion Matrix Analysis**: Detailed confusion matrix statistics and visualization
- **Threshold Analysis**: Performance analysis across different decision thresholds
- **ROC and PR Curves**: Receiver Operating Characteristic and Precision-Recall curves
- **Training History Visualization**: Loss and accuracy curves over training epochs
- **Automated Reporting**: Text and JSON format evaluation reports
- **Evaluation Dashboard**: Comprehensive visualization dashboard
- **Performance Comparison**: Tools for comparing different model configurations

## Quick Start

### Basic Evaluation

```python
from src.evaluation import calculate_metrics, plot_confusion_matrix

# Calculate comprehensive metrics
metrics = calculate_metrics(y_true, y_pred, y_prob, ['Negative', 'Positive'])

# Plot confusion matrix
plot_confusion_matrix(
    metrics['confusion_matrix'],
    class_names=['Negative', 'Positive'],
    title='Model Performance'
)
```

### Command-Line Evaluation

```bash
# Basic evaluation
python src/evaluation/evaluate_model.py \
    --model-path models/lstm_model.pth \
    --vocab-path models/vocabulary.pth \
    --data-dir data/imdb

# Comprehensive evaluation with visualizations
python src/evaluation/evaluate_model.py \
    --model-path models/lstm_model.pth \
    --vocab-path models/vocabulary.pth \
    --data-dir data/imdb \
    --output-dir evaluation_results \
    --show-plots
```

## Module Components

### 1. Metrics Calculator (`metrics.py`)

Comprehensive metrics calculation with support for binary classification:

```python
from src.evaluation import MetricsCalculator

calculator = MetricsCalculator(['Negative', 'Positive'])

# Basic metrics
basic_metrics = calculator.calculate_basic_metrics(y_true, y_pred, y_prob)

# Confusion matrix with statistics
cm, stats = calculator.calculate_confusion_matrix(y_true, y_pred)

# Per-class metrics
per_class = calculator.calculate_per_class_metrics(y_true, y_pred)

# Comprehensive evaluation
comprehensive = calculator.calculate_comprehensive_metrics(y_true, y_pred, y_prob)

# Threshold optimization
optimal_threshold, best_f1 = calculator.find_optimal_threshold(y_true, y_prob, 'f1_score')
```

#### Available Metrics

**Basic Metrics:**
- Accuracy
- Precision
- Recall
- F1-Score
- Specificity (True Negative Rate)
- ROC AUC
- PR AUC (Average Precision)

**Confusion Matrix Statistics:**
- True Positives/Negatives
- False Positives/Negatives
- True/False Positive Rates
- Positive/Negative Predictive Values

**Advanced Analysis:**
- Per-class metrics
- Threshold analysis
- ROC curve data
- Precision-Recall curve data

### 2. Visualization Tools (`visualization.py`)

Comprehensive visualization capabilities for model evaluation:

```python
from src.evaluation import EvaluationVisualizer

visualizer = EvaluationVisualizer()

# Confusion matrix plot
fig = visualizer.plot_confusion_matrix(
    confusion_matrix,
    class_names=['Negative', 'Positive'],
    normalize=False,
    save_path='confusion_matrix.png'
)

# Training history plot
fig = visualizer.plot_training_history(
    history,
    title='Training Progress',
    save_path='training_history.png'
)

# ROC curve
fig = visualizer.plot_roc_curve(
    fpr, tpr, auc_score,
    save_path='roc_curve.png'
)

# Precision-Recall curve
fig = visualizer.plot_precision_recall_curve(
    precision, recall, ap_score,
    save_path='pr_curve.png'
)

# Threshold analysis
fig = visualizer.plot_threshold_analysis(
    threshold_metrics,
    metrics_to_plot=['accuracy', 'precision', 'recall', 'f1_score'],
    save_path='threshold_analysis.png'
)

# Comprehensive dashboard
figures = visualizer.create_evaluation_dashboard(
    metrics,
    history=training_history,
    save_dir='evaluation_plots'
)
```

#### Visualization Features

- **High-Quality Plots**: Publication-ready figures with proper formatting
- **Customizable Styling**: Configurable colors, fonts, and layouts
- **Automatic Saving**: Save plots in various formats (PNG, PDF, SVG)
- **Interactive Elements**: Highlighted best epochs and optimal points
- **Comprehensive Dashboards**: Multiple plots in organized layouts

### 3. Report Generation (`visualization.py`)

Automated report generation in multiple formats:

```python
from src.evaluation import EvaluationReporter

reporter = EvaluationReporter()

# Text report
text_report = reporter.generate_text_report(
    metrics,
    model_info=model_config,
    training_summary=training_stats
)

# JSON report
json_report = reporter.generate_json_report(
    metrics,
    model_info=model_config,
    training_summary=training_stats
)

# Save reports to files
reporter.save_report(metrics, 'evaluation_report.txt', 'text')
reporter.save_report(metrics, 'evaluation_report.json', 'json')
```

#### Report Features

- **Comprehensive Coverage**: All metrics, statistics, and model information
- **Multiple Formats**: Text and JSON formats for different use cases
- **Structured Layout**: Organized sections for easy reading
- **Metadata Inclusion**: Timestamps, model configuration, training summary
- **Export Capabilities**: Save to files or return as strings/dictionaries

## Evaluation Workflow

### 1. Complete Model Evaluation

```python
# Load model predictions
y_true, y_pred, y_prob = get_model_predictions(model, test_loader)

# Calculate comprehensive metrics
metrics = calculate_metrics(y_true, y_pred, y_prob)

# Create visualizations
visualizer = EvaluationVisualizer()
figures = visualizer.create_evaluation_dashboard(
    metrics,
    save_dir='evaluation_results'
)

# Generate reports
reporter = EvaluationReporter()
reporter.save_report(metrics, 'evaluation_report.txt', 'text')
```

### 2. Threshold Analysis

```python
from src.evaluation import MetricsCalculator

calculator = MetricsCalculator()

# Analyze performance across thresholds
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
threshold_metrics = calculator.calculate_threshold_metrics(y_true, y_prob, thresholds)

# Find optimal threshold
optimal_threshold, best_f1 = calculator.find_optimal_threshold(y_true, y_prob, 'f1_score')

# Visualize threshold analysis
visualizer = EvaluationVisualizer()
visualizer.plot_threshold_analysis(threshold_metrics)
```

### 3. Model Comparison

```python
# Compare multiple models
models_metrics = {}

for model_name, (y_true, y_pred, y_prob) in model_predictions.items():
    models_metrics[model_name] = calculate_metrics(y_true, y_pred, y_prob)

# Create comparison visualizations
for model_name, metrics in models_metrics.items():
    visualizer.plot_confusion_matrix(
        metrics['confusion_matrix'],
        title=f'{model_name} - Confusion Matrix',
        save_path=f'{model_name}_confusion_matrix.png'
    )
```

## Command-Line Interface

The evaluation module includes a comprehensive CLI for model evaluation:

```bash
# Basic evaluation
python src/evaluation/evaluate_model.py \
    --model-path models/best_model.pth \
    --vocab-path models/vocabulary.pth \
    --data-dir data/imdb

# Advanced evaluation with all features
python src/evaluation/evaluate_model.py \
    --model-path models/best_model.pth \
    --vocab-path models/vocabulary.pth \
    --data-dir data/imdb \
    --output-dir evaluation_results \
    --batch-size 64 \
    --show-plots \
    --log-level INFO

# Evaluation without visualizations (faster)
python src/evaluation/evaluate_model.py \
    --model-path models/best_model.pth \
    --vocab-path models/vocabulary.pth \
    --data-dir data/imdb \
    --no-visualizations \
    --no-reports
```

### CLI Options

- `--model-path`: Path to trained model checkpoint (required)
- `--vocab-path`: Path to vocabulary file (required)
- `--data-dir`: Path to IMDB dataset directory (required)
- `--output-dir`: Directory to save results (default: evaluation_results)
- `--batch-size`: Batch size for evaluation (default: 64)
- `--device`: Device to use (cpu/cuda/auto)
- `--show-plots`: Display plots during evaluation
- `--no-visualizations`: Skip visualization generation
- `--no-reports`: Skip report generation
- `--log-level`: Logging level (DEBUG/INFO/WARNING/ERROR)

## Testing

Run the comprehensive test suite:

```bash
# Run all evaluation tests
python -m pytest tests/test_evaluation.py -v

# Run specific test categories
python -m pytest tests/test_evaluation.py::TestMetricsCalculator -v
python -m pytest tests/test_evaluation.py::TestEvaluationVisualizer -v
python -m pytest tests/test_evaluation.py::TestEvaluationReporter -v
```

### Test Coverage

- **Metrics Calculation**: Accuracy validation with known ground truth
- **Visualization Generation**: Plot creation without errors
- **Report Generation**: Text and JSON report validation
- **Error Handling**: Input validation and edge cases
- **Integration Workflows**: Complete evaluation pipelines

## Performance Considerations

### Memory Optimization

- **Batch Processing**: Efficient evaluation on large datasets
- **Lazy Loading**: Load data as needed to minimize memory usage
- **GPU Utilization**: Automatic GPU usage when available

### Computation Speed

- **Vectorized Operations**: NumPy and PyTorch optimizations
- **Parallel Processing**: Multi-worker data loading
- **Caching**: Avoid redundant calculations

### Scalability

- **Large Datasets**: Handle datasets of any size
- **Multiple Models**: Compare many models efficiently
- **Batch Evaluation**: Process multiple test sets

## Best Practices

### Evaluation Methodology

1. **Use Separate Test Set**: Never evaluate on training data
2. **Multiple Metrics**: Don't rely on accuracy alone
3. **Threshold Analysis**: Find optimal decision threshold
4. **Cross-Validation**: Use multiple evaluation runs when possible
5. **Statistical Significance**: Consider confidence intervals

### Visualization Guidelines

1. **Clear Labels**: Always label axes and provide legends
2. **Appropriate Scales**: Use log scales when necessary
3. **Color Accessibility**: Use colorblind-friendly palettes
4. **High Resolution**: Save plots at publication quality
5. **Consistent Styling**: Maintain visual consistency across plots

### Reporting Standards

1. **Complete Information**: Include all relevant metrics
2. **Model Details**: Document model architecture and training
3. **Reproducibility**: Provide enough detail for reproduction
4. **Version Control**: Track evaluation results over time
5. **Comparative Analysis**: Compare against baselines

## Integration Examples

### Web Dashboard Integration

```python
from flask import Flask, render_template
from src.evaluation import calculate_metrics, EvaluationVisualizer

app = Flask(__name__)

@app.route('/evaluate/<model_id>')
def evaluate_model(model_id):
    # Load model predictions
    y_true, y_pred, y_prob = load_model_predictions(model_id)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    
    # Create visualizations
    visualizer = EvaluationVisualizer()
    figures = visualizer.create_evaluation_dashboard(
        metrics,
        save_dir=f'static/evaluations/{model_id}'
    )
    
    return render_template('evaluation.html', 
                         metrics=metrics, 
                         figures=figures)
```

### Automated Model Monitoring

```python
import schedule
import time
from src.evaluation import calculate_metrics

def daily_model_evaluation():
    """Daily automated model evaluation."""
    # Load latest test data
    y_true, y_pred, y_prob = get_latest_predictions()
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    
    # Check for performance degradation
    if metrics['basic_metrics']['accuracy'] < 0.8:
        send_alert("Model performance degraded!")
    
    # Save results
    save_evaluation_results(metrics, datetime.now())

# Schedule daily evaluation
schedule.every().day.at("02:00").do(daily_model_evaluation)

while True:
    schedule.run_pending()
    time.sleep(3600)  # Check every hour
```

## Dependencies

- **Core**: NumPy, PyTorch, scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Data Processing**: Pandas (optional)
- **Testing**: pytest
- **Utilities**: tqdm, logging

## File Structure

```
src/evaluation/
├── __init__.py              # Module exports
├── metrics.py              # Metrics calculation
├── visualization.py        # Visualization and reporting
├── evaluate_model.py       # Complete evaluation script
└── README.md               # This file

tests/
└── test_evaluation.py      # Comprehensive test suite
```