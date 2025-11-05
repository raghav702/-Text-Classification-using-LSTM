"""
Configuration file for LSTM Sentiment Classifier
Contains hyperparameters and model settings
"""

import torch

# Model Architecture Configuration
MODEL_CONFIG = {
    'vocab_size': 10000,
    'embedding_dim': 300,
    'hidden_dim': 128,
    'output_dim': 1,
    'n_layers': 2,
    'dropout': 0.3,
    'bidirectional': True
}

# Training Configuration
TRAINING_CONFIG = {
    'batch_size': 64,
    'learning_rate': 0.001,
    'epochs': 20,
    'weight_decay': 1e-5,
    'gradient_clip': 1.0,
    'early_stopping_patience': 5,
    'validation_split': 0.2
}

# Data Processing Configuration
DATA_CONFIG = {
    'max_sequence_length': 500,
    'min_word_frequency': 2,
    'test_size': 0.2,
    'random_seed': 42,
    'unk_token': '<UNK>',
    'pad_token': '<PAD>',
    'special_tokens': ['<PAD>', '<UNK>']
}

# GloVe Embeddings Configuration
EMBEDDING_CONFIG = {
    'glove_dim': 300,
    'glove_name': '6B',
    'freeze_embeddings': False,
    'embedding_dropout': 0.1
}

# Training Infrastructure
DEVICE_CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 4,
    'pin_memory': True if torch.cuda.is_available() else False
}

# Paths Configuration
PATHS = {
    'data_dir': './data',
    'models_dir': './models',
    'logs_dir': './logs',
    'glove_dir': './data/glove',
    'imdb_dir': './data/imdb'
}

# Evaluation Configuration
EVAL_CONFIG = {
    'metrics': ['accuracy', 'precision', 'recall', 'f1'],
    'save_predictions': True,
    'save_plots': True,
    'plot_format': 'png'
}

# Logging Configuration
LOGGING_CONFIG = {
    'log_level': 'INFO',
    'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'save_logs': True,
    'console_output': True
}