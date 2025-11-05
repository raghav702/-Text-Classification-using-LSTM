#!/usr/bin/env python3
"""
Production-quality training script with real IMDB data and GloVe embeddings
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.lstm_model import LSTMClassifier
from data.text_preprocessor import TextPreprocessor

def load_glove_embeddings(glove_path, word_to_idx, embedding_dim):
    """Load GloVe embeddings for vocabulary."""
    print(f"Loading GloVe embeddings from {glove_path}...")
    
    embeddings = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    
    print(f"Loaded {len(embeddings)} GloVe vectors")
    
    # Create embedding matrix
    vocab_size = len(word_to_idx)
    embedding_matrix = np.random.normal(0, 0.1, (vocab_size, embedding_dim))
    
    found_words = 0
    for word, idx in word_to_idx.items():
        if word in embeddings:
            embedding_matrix[idx] = embeddings[word]
            found_words += 1
    
    print(f"Found embeddings for {found_words}/{vocab_size} words ({found_words/vocab_size*100:.1f}%)")
    
    return torch.FloatTensor(embedding_matrix)

def train_production_model():
    """Train production-quality model with IMDB data and GloVe embeddings."""
    
    # Check if data exists
    train_path = "data/imdb/train.csv"
    test_path = "data/imdb/test.csv"
    glove_path = "data/glove/glove.6B.100d.txt"
    
    if not os.path.exists(train_path):
        print("IMDB training data not found. Run: python download_imdb_data.py")
        return
    
    if not os.path.exists(glove_path):
        print("GloVe embeddings not found. Run: python download_glove.py")
        return
    
    print("Loading IMDB dataset...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Use subset for faster training (remove this for full dataset)
    train_df = train_df.sample(n=5000, random_state=42)  # Use 5000 samples
    test_df = test_df.sample(n=1000, random_state=42)    # Use 1000 for testing
    
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(max_vocab_size=10000, min_freq=5, max_length=300)
    
    # Build vocabulary from training data
    print("Building vocabulary...")
    preprocessor.build_vocabulary(train_df['text'].tolist())
    
    # Preprocess data
    print("Preprocessing texts...")
    train_sequences = [preprocessor.text_to_sequence(text) for text in train_df['text']]
    test_sequences = [preprocessor.text_to_sequence(text) for text in test_df['text']]
    
    train_padded = preprocessor.pad_sequences(train_sequences)
    test_padded = preprocessor.pad_sequences(test_sequences)
    
    train_labels = torch.tensor(train_df['label'].values, dtype=torch.float32)
    test_labels = torch.tensor(test_df['label'].values, dtype=torch.float32)
    
    # Create data loaders
    train_dataset = TensorDataset(train_padded, train_labels)
    test_dataset = TensorDataset(test_padded, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Model configuration
    embedding_dim = 100  # Match GloVe dimension
    model_config = {
        'vocab_size': preprocessor.vocab_size,
        'embedding_dim': embedding_dim,
        'hidden_dim': 128,
        'output_dim': 1,
        'n_layers': 2,
        'dropout': 0.3,
        'bidirectional': True,
        'pad_idx': preprocessor.word_to_idx[preprocessor.PAD_TOKEN]
    }
    
    print("Model configuration:")
    for key, value in model_config.items():
        print(f"  {key}: {value}")
    
    # Initialize model
    model = LSTMClassifier(**model_config)
    
    # Load GloVe embeddings
    embedding_matrix = load_glove_embeddings(glove_path, preprocessor.word_to_idx, embedding_dim)
    model.load_pretrained_embeddings(embedding_matrix)
    
    # Training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    
    # Training loop
    num_epochs = 10
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    print("\nStarting training...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        train_predictions = []
        train_targets = []
        
        for batch_sequences, batch_labels in train_loader:
            optimizer.zero_grad()
            
            outputs = model(batch_sequences).squeeze()
            loss = criterion(outputs, batch_labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            
            # Collect predictions for accuracy
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            train_predictions.extend(predictions.cpu().numpy())
            train_targets.extend(batch_labels.cpu().numpy())
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch_sequences, batch_labels in test_loader:
                outputs = model(batch_sequences).squeeze()
                loss = criterion(outputs, batch_labels)
                
                total_val_loss += loss.item()
                
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                val_predictions.extend(predictions.cpu().numpy())
                val_targets.extend(batch_labels.cpu().numpy())
        
        # Calculate metrics
        train_loss = total_train_loss / len(train_loader)
        val_loss = total_val_loss / len(test_loader)
        train_acc = accuracy_score(train_targets, train_predictions)
        val_acc = accuracy_score(val_targets, val_predictions)
        
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Final evaluation
    print("\nFinal Evaluation:")
    precision, recall, f1, _ = precision_recall_fscore_support(val_targets, val_predictions, average='binary')
    print(f"Accuracy: {val_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Test with sample texts
    print("\nTesting sample predictions:")
    test_texts = [
        "This movie is absolutely amazing and fantastic!",
        "This movie is terrible and boring!",
        "The film was okay, nothing special",
        "Outstanding performance and brilliant direction",
        "Poor acting and horrible plot"
    ]
    
    model.eval()
    with torch.no_grad():
        for text in test_texts:
            sequence = preprocessor.text_to_sequence(text)
            padded = preprocessor.pad_sequences([sequence])
            
            logits = model(padded).squeeze()
            probability = torch.sigmoid(logits).item()
            sentiment = "positive" if probability > 0.5 else "negative"
            
            print(f"'{text}' → {sentiment} ({probability:.3f})")
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"production_lstm_model_{timestamp}.pth"
    vocab_filename = f"production_lstm_model_{timestamp}_vocabulary.pth"
    
    model_path = os.path.join("models", model_filename)
    vocab_path = os.path.join("models", vocab_filename)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': model_config,
        'training_history': {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies
        },
        'final_metrics': {
            'accuracy': val_acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        },
        'timestamp': timestamp
    }
    
    torch.save(checkpoint, model_path)
    preprocessor.save_vocabulary(vocab_path)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Vocabulary saved to: {vocab_path}")
    
    return model_path, vocab_path

if __name__ == "__main__":
    train_production_model()