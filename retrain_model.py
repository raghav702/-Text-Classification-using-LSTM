#!/usr/bin/env python3
"""
Retrain the LSTM model with better parameters to fix the prediction issue
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.lstm_model import LSTMClassifier
from data.text_preprocessor import TextPreprocessor

def create_sample_data():
    """Create sample training data for quick testing."""
    positive_texts = [
        "This movie is absolutely amazing and fantastic!",
        "Great film with excellent acting and wonderful story",
        "I love this movie so much, it's incredible",
        "Best movie ever made, highly recommend",
        "Outstanding performance and brilliant direction",
        "Fantastic cinematography and amazing plot",
        "Excellent movie with great characters",
        "Wonderful film that I really enjoyed",
        "Amazing story with perfect execution",
        "Brilliant movie with outstanding acting",
        "Incredible film with fantastic visuals",
        "Perfect movie with excellent script",
        "Great story with wonderful performances",
        "Amazing cinematography and brilliant acting",
        "Excellent direction and fantastic story"
    ]
    
    negative_texts = [
        "This movie is terrible and awful, completely boring",
        "Boring movie with bad plot and poor acting",
        "I hate this movie completely, waste of time",
        "Worst movie ever made, terrible in every way",
        "Poor performance and horrible direction",
        "Terrible cinematography and awful plot",
        "Bad movie with terrible characters",
        "Horrible film that I really disliked",
        "Awful story with poor execution",
        "Terrible movie with bad acting",
        "Horrible film with poor visuals",
        "Bad movie with terrible script",
        "Poor story with awful performances",
        "Terrible cinematography and bad acting",
        "Poor direction and horrible story"
    ]
    
    # Create labels
    texts = positive_texts + negative_texts
    labels = [1] * len(positive_texts) + [0] * len(negative_texts)
    
    return texts, labels

def train_improved_model():
    """Train an improved LSTM model."""
    
    print("Creating sample training data...")
    texts, labels = create_sample_data()
    
    print("Initializing text preprocessor...")
    preprocessor = TextPreprocessor(max_vocab_size=5000, min_freq=1, max_length=200)
    
    # Build vocabulary and preprocess texts
    preprocessor.build_vocabulary(texts)
    sequences = [preprocessor.text_to_sequence(text) for text in texts]
    padded_sequences = preprocessor.pad_sequences(sequences)
    
    print(f"Vocabulary size: {preprocessor.vocab_size}")
    print(f"Training samples: {len(texts)}")
    
    # Create dataset
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    dataset = TensorDataset(padded_sequences, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Initialize improved model with better parameters
    model_config = {
        'vocab_size': preprocessor.vocab_size,
        'embedding_dim': 128,  # Increased from 100
        'hidden_dim': 128,     # Increased from 64
        'output_dim': 1,
        'n_layers': 2,         # Increased from 1
        'dropout': 0.3,        # Increased from 0.2
        'bidirectional': True, # Changed from False
        'pad_idx': preprocessor.word_to_idx[preprocessor.PAD_TOKEN]
    }
    
    print("Model configuration:")
    for key, value in model_config.items():
        print(f"  {key}: {value}")
    
    model = LSTMClassifier(**model_config)
    
    # Use BCEWithLogitsLoss for better numerical stability
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    print("\nStarting training...")
    model.train()
    
    # Training loop
    num_epochs = 50  # More epochs
    for epoch in range(num_epochs):
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        for batch_idx, (batch_sequences, batch_labels) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_sequences).squeeze()
            loss = criterion(outputs, batch_labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Calculate accuracy
            predictions = torch.sigmoid(outputs) > 0.5
            correct_predictions += (predictions == batch_labels.bool()).sum().item()
            total_predictions += batch_labels.size(0)
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    print("\nTraining completed!")
    
    # Test the model with diverse inputs
    print("\nTesting model predictions:")
    model.eval()
    
    test_texts = [
        "This movie is absolutely amazing!",
        "This movie is terrible and boring!",
        "Great film with excellent acting",
        "Boring movie with bad plot",
        "I love this movie",
        "I hate this movie"
    ]
    
    with torch.no_grad():
        for text in test_texts:
            sequence = preprocessor.text_to_sequence(text)
            padded = preprocessor.pad_sequences([sequence])
            
            logits = model(padded).squeeze()
            probability = torch.sigmoid(logits).item()
            sentiment = "positive" if probability > 0.5 else "negative"
            
            print(f"Text: '{text}'")
            print(f"  Probability: {probability:.4f}, Sentiment: {sentiment}")
    
    # Save the improved model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"improved_lstm_model_{timestamp}.pth"
    vocab_filename = f"improved_lstm_model_{timestamp}_vocabulary.pth"
    
    model_path = os.path.join("models", model_filename)
    vocab_path = os.path.join("models", vocab_filename)
    
    # Save model
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': model_config,
        'training_summary': {
            'final_loss': avg_loss,
            'final_accuracy': accuracy,
            'epochs': num_epochs
        },
        'timestamp': timestamp
    }
    
    torch.save(checkpoint, model_path)
    preprocessor.save_vocabulary(vocab_path)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Vocabulary saved to: {vocab_path}")
    
    return model_path, vocab_path

if __name__ == "__main__":
    train_improved_model()