#!/usr/bin/env python3
"""
Retrain the LSTM model with proper GloVe embeddings and sufficient training.
This script addresses the issue where the model gives consistent predictions.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.lstm_model import LSTMClassifier
from data.text_preprocessor import TextPreprocessor
from data.dataset import IMDBDataset
from training.glove_loader import initialize_model_with_glove
from training.trainer import create_trainer

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/retrain_model.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_imdb_data():
    """Load IMDB dataset."""
    train_path = "data/imdb/train.csv"
    test_path = "data/imdb/test.csv"
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("IMDB dataset not found. Please run: python download_imdb_data.py")
        return None, None
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Loaded training data: {len(train_df)} samples")
    print(f"Loaded test data: {len(test_df)} samples")
    
    return train_df, test_df

def collate_fn(batch):
    """Custom collate function to handle variable length sequences."""
    sequences, labels = zip(*batch)
    
    # Find maximum length in this batch
    max_len = max(len(seq) for seq in sequences)
    
    # Pad sequences to max length in batch
    padded_sequences = []
    for seq in sequences:
        if len(seq) < max_len:
            # Pad with zeros (PAD token index is 0)
            padding = torch.zeros(max_len - len(seq), dtype=torch.long)
            padded_seq = torch.cat([seq, padding])
        else:
            padded_seq = seq
        padded_sequences.append(padded_seq)
    
    # Stack sequences and labels
    sequences_tensor = torch.stack(padded_sequences)
    labels_tensor = torch.stack(labels)
    
    return sequences_tensor, labels_tensor

def create_datasets_and_loaders(train_df, test_df, preprocessor, batch_size=32):
    """Create datasets and data loaders."""
    
    # Check column names and convert labels if needed
    text_col = 'review' if 'review' in train_df.columns else 'text'
    label_col = 'sentiment' if 'sentiment' in train_df.columns else 'label'
    
    # Convert string labels to integers if needed
    train_labels = train_df[label_col].tolist()
    test_labels = test_df[label_col].tolist()
    
    if isinstance(train_labels[0], str):
        # Convert 'positive'/'negative' to 1/0
        train_labels = [1 if label == 'positive' else 0 for label in train_labels]
        test_labels = [1 if label == 'positive' else 0 for label in test_labels]
    
    # Create datasets
    train_dataset = IMDBDataset(
        texts=train_df[text_col].tolist(),
        labels=train_labels,
        preprocessor=preprocessor
    )
    
    test_dataset = IMDBDataset(
        texts=test_df[text_col].tolist(),
        labels=test_labels,
        preprocessor=preprocessor
    )
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=False,  # Disable for CPU training
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn
    )
    
    return train_loader, test_loader

def train_epoch(model, train_loader, optimizer, criterion, device, logger):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (texts, labels) in enumerate(train_loader):
        texts, labels = texts.to(device), labels.to(device).float()
        
        optimizer.zero_grad()
        outputs = model(texts).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        if batch_idx % 100 == 0:
            logger.info(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, '
                       f'Acc: {100.*correct/total:.2f}%')
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

def evaluate_model(model, test_loader, criterion, device):
    """Evaluate model on test set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device).float()
            outputs = model(texts).squeeze()
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

def main():
    """Main training function."""
    logger = setup_logging()
    logger.info("Starting improved model training with GloVe embeddings")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    train_df, test_df = load_imdb_data()
    if train_df is None:
        return
    
    # Initialize preprocessor and build vocabulary
    logger.info("Building vocabulary...")
    preprocessor = TextPreprocessor(max_vocab_size=10000, min_freq=2, max_length=200)
    
    # Build vocabulary from training data
    text_col = 'review' if 'review' in train_df.columns else 'text'
    train_texts = train_df[text_col].tolist()
    vocab_stats = preprocessor.build_vocabulary(train_texts)
    logger.info(f"Vocabulary built: {preprocessor.vocab_size} words")
    
    # Create datasets and loaders
    logger.info("Creating datasets...")
    train_loader, test_loader = create_datasets_and_loaders(
        train_df, test_df, preprocessor, batch_size=64
    )
    
    # Model configuration
    model_config = {
        'vocab_size': preprocessor.vocab_size,
        'embedding_dim': 300,  # Match GloVe 300d
        'hidden_dim': 128,
        'output_dim': 1,
        'n_layers': 2,
        'dropout': 0.3,
        'bidirectional': True,
        'pad_idx': preprocessor.word_to_idx[preprocessor.PAD_TOKEN]
    }
    
    # Initialize model
    logger.info("Initializing model...")
    model = LSTMClassifier(**model_config)
    
    # Initialize with GloVe embeddings
    logger.info("Loading GloVe embeddings...")
    try:
        embedding_stats = initialize_model_with_glove(
            model=model,
            preprocessor=preprocessor,
            corpus='6B',
            dimension='300d',
            freeze_embeddings=False,  # Allow fine-tuning
            cache_dir='data/glove'
        )
        logger.info(f"GloVe coverage: {embedding_stats['coverage_ratio']:.2%}")
    except Exception as e:
        logger.warning(f"GloVe initialization failed: {e}")
        logger.info("Using random embeddings")
    
    model.to(device)
    
    # Training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Training loop
    num_epochs = 15
    best_val_acc = 0
    patience = 5
    patience_counter = 0
    
    logger.info("Starting training...")
    
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, logger
        )
        
        # Evaluate
        val_loss, val_acc = evaluate_model(model, test_loader, criterion, device)
        
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping and model saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save best model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = f"models/improved_lstm_model_{timestamp}.pth"
            vocab_path = f"models/improved_lstm_model_{timestamp}_vocabulary.pth"
            
            # Save model with full configuration
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': model_config,
                'embedding_stats': embedding_stats if 'embedding_stats' in locals() else None,
                'epoch': epoch,
                'best_val_acc': best_val_acc,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'optimizer_state_dict': optimizer.state_dict()
            }, model_path)
            
            # Save vocabulary
            preprocessor.save_vocabulary(vocab_path)
            
            logger.info(f"New best model saved: {model_path}")
            logger.info(f"Vocabulary saved: {vocab_path}")
            
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")
    
    # Test the model with some examples
    logger.info("\nTesting model with sample texts...")
    model.eval()
    
    test_texts = [
        "This movie is absolutely amazing and fantastic!",
        "This movie is terrible and awful!",
        "Great film with excellent acting",
        "Boring movie with bad plot",
        "I love this movie so much",
        "I hate this movie completely"
    ]
    
    with torch.no_grad():
        for text in test_texts:
            sequence = preprocessor.text_to_sequence(text)
            padded = preprocessor.pad_sequences([sequence]).to(device)
            output = model(padded)
            probability = torch.sigmoid(output).item()
            sentiment = "positive" if probability > 0.5 else "negative"
            
            logger.info(f"'{text}' -> {sentiment} ({probability:.4f})")

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    main()