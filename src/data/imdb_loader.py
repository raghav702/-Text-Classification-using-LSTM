"""
IMDB dataset loader for sentiment classification.

This module provides functionality to download, load, and prepare
the IMDB movie review dataset for training and evaluation.
"""

import os
import torch
import pandas as pd
from typing import Tuple, List, Dict, Optional
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from .text_preprocessor import TextPreprocessor


class IMDBDataset:
    """
    IMDB dataset loader that reads from local directory structure.
    
    This class provides compatibility with the expected interface
    for loading IMDB data from the standard directory structure.
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize IMDB dataset loader.
        
        Args:
            data_dir: Path to IMDB dataset directory
        """
        self.data_dir = data_dir
        
    def load_data(self) -> Tuple[List[str], List[int], List[str], List[int]]:
        """
        Load IMDB data from directory structure.
        
        Returns:
            Tuple of (train_texts, train_labels, test_texts, test_labels)
        """
        def load_reviews_from_dir(dir_path: str) -> Tuple[List[str], List[int]]:
            """Load reviews from pos/neg subdirectories."""
            texts = []
            labels = []
            
            # Load positive reviews
            pos_dir = os.path.join(dir_path, 'pos')
            if os.path.exists(pos_dir):
                for filename in os.listdir(pos_dir):
                    if filename.endswith('.txt'):
                        filepath = os.path.join(pos_dir, filename)
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                text = f.read().strip()
                                texts.append(text)
                                labels.append(1)  # positive
                        except Exception as e:
                            print(f"Warning: Could not read {filepath}: {e}")
            
            # Load negative reviews
            neg_dir = os.path.join(dir_path, 'neg')
            if os.path.exists(neg_dir):
                for filename in os.listdir(neg_dir):
                    if filename.endswith('.txt'):
                        filepath = os.path.join(neg_dir, filename)
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                text = f.read().strip()
                                texts.append(text)
                                labels.append(0)  # negative
                        except Exception as e:
                            print(f"Warning: Could not read {filepath}: {e}")
            
            return texts, labels
        
        # Load training data
        train_dir = os.path.join(self.data_dir, 'train')
        train_texts, train_labels = load_reviews_from_dir(train_dir)
        
        # Load test data
        test_dir = os.path.join(self.data_dir, 'test')
        test_texts, test_labels = load_reviews_from_dir(test_dir)
        
        print(f"Loaded {len(train_texts)} training samples and {len(test_texts)} test samples")
        
        return train_texts, train_labels, test_texts, test_labels


class IMDBLoader:
    """
    IMDB dataset loader and manager.
    
    Handles downloading, loading, and splitting the IMDB movie review dataset
    while maintaining class balance and providing easy access to train/test splits.
    """
    
    def __init__(self, data_dir: str = "data", test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42):
        """
        Initialize IMDB dataset loader.
        
        Args:
            data_dir: Directory to store dataset files
            test_size: Proportion of data for test set
            val_size: Proportion of training data for validation set
            random_state: Random seed for reproducible splits
        """
        self.data_dir = data_dir
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Dataset storage
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.dataset_info = {}
    
    def download_and_load_imdb(self) -> Dict[str, any]:
        """
        Download and load IMDB dataset using Hugging Face datasets.
        
        Returns:
            Dictionary containing dataset information
        """
        print("Loading IMDB dataset...")
        
        try:
            # Load IMDB dataset from Hugging Face
            dataset = load_dataset("imdb")
            
            # Extract train and test data
            train_dataset = dataset['train']
            test_dataset = dataset['test']
            
            # Convert to pandas DataFrames for easier manipulation
            train_df = pd.DataFrame({
                'text': train_dataset['text'],
                'label': train_dataset['label']
            })
            
            test_df = pd.DataFrame({
                'text': test_dataset['text'],
                'label': test_dataset['label']
            })
            
            print(f"Loaded {len(train_df)} training samples and {len(test_df)} test samples")
            
            # Store dataset info
            self.dataset_info = {
                'total_samples': len(train_df) + len(test_df),
                'train_samples': len(train_df),
                'test_samples': len(test_df),
                'num_classes': 2,
                'class_names': ['negative', 'positive'],
                'train_class_distribution': train_df['label'].value_counts().to_dict(),
                'test_class_distribution': test_df['label'].value_counts().to_dict()
            }
            
            return {
                'train': train_df,
                'test': test_df,
                'info': self.dataset_info
            }
            
        except Exception as e:
            print(f"Error loading IMDB dataset: {e}")
            raise
    
    def create_train_val_split(self, train_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create train/validation split from training data.
        
        Args:
            train_df: Training DataFrame
            
        Returns:
            Tuple of (train_df, val_df)
        """
        if self.val_size <= 0:
            return train_df, pd.DataFrame()
        
        # Stratified split to maintain class balance
        train_texts = train_df['text'].tolist()
        train_labels = train_df['label'].tolist()
        
        X_train, X_val, y_train, y_val = train_test_split(
            train_texts, train_labels,
            test_size=self.val_size,
            random_state=self.random_state,
            stratify=train_labels
        )
        
        train_split = pd.DataFrame({'text': X_train, 'label': y_train})
        val_split = pd.DataFrame({'text': X_val, 'label': y_val})
        
        print(f"Created train split: {len(train_split)} samples")
        print(f"Created validation split: {len(val_split)} samples")
        
        return train_split, val_split
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load and split IMDB dataset.
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Download and load dataset
        dataset_dict = self.download_and_load_imdb()
        
        # Create train/validation split
        train_df, val_df = self.create_train_val_split(dataset_dict['train'])
        test_df = dataset_dict['test']
        
        # Store splits
        self.train_data = train_df
        self.val_data = val_df
        self.test_data = test_df
        
        # Print dataset statistics
        self.print_dataset_stats()
        
        return train_df, val_df, test_df
    
    def print_dataset_stats(self):
        """Print dataset statistics and class distributions."""
        print("\n=== IMDB Dataset Statistics ===")
        print(f"Total samples: {self.dataset_info['total_samples']}")
        print(f"Training samples: {len(self.train_data) if self.train_data is not None else 0}")
        print(f"Validation samples: {len(self.val_data) if self.val_data is not None else 0}")
        print(f"Test samples: {len(self.test_data) if self.test_data is not None else 0}")
        print(f"Number of classes: {self.dataset_info['num_classes']}")
        print(f"Class names: {self.dataset_info['class_names']}")
        
        if self.train_data is not None:
            train_dist = self.train_data['label'].value_counts().sort_index()
            print(f"Training class distribution: {dict(train_dist)}")
        
        if self.val_data is not None and len(self.val_data) > 0:
            val_dist = self.val_data['label'].value_counts().sort_index()
            print(f"Validation class distribution: {dict(val_dist)}")
        
        if self.test_data is not None:
            test_dist = self.test_data['label'].value_counts().sort_index()
            print(f"Test class distribution: {dict(test_dist)}")
        print("=" * 35)
    
    def get_sample_texts(self, split: str = 'train', num_samples: int = 5) -> List[Tuple[str, int]]:
        """
        Get sample texts from specified split.
        
        Args:
            split: Dataset split ('train', 'val', 'test')
            num_samples: Number of samples to return
            
        Returns:
            List of (text, label) tuples
        """
        if split == 'train' and self.train_data is not None:
            data = self.train_data
        elif split == 'val' and self.val_data is not None:
            data = self.val_data
        elif split == 'test' and self.test_data is not None:
            data = self.test_data
        else:
            return []
        
        samples = data.sample(n=min(num_samples, len(data)), random_state=self.random_state)
        return [(row['text'], row['label']) for _, row in samples.iterrows()]
    
    def save_splits(self, save_dir: str = None):
        """
        Save train/val/test splits to CSV files.
        
        Args:
            save_dir: Directory to save splits (uses self.data_dir if None)
        """
        if save_dir is None:
            save_dir = self.data_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        if self.train_data is not None:
            self.train_data.to_csv(os.path.join(save_dir, 'train.csv'), index=False)
            print(f"Saved training data to {os.path.join(save_dir, 'train.csv')}")
        
        if self.val_data is not None and len(self.val_data) > 0:
            self.val_data.to_csv(os.path.join(save_dir, 'val.csv'), index=False)
            print(f"Saved validation data to {os.path.join(save_dir, 'val.csv')}")
        
        if self.test_data is not None:
            self.test_data.to_csv(os.path.join(save_dir, 'test.csv'), index=False)
            print(f"Saved test data to {os.path.join(save_dir, 'test.csv')}")
    
    def load_splits_from_csv(self, data_dir: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load previously saved splits from CSV files.
        
        Args:
            data_dir: Directory containing CSV files (uses self.data_dir if None)
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if data_dir is None:
            data_dir = self.data_dir
        
        train_path = os.path.join(data_dir, 'train.csv')
        val_path = os.path.join(data_dir, 'val.csv')
        test_path = os.path.join(data_dir, 'test.csv')
        
        train_df = pd.read_csv(train_path) if os.path.exists(train_path) else pd.DataFrame()
        val_df = pd.read_csv(val_path) if os.path.exists(val_path) else pd.DataFrame()
        test_df = pd.read_csv(test_path) if os.path.exists(test_path) else pd.DataFrame()
        
        self.train_data = train_df if not train_df.empty else None
        self.val_data = val_df if not val_df.empty else None
        self.test_data = test_df if not test_df.empty else None
        
        print(f"Loaded splits from {data_dir}")
        if self.train_data is not None:
            print(f"Training samples: {len(self.train_data)}")
        if self.val_data is not None:
            print(f"Validation samples: {len(self.val_data)}")
        if self.test_data is not None:
            print(f"Test samples: {len(self.test_data)}")
        
        return train_df, val_df, test_df


def load_imdb_data(data_dir: str = "data", test_size: float = 0.2, val_size: float = 0.1, 
                   random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to load IMDB dataset with train/val/test splits.
    
    Args:
        data_dir: Directory to store dataset files
        test_size: Proportion of data for test set
        val_size: Proportion of training data for validation set
        random_state: Random seed for reproducible splits
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    loader = IMDBLoader(data_dir=data_dir, test_size=test_size, val_size=val_size, random_state=random_state)
    return loader.load_data()


if __name__ == "__main__":
    # Example usage
    loader = IMDBLoader()
    train_df, val_df, test_df = loader.load_data()
    
    # Show sample data
    print("\nSample training data:")
    samples = loader.get_sample_texts('train', 3)
    for i, (text, label) in enumerate(samples):
        sentiment = "positive" if label == 1 else "negative"
        print(f"{i+1}. [{sentiment}] {text[:100]}...")