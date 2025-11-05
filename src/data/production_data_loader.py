"""
Production-grade data loading pipeline for IMDB sentiment classification.

This module provides efficient data loading with proper batching, memory management,
caching, and streaming capabilities for large datasets.
"""

import os
import pickle
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterator, Union
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IMDBDataset(Dataset):
    """
    PyTorch Dataset for IMDB reviews with efficient loading and caching.
    """
    
    def __init__(
        self,
        csv_path: str,
        preprocessor=None,
        max_length: int = 512,
        cache_dir: Optional[str] = None,
        use_cache: bool = True
    ):
        """
        Initialize IMDB dataset.
        
        Args:
            csv_path: Path to CSV file containing text and label columns
            preprocessor: Text preprocessor instance
            max_length: Maximum sequence length for padding/truncation
            cache_dir: Directory for caching preprocessed data
            use_cache: Whether to use caching for preprocessed data
        """
        self.csv_path = Path(csv_path)
        self.preprocessor = preprocessor
        self.max_length = max_length
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_cache = use_cache
        
        # Load data
        self.data = self._load_data()
        self.texts = self.data['text'].tolist()
        self.labels = self.data['label'].tolist()
        
        # Setup caching
        if self.use_cache and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_file = self._get_cache_file_path()
            self.processed_data = self._load_or_create_cache()
        else:
            self.processed_data = None
    
    def _load_data(self) -> pd.DataFrame:
        """Load data from CSV file with validation."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        try:
            data = pd.read_csv(self.csv_path)
            
            # Validate required columns
            if 'text' not in data.columns or 'label' not in data.columns:
                raise ValueError("CSV must contain 'text' and 'label' columns")
            
            # Remove any null values
            initial_size = len(data)
            data = data.dropna(subset=['text', 'label'])
            if len(data) < initial_size:
                logger.warning(f"Removed {initial_size - len(data)} rows with null values")
            
            # Validate labels
            unique_labels = data['label'].unique()
            if not all(label in [0, 1] for label in unique_labels):
                raise ValueError("Labels must be 0 (negative) or 1 (positive)")
            
            logger.info(f"Loaded {len(data)} samples from {self.csv_path}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data from {self.csv_path}: {str(e)}")
            raise
    
    def _get_cache_file_path(self) -> Path:
        """Generate cache file path based on data and preprocessing parameters."""
        # Create hash from file content and preprocessing parameters
        with open(self.csv_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()[:8]
        
        preprocessor_hash = "none"
        if self.preprocessor:
            # Create hash from preprocessor configuration
            config_str = f"{self.max_length}_{getattr(self.preprocessor, 'vocab_size', 0)}"
            preprocessor_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        cache_filename = f"imdb_cache_{file_hash}_{preprocessor_hash}.pkl"
        return self.cache_dir / cache_filename
    
    def _load_or_create_cache(self) -> Optional[Dict]:
        """Load cached preprocessed data or create new cache."""
        if self.cache_file.exists():
            try:
                logger.info(f"Loading cached data from {self.cache_file}")
                with open(self.cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Validate cache integrity
                if len(cached_data['sequences']) == len(self.texts):
                    logger.info("Cache loaded successfully")
                    return cached_data
                else:
                    logger.warning("Cache size mismatch, regenerating...")
            except Exception as e:
                logger.warning(f"Error loading cache: {str(e)}, regenerating...")
        
        # Create new cache
        return self._create_cache()
    
    def _create_cache(self) -> Optional[Dict]:
        """Create cache of preprocessed data."""
        if not self.preprocessor:
            return None
        
        logger.info("Creating preprocessed data cache...")
        sequences = []
        
        for text in tqdm(self.texts, desc="Preprocessing texts"):
            try:
                # Preprocess text to sequence
                sequence = self.preprocessor.text_to_sequence(text)
                
                # Pad or truncate to max_length
                if len(sequence) > self.max_length:
                    sequence = sequence[:self.max_length]
                else:
                    sequence = sequence + [0] * (self.max_length - len(sequence))
                
                sequences.append(sequence)
            except Exception as e:
                logger.warning(f"Error preprocessing text: {str(e)}")
                # Use empty sequence as fallback
                sequences.append([0] * self.max_length)
        
        cached_data = {
            'sequences': sequences,
            'labels': self.labels,
            'max_length': self.max_length,
            'vocab_size': getattr(self.preprocessor, 'vocab_size', 0)
        }
        
        # Save cache
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cached_data, f)
            logger.info(f"Cache saved to {self.cache_file}")
        except Exception as e:
            logger.warning(f"Error saving cache: {str(e)}")
        
        return cached_data
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item by index."""
        if self.processed_data:
            # Use cached preprocessed data
            sequence = self.processed_data['sequences'][idx]
            label = self.processed_data['labels'][idx]
        else:
            # Process on-the-fly
            text = self.texts[idx]
            label = self.labels[idx]
            
            if self.preprocessor:
                sequence = self.preprocessor.text_to_sequence(text)
                # Pad or truncate
                if len(sequence) > self.max_length:
                    sequence = sequence[:self.max_length]
                else:
                    sequence = sequence + [0] * (self.max_length - len(sequence))
            else:
                # Fallback: return text as-is (for debugging)
                sequence = [0] * self.max_length
        
        return torch.tensor(sequence, dtype=torch.long), torch.tensor(label, dtype=torch.float)
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for balanced training."""
        label_counts = pd.Series(self.labels).value_counts().sort_index()
        total_samples = len(self.labels)
        
        # Calculate inverse frequency weights
        weights = []
        for label in [0, 1]:
            count = label_counts.get(label, 1)
            weight = total_samples / (2 * count)
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float)
    
    def get_sample_weights(self) -> List[float]:
        """Get sample weights for weighted sampling."""
        class_weights = self.get_class_weights()
        sample_weights = [class_weights[label].item() for label in self.labels]
        return sample_weights


class ProductionDataLoader:
    """
    Production-grade data loader with advanced features for IMDB sentiment classification.
    """
    
    def __init__(
        self,
        data_dir: str = "data/imdb",
        batch_size: int = 32,
        max_length: int = 512,
        num_workers: int = 4,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        pin_memory: bool = True
    ):
        """
        Initialize production data loader.
        
        Args:
            data_dir: Directory containing CSV files
            batch_size: Batch size for data loading
            max_length: Maximum sequence length
            num_workers: Number of worker processes for data loading
            cache_dir: Directory for caching preprocessed data
            use_cache: Whether to use preprocessing cache
            pin_memory: Whether to pin memory for faster GPU transfer
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.cache_dir = cache_dir or str(self.data_dir / "cache")
        self.use_cache = use_cache
        self.pin_memory = pin_memory
        
        # Dataset storage
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # DataLoader storage
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Preprocessor
        self.preprocessor = None
    
    def setup_datasets(self, preprocessor=None) -> Dict[str, IMDBDataset]:
        """
        Setup datasets from CSV files.
        
        Args:
            preprocessor: Text preprocessor instance
            
        Returns:
            Dictionary of datasets
        """
        self.preprocessor = preprocessor
        datasets = {}
        
        # Define expected CSV files
        csv_files = {
            'train': self.data_dir / 'train.csv',
            'validation': self.data_dir / 'validation.csv',
            'test': self.data_dir / 'test.csv'
        }
        
        # Create datasets
        for split_name, csv_path in csv_files.items():
            if csv_path.exists():
                logger.info(f"Setting up {split_name} dataset from {csv_path}")
                
                dataset = IMDBDataset(
                    csv_path=str(csv_path),
                    preprocessor=preprocessor,
                    max_length=self.max_length,
                    cache_dir=self.cache_dir,
                    use_cache=self.use_cache
                )
                
                datasets[split_name] = dataset
                
                # Store in instance variables
                if split_name == 'train':
                    self.train_dataset = dataset
                elif split_name == 'validation':
                    self.val_dataset = dataset
                elif split_name == 'test':
                    self.test_dataset = dataset
                
                logger.info(f"{split_name.capitalize()} dataset: {len(dataset)} samples")
            else:
                logger.warning(f"CSV file not found: {csv_path}")
        
        return datasets
    
    def create_data_loaders(
        self,
        use_weighted_sampling: bool = False,
        stratified_sampling: bool = True
    ) -> Dict[str, DataLoader]:
        """
        Create PyTorch DataLoaders with advanced sampling strategies.
        
        Args:
            use_weighted_sampling: Use weighted sampling for class balance
            stratified_sampling: Maintain class balance in batches
            
        Returns:
            Dictionary of DataLoaders
        """
        loaders = {}
        
        # Training loader with optional weighted sampling
        if self.train_dataset:
            sampler = None
            shuffle = True
            
            if use_weighted_sampling:
                sample_weights = self.train_dataset.get_sample_weights()
                sampler = WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(sample_weights),
                    replacement=True
                )
                shuffle = False  # Can't use shuffle with sampler
            
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=True,  # Drop incomplete batches for consistent training
                persistent_workers=True if self.num_workers > 0 else False
            )
            loaders['train'] = self.train_loader
        
        # Validation loader
        if self.val_dataset:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=True if self.num_workers > 0 else False
            )
            loaders['validation'] = self.val_loader
        
        # Test loader
        if self.test_dataset:
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=True if self.num_workers > 0 else False
            )
            loaders['test'] = self.test_loader
        
        # Log loader information
        for name, loader in loaders.items():
            logger.info(f"{name.capitalize()} loader: {len(loader)} batches")
        
        return loaders
    
    def get_data_statistics(self) -> Dict[str, Dict]:
        """Get comprehensive statistics about the loaded datasets."""
        stats = {}
        
        datasets = {
            'train': self.train_dataset,
            'validation': self.val_dataset,
            'test': self.test_dataset
        }
        
        for name, dataset in datasets.items():
            if dataset:
                labels = dataset.labels
                texts = dataset.texts
                
                # Calculate statistics
                label_counts = pd.Series(labels).value_counts().sort_index()
                text_lengths = [len(text.split()) for text in texts]
                
                stats[name] = {
                    'total_samples': len(dataset),
                    'positive_samples': int(label_counts.get(1, 0)),
                    'negative_samples': int(label_counts.get(0, 0)),
                    'class_balance': float(label_counts.get(1, 0)) / len(dataset),
                    'avg_text_length_words': float(np.mean(text_lengths)),
                    'max_text_length_words': int(np.max(text_lengths)),
                    'min_text_length_words': int(np.min(text_lengths)),
                    'std_text_length_words': float(np.std(text_lengths))
                }
        
        return stats
    
    def clear_cache(self):
        """Clear all cached preprocessed data."""
        if self.cache_dir:
            cache_path = Path(self.cache_dir)
            if cache_path.exists():
                for cache_file in cache_path.glob("*.pkl"):
                    try:
                        cache_file.unlink()
                        logger.info(f"Removed cache file: {cache_file}")
                    except Exception as e:
                        logger.warning(f"Error removing cache file {cache_file}: {str(e)}")
    
    def validate_data_integrity(self) -> bool:
        """Validate data integrity across all datasets."""
        try:
            datasets = [self.train_dataset, self.val_dataset, self.test_dataset]
            dataset_names = ['train', 'validation', 'test']
            
            for dataset, name in zip(datasets, dataset_names):
                if dataset:
                    logger.info(f"Validating {name} dataset...")
                    
                    # Check dataset size
                    if len(dataset) == 0:
                        logger.error(f"{name} dataset is empty")
                        return False
                    
                    # Sample a few items to check data loading
                    for i in range(min(5, len(dataset))):
                        try:
                            sequence, label = dataset[i]
                            
                            # Validate tensor shapes and types
                            if not isinstance(sequence, torch.Tensor):
                                logger.error(f"Invalid sequence type in {name} dataset")
                                return False
                            
                            if not isinstance(label, torch.Tensor):
                                logger.error(f"Invalid label type in {name} dataset")
                                return False
                            
                            if sequence.shape[0] != self.max_length:
                                logger.error(f"Invalid sequence length in {name} dataset")
                                return False
                            
                            if label.item() not in [0.0, 1.0]:
                                logger.error(f"Invalid label value in {name} dataset")
                                return False
                                
                        except Exception as e:
                            logger.error(f"Error accessing item {i} in {name} dataset: {str(e)}")
                            return False
            
            logger.info("Data integrity validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Error during data integrity validation: {str(e)}")
            return False


def create_production_data_loaders(
    data_dir: str = "data/imdb",
    preprocessor=None,
    batch_size: int = 32,
    max_length: int = 512,
    num_workers: int = 4,
    use_cache: bool = True,
    use_weighted_sampling: bool = False
) -> Tuple[Dict[str, DataLoader], ProductionDataLoader]:
    """
    Convenience function to create production-grade data loaders.
    
    Args:
        data_dir: Directory containing CSV files
        preprocessor: Text preprocessor instance
        batch_size: Batch size for data loading
        max_length: Maximum sequence length
        num_workers: Number of worker processes
        use_cache: Whether to use preprocessing cache
        use_weighted_sampling: Use weighted sampling for class balance
        
    Returns:
        Tuple of (data_loaders_dict, data_loader_manager)
    """
    # Create data loader manager
    loader_manager = ProductionDataLoader(
        data_dir=data_dir,
        batch_size=batch_size,
        max_length=max_length,
        num_workers=num_workers,
        use_cache=use_cache
    )
    
    # Setup datasets
    datasets = loader_manager.setup_datasets(preprocessor)
    
    # Create data loaders
    data_loaders = loader_manager.create_data_loaders(
        use_weighted_sampling=use_weighted_sampling
    )
    
    # Validate data integrity
    if not loader_manager.validate_data_integrity():
        raise RuntimeError("Data integrity validation failed")
    
    # Print statistics
    stats = loader_manager.get_data_statistics()
    logger.info("Dataset Statistics:")
    for split_name, split_stats in stats.items():
        logger.info(f"  {split_name.capitalize()}: {split_stats}")
    
    return data_loaders, loader_manager


if __name__ == "__main__":
    # Example usage
    logger.info("Testing production data loader...")
    
    try:
        # Create data loaders without preprocessor for testing
        data_loaders, manager = create_production_data_loaders(
            data_dir="data/imdb",
            batch_size=16,
            max_length=256,
            num_workers=2,
            use_cache=False  # Disable cache for testing
        )
        
        # Test training loader
        if 'train' in data_loaders:
            train_loader = data_loaders['train']
            logger.info(f"Training loader created with {len(train_loader)} batches")
            
            # Test first batch
            for batch_idx, (sequences, labels) in enumerate(train_loader):
                logger.info(f"Batch {batch_idx}: sequences shape {sequences.shape}, labels shape {labels.shape}")
                if batch_idx >= 2:  # Test only first few batches
                    break
        
        logger.info("Production data loader test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error testing production data loader: {str(e)}")
        raise