"""
PyTorch Dataset and DataLoader classes for IMDB sentiment classification.

This module provides custom Dataset classes that handle text preprocessing,
label encoding, and efficient batch processing for the LSTM sentiment classifier.
"""

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict, Any
from .text_preprocessor import TextPreprocessor


class IMDBDataset(Dataset):
    """
    Custom PyTorch Dataset for IMDB movie reviews.
    
    Handles text preprocessing, label encoding, and provides
    efficient access to individual samples for DataLoader.
    """
    
    def __init__(self, texts: List[str], labels: List[int], preprocessor: TextPreprocessor, 
                 max_length: Optional[int] = None):
        """
        Initialize IMDB dataset.
        
        Args:
            texts: List of review texts
            labels: List of sentiment labels (0=negative, 1=positive)
            preprocessor: TextPreprocessor instance for text processing
            max_length: Maximum sequence length (uses preprocessor default if None)
        """
        self.texts = texts
        self.labels = labels
        self.preprocessor = preprocessor
        self.max_length = max_length or preprocessor.max_length
        
        # Validate inputs
        if len(texts) != len(labels):
            raise ValueError(f"Number of texts ({len(texts)}) must match number of labels ({len(labels)})")
        
        # Convert labels to tensor
        self.labels_tensor = torch.tensor(labels, dtype=torch.float32)
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (processed_text_tensor, label_tensor)
        """
        # Get text and label
        text = self.texts[idx]
        label = self.labels_tensor[idx]
        
        # Preprocess text to sequence
        sequence = self.preprocessor.text_to_sequence(text)
        
        # Truncate if necessary
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
        
        return sequence, label
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """
        Get detailed information about a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with sample information
        """
        text = self.texts[idx]
        label = self.labels[idx]
        tokens = self.preprocessor.tokenize(text)
        sequence = self.preprocessor.text_to_sequence(text)
        
        return {
            'index': idx,
            'text': text,
            'label': label,
            'sentiment': 'positive' if label == 1 else 'negative',
            'num_tokens': len(tokens),
            'sequence_length': len(sequence),
            'tokens': tokens[:10],  # First 10 tokens
            'sequence': sequence[:10].tolist()  # First 10 indices
        }


class IMDBDatasetFromDataFrame(IMDBDataset):
    """
    IMDB Dataset that loads data from pandas DataFrame.
    
    Convenience class for creating datasets directly from DataFrames
    returned by the IMDB loader.
    """
    
    def __init__(self, dataframe: pd.DataFrame, preprocessor: TextPreprocessor, 
                 text_column: str = 'text', label_column: str = 'label',
                 max_length: Optional[int] = None):
        """
        Initialize dataset from DataFrame.
        
        Args:
            dataframe: pandas DataFrame with text and label columns
            preprocessor: TextPreprocessor instance
            text_column: Name of the text column
            label_column: Name of the label column
            max_length: Maximum sequence length
        """
        if dataframe.empty:
            texts, labels = [], []
        else:
            texts = dataframe[text_column].tolist()
            labels = dataframe[label_column].tolist()
        
        super().__init__(texts, labels, preprocessor, max_length)
        self.dataframe = dataframe


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Custom collate function for DataLoader to handle variable-length sequences.
    
    Args:
        batch: List of (sequence, label) tuples
        
    Returns:
        Tuple of (padded_sequences, labels)
    """
    sequences, labels = zip(*batch)
    
    # Find maximum length in this batch
    max_len = max(len(seq) for seq in sequences)
    
    # Pad sequences to max length in batch
    padded_sequences = []
    for seq in sequences:
        if len(seq) < max_len:
            # Pad with zeros (assuming PAD token index is 0)
            padding = torch.zeros(max_len - len(seq), dtype=torch.long)
            padded_seq = torch.cat([seq, padding])
        else:
            padded_seq = seq
        padded_sequences.append(padded_seq)
    
    # Stack sequences and labels
    sequences_tensor = torch.stack(padded_sequences)
    labels_tensor = torch.stack(labels)
    
    return sequences_tensor, labels_tensor


class IMDBDataLoaderManager:
    """
    Manager class for creating and managing IMDB DataLoaders.
    
    Provides convenient methods to create DataLoaders with appropriate
    configurations for training, validation, and testing.
    """
    
    def __init__(self, preprocessor: TextPreprocessor, batch_size: int = 64, 
                 num_workers: int = 0, pin_memory: bool = True):
        """
        Initialize DataLoader manager.
        
        Args:
            preprocessor: TextPreprocessor instance
            batch_size: Batch size for DataLoaders
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory for faster GPU transfer
        """
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
    
    def create_dataloader(self, dataset: Dataset, shuffle: bool = True, 
                         drop_last: bool = False) -> DataLoader:
        """
        Create a DataLoader for the given dataset.
        
        Args:
            dataset: PyTorch Dataset instance
            shuffle: Whether to shuffle the data
            drop_last: Whether to drop the last incomplete batch
            
        Returns:
            DataLoader instance
        """
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
            drop_last=drop_last
        )
    
    def create_dataloaders_from_dataframes(self, train_df: pd.DataFrame, 
                                         val_df: Optional[pd.DataFrame] = None,
                                         test_df: Optional[pd.DataFrame] = None) -> Dict[str, DataLoader]:
        """
        Create DataLoaders from pandas DataFrames.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame (optional)
            test_df: Test DataFrame (optional)
            
        Returns:
            Dictionary mapping split names to DataLoaders
        """
        dataloaders = {}
        
        # Training DataLoader
        if not train_df.empty:
            train_dataset = IMDBDatasetFromDataFrame(train_df, self.preprocessor)
            dataloaders['train'] = self.create_dataloader(train_dataset, shuffle=True, drop_last=True)
        
        # Validation DataLoader
        if val_df is not None and not val_df.empty:
            val_dataset = IMDBDatasetFromDataFrame(val_df, self.preprocessor)
            dataloaders['val'] = self.create_dataloader(val_dataset, shuffle=False, drop_last=False)
        
        # Test DataLoader
        if test_df is not None and not test_df.empty:
            test_dataset = IMDBDatasetFromDataFrame(test_df, self.preprocessor)
            dataloaders['test'] = self.create_dataloader(test_dataset, shuffle=False, drop_last=False)
        
        return dataloaders
    
    def create_dataloaders_from_lists(self, train_texts: List[str], train_labels: List[int],
                                    val_texts: Optional[List[str]] = None, val_labels: Optional[List[int]] = None,
                                    test_texts: Optional[List[str]] = None, test_labels: Optional[List[int]] = None) -> Dict[str, DataLoader]:
        """
        Create DataLoaders from lists of texts and labels.
        
        Args:
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Validation texts (optional)
            val_labels: Validation labels (optional)
            test_texts: Test texts (optional)
            test_labels: Test labels (optional)
            
        Returns:
            Dictionary mapping split names to DataLoaders
        """
        dataloaders = {}
        
        # Training DataLoader
        train_dataset = IMDBDataset(train_texts, train_labels, self.preprocessor)
        dataloaders['train'] = self.create_dataloader(train_dataset, shuffle=True, drop_last=True)
        
        # Validation DataLoader
        if val_texts is not None and val_labels is not None:
            val_dataset = IMDBDataset(val_texts, val_labels, self.preprocessor)
            dataloaders['val'] = self.create_dataloader(val_dataset, shuffle=False, drop_last=False)
        
        # Test DataLoader
        if test_texts is not None and test_labels is not None:
            test_dataset = IMDBDataset(test_texts, test_labels, self.preprocessor)
            dataloaders['test'] = self.create_dataloader(test_dataset, shuffle=False, drop_last=False)
        
        return dataloaders
    
    def get_dataloader_info(self, dataloader: DataLoader) -> Dict[str, Any]:
        """
        Get information about a DataLoader.
        
        Args:
            dataloader: DataLoader instance
            
        Returns:
            Dictionary with DataLoader information
        """
        dataset = dataloader.dataset
        
        return {
            'dataset_size': len(dataset),
            'batch_size': dataloader.batch_size,
            'num_batches': len(dataloader),
            'shuffle': dataloader.sampler is not None,
            'num_workers': dataloader.num_workers,
            'pin_memory': dataloader.pin_memory,
            'drop_last': dataloader.drop_last
        }
    
    def print_dataloader_stats(self, dataloaders: Dict[str, DataLoader]):
        """
        Print statistics for all DataLoaders.
        
        Args:
            dataloaders: Dictionary of DataLoaders
        """
        print("\n=== DataLoader Statistics ===")
        for split_name, dataloader in dataloaders.items():
            info = self.get_dataloader_info(dataloader)
            print(f"{split_name.upper()} DataLoader:")
            print(f"  Dataset size: {info['dataset_size']}")
            print(f"  Batch size: {info['batch_size']}")
            print(f"  Number of batches: {info['num_batches']}")
            print(f"  Shuffle: {info['shuffle']}")
        print("=" * 30)


def create_imdb_dataloaders(train_df: pd.DataFrame, preprocessor: TextPreprocessor,
                           val_df: Optional[pd.DataFrame] = None, test_df: Optional[pd.DataFrame] = None,
                           batch_size: int = 64, num_workers: int = 0) -> Dict[str, DataLoader]:
    """
    Convenience function to create IMDB DataLoaders from DataFrames.
    
    Args:
        train_df: Training DataFrame
        preprocessor: TextPreprocessor instance
        val_df: Validation DataFrame (optional)
        test_df: Test DataFrame (optional)
        batch_size: Batch size for DataLoaders
        num_workers: Number of worker processes
        
    Returns:
        Dictionary mapping split names to DataLoaders
    """
    manager = IMDBDataLoaderManager(preprocessor, batch_size=batch_size, num_workers=num_workers)
    return manager.create_dataloaders_from_dataframes(train_df, val_df, test_df)


if __name__ == "__main__":
    # Example usage
    from .imdb_loader import load_imdb_data
    
    # Load data
    train_df, val_df, test_df = load_imdb_data()
    
    # Create preprocessor and build vocabulary
    preprocessor = TextPreprocessor(max_vocab_size=10000, max_length=500)
    preprocessor.build_vocabulary(train_df['text'].tolist())
    
    # Create DataLoaders
    dataloaders = create_imdb_dataloaders(train_df, preprocessor, val_df, test_df, batch_size=32)
    
    # Print statistics
    manager = IMDBDataLoaderManager(preprocessor)
    manager.print_dataloader_stats(dataloaders)
    
    # Test a batch
    if 'train' in dataloaders:
        train_loader = dataloaders['train']
        batch_sequences, batch_labels = next(iter(train_loader))
        print(f"\nSample batch:")
        print(f"Sequences shape: {batch_sequences.shape}")
        print(f"Labels shape: {batch_labels.shape}")
        print(f"First sequence length: {(batch_sequences[0] != 0).sum().item()}")  # Non-padding tokens
        print(f"First label: {batch_labels[0].item()}")