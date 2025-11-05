"""
GloVe embeddings loader and integration utilities.

This module provides functionality to download, load, and integrate
pre-trained GloVe embeddings with the LSTM sentiment classifier.
"""

import os
import urllib.request
import zipfile
import torch
import numpy as np
from typing import Dict, Tuple, Optional, Set
import logging
from tqdm import tqdm

from data.text_preprocessor import TextPreprocessor


class GloVeLoader:
    """
    Loader for GloVe pre-trained word embeddings.
    
    Handles downloading, loading, and vocabulary alignment for GloVe embeddings.
    """
    
    # GloVe download URLs
    GLOVE_URLS = {
        '6B': {
            '50d': 'http://nlp.stanford.edu/data/glove.6B.zip',
            '100d': 'http://nlp.stanford.edu/data/glove.6B.zip',
            '200d': 'http://nlp.stanford.edu/data/glove.6B.zip',
            '300d': 'http://nlp.stanford.edu/data/glove.6B.zip'
        },
        '42B': {
            '300d': 'http://nlp.stanford.edu/data/glove.42B.300d.zip'
        },
        '840B': {
            '300d': 'http://nlp.stanford.edu/data/glove.840B.300d.zip'
        }
    }
    
    def __init__(self, cache_dir: str = "data/glove"):
        """
        Initialize GloVe loader.
        
        Args:
            cache_dir: Directory to cache downloaded GloVe files
        """
        self.cache_dir = cache_dir
        self.logger = logging.getLogger(__name__)
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
    
    def download_glove(self, corpus: str = '6B', dimension: str = '300d') -> str:
        """
        Download GloVe embeddings if not already cached.
        
        Args:
            corpus: GloVe corpus ('6B', '42B', '840B')
            dimension: Embedding dimension ('50d', '100d', '200d', '300d')
            
        Returns:
            Path to the downloaded/cached GloVe file
            
        Raises:
            ValueError: If corpus/dimension combination not available
            RuntimeError: If download fails
        """
        if corpus not in self.GLOVE_URLS:
            raise ValueError(f"Corpus '{corpus}' not available. Choose from: {list(self.GLOVE_URLS.keys())}")
        
        if dimension not in self.GLOVE_URLS[corpus]:
            available_dims = list(self.GLOVE_URLS[corpus].keys())
            raise ValueError(f"Dimension '{dimension}' not available for corpus '{corpus}'. "
                           f"Available: {available_dims}")
        
        # Determine file paths
        filename = f"glove.{corpus}.{dimension}.txt"
        filepath = os.path.join(self.cache_dir, filename)
        zip_filename = f"glove.{corpus}.zip"
        zip_filepath = os.path.join(self.cache_dir, zip_filename)
        
        # Check if already downloaded
        if os.path.exists(filepath):
            self.logger.info(f"GloVe embeddings already cached: {filepath}")
            return filepath
        
        # Download if needed
        url = self.GLOVE_URLS[corpus][dimension]
        
        try:
            self.logger.info(f"Downloading GloVe embeddings from {url}")
            
            # Download with progress bar
            def progress_hook(block_num, block_size, total_size):
                if hasattr(progress_hook, 'pbar'):
                    progress_hook.pbar.update(block_size)
                else:
                    progress_hook.pbar = tqdm(
                        total=total_size, unit='B', unit_scale=True,
                        desc=f"Downloading {zip_filename}"
                    )
            
            urllib.request.urlretrieve(url, zip_filepath, progress_hook)
            
            if hasattr(progress_hook, 'pbar'):
                progress_hook.pbar.close()
            
            # Extract the specific file we need
            self.logger.info(f"Extracting {filename} from archive")
            with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
                # Extract only the file we need
                zip_ref.extract(filename, self.cache_dir)
            
            # Clean up zip file
            os.remove(zip_filepath)
            
            self.logger.info(f"GloVe embeddings downloaded and extracted: {filepath}")
            return filepath
            
        except Exception as e:
            # Clean up partial downloads
            for temp_file in [zip_filepath, filepath]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            
            raise RuntimeError(f"Failed to download GloVe embeddings: {str(e)}")
    
    def load_glove_embeddings(
        self,
        filepath: str,
        vocab_size: Optional[int] = None
    ) -> Tuple[Dict[str, np.ndarray], int]:
        """
        Load GloVe embeddings from file.
        
        Args:
            filepath: Path to GloVe embeddings file
            vocab_size: Maximum vocabulary size to load (None for all)
            
        Returns:
            Tuple of (word_to_embedding_dict, embedding_dimension)
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"GloVe file not found: {filepath}")
        
        self.logger.info(f"Loading GloVe embeddings from {filepath}")
        
        embeddings = {}
        embedding_dim = None
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(tqdm(f, desc="Loading GloVe")):
                if vocab_size and line_num >= vocab_size:
                    break
                
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                
                word = parts[0]
                vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                
                if embedding_dim is None:
                    embedding_dim = len(vector)
                elif len(vector) != embedding_dim:
                    self.logger.warning(f"Inconsistent embedding dimension for word '{word}': "
                                      f"expected {embedding_dim}, got {len(vector)}")
                    continue
                
                embeddings[word] = vector
        
        self.logger.info(f"Loaded {len(embeddings)} embeddings with dimension {embedding_dim}")
        return embeddings, embedding_dim
    
    def create_embedding_matrix(
        self,
        preprocessor: TextPreprocessor,
        glove_embeddings: Dict[str, np.ndarray],
        embedding_dim: int,
        init_method: str = 'normal'
    ) -> Tuple[torch.Tensor, Dict[str, int]]:
        """
        Create embedding matrix aligned with vocabulary.
        
        Args:
            preprocessor: Text preprocessor with vocabulary
            glove_embeddings: Dictionary of GloVe embeddings
            embedding_dim: Embedding dimension
            init_method: Initialization method for OOV words ('normal', 'uniform', 'zero')
            
        Returns:
            Tuple of (embedding_matrix, alignment_stats)
        """
        vocab_size = preprocessor.vocab_size
        
        # Initialize embedding matrix
        embedding_matrix = torch.zeros(vocab_size, embedding_dim, dtype=torch.float32)
        
        # Statistics
        found_words = 0
        oov_words = 0
        
        # Fill embedding matrix
        for word, idx in preprocessor.word_to_idx.items():
            if word in glove_embeddings:
                embedding_matrix[idx] = torch.from_numpy(glove_embeddings[word])
                found_words += 1
            else:
                # Initialize OOV words
                if init_method == 'normal':
                    embedding_matrix[idx] = torch.randn(embedding_dim) * 0.1
                elif init_method == 'uniform':
                    embedding_matrix[idx] = torch.uniform_(-0.1, 0.1, (embedding_dim,))
                elif init_method == 'zero':
                    embedding_matrix[idx] = torch.zeros(embedding_dim)
                
                oov_words += 1
        
        # Ensure padding token is zero
        pad_idx = preprocessor.word_to_idx.get(preprocessor.PAD_TOKEN, 0)
        embedding_matrix[pad_idx] = torch.zeros(embedding_dim)
        
        alignment_stats = {
            'total_vocab_size': vocab_size,
            'found_in_glove': found_words,
            'oov_words': oov_words,
            'coverage_ratio': found_words / vocab_size,
            'embedding_dim': embedding_dim
        }
        
        self.logger.info(f"Embedding matrix created: {vocab_size} x {embedding_dim}")
        self.logger.info(f"GloVe coverage: {found_words}/{vocab_size} "
                        f"({alignment_stats['coverage_ratio']:.2%})")
        
        return embedding_matrix, alignment_stats


class EmbeddingInitializer:
    """
    Utility class for initializing model embeddings with GloVe.
    """
    
    def __init__(self, glove_loader: GloVeLoader = None):
        """
        Initialize embedding initializer.
        
        Args:
            glove_loader: GloVe loader instance (creates new if None)
        """
        self.glove_loader = glove_loader or GloVeLoader()
        self.logger = logging.getLogger(__name__)
    
    def initialize_embeddings(
        self,
        model,
        preprocessor: TextPreprocessor,
        corpus: str = '6B',
        dimension: str = '300d',
        freeze_embeddings: bool = False,
        init_method: str = 'normal'
    ) -> Dict[str, any]:
        """
        Initialize model embeddings with GloVe.
        
        Args:
            model: LSTM model with embedding layer
            preprocessor: Text preprocessor with vocabulary
            corpus: GloVe corpus to use
            dimension: Embedding dimension
            freeze_embeddings: Whether to freeze embedding weights
            init_method: Initialization method for OOV words
            
        Returns:
            Dictionary with initialization statistics
        """
        # Download and load GloVe embeddings
        glove_path = self.glove_loader.download_glove(corpus, dimension)
        glove_embeddings, glove_dim = self.glove_loader.load_glove_embeddings(glove_path)
        
        # Verify dimension compatibility
        model_embedding_dim = model.embedding_dim
        if glove_dim != model_embedding_dim:
            raise ValueError(
                f"GloVe dimension ({glove_dim}) doesn't match model embedding dimension "
                f"({model_embedding_dim})"
            )
        
        # Create aligned embedding matrix
        embedding_matrix, alignment_stats = self.glove_loader.create_embedding_matrix(
            preprocessor, glove_embeddings, glove_dim, init_method
        )
        
        # Load into model
        model.load_pretrained_embeddings(embedding_matrix)
        
        # Freeze embeddings if requested
        if freeze_embeddings:
            model.embedding.weight.requires_grad = False
            self.logger.info("Embedding weights frozen")
        
        initialization_stats = {
            'glove_corpus': corpus,
            'glove_dimension': dimension,
            'glove_path': glove_path,
            'frozen': freeze_embeddings,
            'init_method': init_method,
            **alignment_stats
        }
        
        self.logger.info("Model embeddings initialized with GloVe")
        
        return initialization_stats


def initialize_model_with_glove(
    model,
    preprocessor: TextPreprocessor,
    corpus: str = '6B',
    dimension: str = '300d',
    freeze_embeddings: bool = False,
    cache_dir: str = "data/glove"
) -> Dict[str, any]:
    """
    Convenience function to initialize model with GloVe embeddings.
    
    Args:
        model: LSTM model to initialize
        preprocessor: Text preprocessor with vocabulary
        corpus: GloVe corpus ('6B', '42B', '840B')
        dimension: Embedding dimension ('50d', '100d', '200d', '300d')
        freeze_embeddings: Whether to freeze embedding weights
        cache_dir: Directory to cache GloVe files
        
    Returns:
        Dictionary with initialization statistics
    """
    glove_loader = GloVeLoader(cache_dir)
    initializer = EmbeddingInitializer(glove_loader)
    
    return initializer.initialize_embeddings(
        model=model,
        preprocessor=preprocessor,
        corpus=corpus,
        dimension=dimension,
        freeze_embeddings=freeze_embeddings
    )