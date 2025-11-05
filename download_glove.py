#!/usr/bin/env python3
"""
Download and process GloVe embeddings for LSTM sentiment classifier.

This script downloads GloVe 6B embeddings, processes them for vocabulary alignment,
and provides embedding coverage analysis and visualization tools.
"""

import os
import requests
import zipfile
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


class GloVeEmbeddingProcessor:
    """
    Processor for GloVe embeddings with vocabulary alignment and analysis capabilities.
    """
    
    def __init__(self, glove_dir: str = "data/glove"):
        """
        Initialize the GloVe embedding processor.
        
        Args:
            glove_dir: Directory to store GloVe embeddings
        """
        self.glove_dir = Path(glove_dir)
        self.glove_dir.mkdir(parents=True, exist_ok=True)
        
        # Available embedding dimensions
        self.available_dims = [50, 100, 200, 300]
        self.embeddings = {}  # Cache for loaded embeddings
        
    def download_glove_embeddings(self) -> bool:
        """
        Download GloVe 6B embeddings if not already present.
        
        Returns:
            True if download successful or already exists, False otherwise
        """
        url = "https://nlp.stanford.edu/data/glove.6B.zip"
        zip_path = self.glove_dir / "glove.6B.zip"
        
        # Check if embeddings already exist
        if self._check_embeddings_exist():
            print("GloVe embeddings already available.")
            return True
        
        print("Downloading GloVe 6B embeddings (862MB)...")
        
        try:
            if not zip_path.exists():
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                
                with open(zip_path, 'wb') as f:
                    downloaded = 0
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                pbar.update(len(chunk))
                
                print("Download completed!")
            
            # Extract embeddings
            print("Extracting GloVe embeddings...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.glove_dir)
            
            # Clean up zip file
            zip_path.unlink()
            
            print("GloVe embeddings ready!")
            self._print_available_files()
            return True
            
        except Exception as e:
            print(f"Error downloading GloVe embeddings: {e}")
            return False
    
    def _check_embeddings_exist(self) -> bool:
        """Check if GloVe embedding files exist."""
        required_files = [f"glove.6B.{dim}d.txt" for dim in self.available_dims]
        return all((self.glove_dir / file).exists() for file in required_files)
    
    def _print_available_files(self):
        """Print available GloVe embedding files."""
        print("Available embedding files:")
        for file in sorted(self.glove_dir.glob("glove.6B.*.txt")):
            file_size = file.stat().st_size / (1024 * 1024)  # MB
            print(f"  {file.name} ({file_size:.1f} MB)")
    
    def load_glove_embeddings(self, embedding_dim: int = 300) -> Dict[str, np.ndarray]:
        """
        Load GloVe embeddings from file.
        
        Args:
            embedding_dim: Dimension of embeddings to load (50, 100, 200, or 300)
            
        Returns:
            Dictionary mapping words to embedding vectors
        """
        if embedding_dim not in self.available_dims:
            raise ValueError(f"Embedding dimension {embedding_dim} not available. "
                           f"Choose from {self.available_dims}")
        
        # Check cache first
        if embedding_dim in self.embeddings:
            print(f"Using cached {embedding_dim}d embeddings")
            return self.embeddings[embedding_dim]
        
        embedding_file = self.glove_dir / f"glove.6B.{embedding_dim}d.txt"
        
        if not embedding_file.exists():
            raise FileNotFoundError(f"GloVe embedding file not found: {embedding_file}")
        
        print(f"Loading GloVe {embedding_dim}d embeddings...")
        embeddings = {}
        
        with open(embedding_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"Loading {embedding_dim}d embeddings"):
                values = line.strip().split()
                word = values[0]
                vector = np.array(values[1:], dtype=np.float32)
                
                if len(vector) == embedding_dim:
                    embeddings[word] = vector
        
        print(f"Loaded {len(embeddings):,} word embeddings")
        
        # Cache embeddings
        self.embeddings[embedding_dim] = embeddings
        return embeddings
    
    def align_embeddings_with_vocabulary(
        self, 
        vocabulary: Dict[str, int], 
        embedding_dim: int = 300,
        init_strategy: str = 'random'
    ) -> Tuple[torch.Tensor, Dict[str, any]]:
        """
        Align GloVe embeddings with model vocabulary.
        
        Args:
            vocabulary: Model vocabulary (word_to_idx mapping)
            embedding_dim: Dimension of embeddings
            init_strategy: Strategy for OOV words ('random', 'zero', 'mean')
            
        Returns:
            Tuple of (embedding_matrix, alignment_stats)
        """
        glove_embeddings = self.load_glove_embeddings(embedding_dim)
        vocab_size = len(vocabulary)
        
        # Initialize embedding matrix
        embedding_matrix = torch.zeros(vocab_size, embedding_dim, dtype=torch.float32)
        
        # Track alignment statistics
        found_words = 0
        oov_words = []
        
        print(f"Aligning embeddings with vocabulary ({vocab_size:,} words)...")
        
        for word, idx in tqdm(vocabulary.items(), desc="Aligning embeddings"):
            if word in glove_embeddings:
                embedding_matrix[idx] = torch.from_numpy(glove_embeddings[word])
                found_words += 1
            else:
                oov_words.append(word)
                
                # Handle out-of-vocabulary words
                if init_strategy == 'random':
                    # Xavier uniform initialization
                    embedding_matrix[idx] = torch.empty(embedding_dim).uniform_(-0.1, 0.1)
                elif init_strategy == 'zero':
                    # Keep as zeros (already initialized)
                    pass
                elif init_strategy == 'mean':
                    # Use mean of all GloVe embeddings
                    if hasattr(self, '_glove_mean'):
                        embedding_matrix[idx] = self._glove_mean
                    else:
                        # Calculate mean on first use
                        all_embeddings = np.array(list(glove_embeddings.values()))
                        self._glove_mean = torch.from_numpy(np.mean(all_embeddings, axis=0))
                        embedding_matrix[idx] = self._glove_mean
        
        # Ensure padding token (index 0) is zero
        if 0 < len(embedding_matrix):
            embedding_matrix[0] = torch.zeros(embedding_dim)
        
        # Calculate alignment statistics
        coverage = found_words / vocab_size if vocab_size > 0 else 0
        
        alignment_stats = {
            'vocab_size': vocab_size,
            'embedding_dim': embedding_dim,
            'found_words': found_words,
            'oov_words': len(oov_words),
            'coverage': coverage,
            'oov_word_list': oov_words[:50],  # First 50 OOV words for analysis
            'init_strategy': init_strategy
        }
        
        print(f"Embedding alignment complete:")
        print(f"  Coverage: {coverage:.2%} ({found_words:,}/{vocab_size:,} words)")
        print(f"  OOV words: {len(oov_words):,}")
        
        return embedding_matrix, alignment_stats
    
    def analyze_embedding_coverage(
        self, 
        vocabulary: Dict[str, int], 
        embedding_dim: int = 300
    ) -> Dict[str, any]:
        """
        Analyze embedding coverage for vocabulary.
        
        Args:
            vocabulary: Model vocabulary
            embedding_dim: Dimension of embeddings to analyze
            
        Returns:
            Dictionary with coverage analysis results
        """
        glove_embeddings = self.load_glove_embeddings(embedding_dim)
        
        # Analyze coverage
        total_words = len(vocabulary)
        found_words = sum(1 for word in vocabulary if word in glove_embeddings)
        oov_words = [word for word in vocabulary if word not in glove_embeddings]
        
        # Analyze OOV word patterns
        oov_patterns = defaultdict(int)
        for word in oov_words:
            if word.isdigit():
                oov_patterns['numeric'] += 1
            elif any(char.isdigit() for char in word):
                oov_patterns['alphanumeric'] += 1
            elif len(word) == 1:
                oov_patterns['single_char'] += 1
            elif word.startswith('<') and word.endswith('>'):
                oov_patterns['special_token'] += 1
            else:
                oov_patterns['other'] += 1
        
        coverage_stats = {
            'total_vocab_words': total_words,
            'found_in_glove': found_words,
            'oov_count': len(oov_words),
            'coverage_rate': found_words / total_words if total_words > 0 else 0,
            'oov_patterns': dict(oov_patterns),
            'sample_oov_words': oov_words[:20],
            'glove_vocab_size': len(glove_embeddings)
        }
        
        return coverage_stats
    
    def visualize_embedding_coverage(
        self, 
        coverage_stats: Dict[str, any], 
        save_path: Optional[str] = None
    ):
        """
        Create visualizations for embedding coverage analysis.
        
        Args:
            coverage_stats: Coverage statistics from analyze_embedding_coverage
            save_path: Optional path to save the plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Coverage pie chart
        coverage_rate = coverage_stats['coverage_rate']
        oov_rate = 1 - coverage_rate
        
        ax1.pie([coverage_rate, oov_rate], 
                labels=['Found in GloVe', 'Out-of-Vocabulary'],
                colors=['lightgreen', 'lightcoral'],
                autopct='%1.1f%%',
                startangle=90)
        ax1.set_title('Embedding Coverage')
        
        # OOV pattern distribution
        if coverage_stats['oov_patterns']:
            patterns = list(coverage_stats['oov_patterns'].keys())
            counts = list(coverage_stats['oov_patterns'].values())
            
            ax2.bar(patterns, counts, color='lightcoral', alpha=0.7)
            ax2.set_title('OOV Word Patterns')
            ax2.set_xlabel('Pattern Type')
            ax2.set_ylabel('Count')
            ax2.tick_params(axis='x', rotation=45)
        
        # Coverage statistics text
        stats_text = f"""
        Total Vocabulary: {coverage_stats['total_vocab_words']:,}
        Found in GloVe: {coverage_stats['found_in_glove']:,}
        Out-of-Vocabulary: {coverage_stats['oov_count']:,}
        Coverage Rate: {coverage_rate:.2%}
        GloVe Vocab Size: {coverage_stats['glove_vocab_size']:,}
        """
        
        ax3.text(0.1, 0.5, stats_text, transform=ax3.transAxes, 
                fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis('off')
        ax3.set_title('Coverage Statistics')
        
        # Sample OOV words
        if coverage_stats['sample_oov_words']:
            oov_sample = coverage_stats['sample_oov_words'][:10]
            ax4.barh(range(len(oov_sample)), [1] * len(oov_sample), color='lightcoral', alpha=0.7)
            ax4.set_yticks(range(len(oov_sample)))
            ax4.set_yticklabels(oov_sample)
            ax4.set_xlabel('Sample Count')
            ax4.set_title('Sample OOV Words')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Coverage visualization saved to: {save_path}")
        
        plt.show()
    
    def create_embedding_similarity_analysis(
        self, 
        words: List[str], 
        embedding_dim: int = 300,
        top_k: int = 5
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Analyze word similarities using GloVe embeddings.
        
        Args:
            words: List of words to analyze
            embedding_dim: Dimension of embeddings
            top_k: Number of most similar words to return
            
        Returns:
            Dictionary mapping each word to its most similar words
        """
        glove_embeddings = self.load_glove_embeddings(embedding_dim)
        
        def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
            """Calculate cosine similarity between two vectors."""
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            return dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0
        
        similarity_results = {}
        
        for word in words:
            if word not in glove_embeddings:
                similarity_results[word] = []
                continue
            
            word_vec = glove_embeddings[word]
            similarities = []
            
            # Calculate similarities with all other words (sample for efficiency)
            sample_words = list(glove_embeddings.keys())[:10000]  # Sample for speed
            
            for other_word in sample_words:
                if other_word != word:
                    other_vec = glove_embeddings[other_word]
                    sim = cosine_similarity(word_vec, other_vec)
                    similarities.append((other_word, sim))
            
            # Get top-k most similar words
            similarities.sort(key=lambda x: x[1], reverse=True)
            similarity_results[word] = similarities[:top_k]
        
        return similarity_results


def download_glove_embeddings():
    """Main function to download GloVe embeddings."""
    processor = GloVeEmbeddingProcessor()
    success = processor.download_glove_embeddings()
    
    if success:
        print("\nGloVe embeddings successfully downloaded and processed!")
        print("You can now use them with the GloVeEmbeddingProcessor class.")
    else:
        print("Failed to download GloVe embeddings.")
        return False
    
    return True


def main():
    """Main function with example usage."""
    # Download embeddings
    if not download_glove_embeddings():
        return
    
    # Example usage
    processor = GloVeEmbeddingProcessor()
    
    # Example vocabulary (you would use your actual model vocabulary)
    example_vocab = {
        '<PAD>': 0, '<UNK>': 1, 'good': 2, 'bad': 3, 'movie': 4,
        'great': 5, 'terrible': 6, 'amazing': 7, 'awful': 8
    }
    
    print("\n" + "="*50)
    print("EXAMPLE USAGE")
    print("="*50)
    
    # Analyze coverage
    coverage_stats = processor.analyze_embedding_coverage(example_vocab, embedding_dim=300)
    print(f"\nCoverage Analysis:")
    print(f"Coverage Rate: {coverage_stats['coverage_rate']:.2%}")
    print(f"OOV Words: {coverage_stats['sample_oov_words']}")
    
    # Align embeddings
    embedding_matrix, alignment_stats = processor.align_embeddings_with_vocabulary(
        example_vocab, embedding_dim=300, init_strategy='random'
    )
    print(f"\nEmbedding Matrix Shape: {embedding_matrix.shape}")
    
    # Word similarity analysis
    similarity_results = processor.create_embedding_similarity_analysis(
        ['good', 'bad', 'movie'], embedding_dim=300, top_k=3
    )
    
    print(f"\nWord Similarities:")
    for word, similar_words in similarity_results.items():
        print(f"{word}: {similar_words}")


if __name__ == "__main__":
    main()