#!/usr/bin/env python3
"""
Download and prepare IMDB dataset for training with production-grade data pipeline
"""

import os
import requests
import tarfile
import pandas as pd
import hashlib
import logging
from pathlib import Path
from typing import Tuple, Dict, List
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IMDBDatasetDownloader:
    """Production-grade IMDB dataset downloader with validation and quality checks."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.imdb_dir = self.data_dir / "imdb"
        self.url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
        self.expected_md5 = "7c2ac02c03563afcf9b574c7e56c153a"  # Known MD5 for IMDB dataset
        
    def download_and_prepare(self) -> bool:
        """Main method to download and prepare IMDB dataset."""
        try:
            logger.info("Starting IMDB dataset download and preparation...")
            
            # Create directories
            self.imdb_dir.mkdir(parents=True, exist_ok=True)
            
            # Download dataset
            tar_path = self._download_dataset()
            if not tar_path:
                return False
                
            # Extract dataset
            if not self._extract_dataset(tar_path):
                return False
                
            # Convert to structured format
            if not self._convert_to_structured_format():
                return False
                
            # Validate data quality
            if not self._validate_data_quality():
                return False
                
            logger.info("IMDB dataset successfully prepared!")
            return True
            
        except Exception as e:
            logger.error(f"Error preparing IMDB dataset: {str(e)}")
            return False
    
    def _download_dataset(self) -> Path:
        """Download IMDB dataset with integrity checks."""
        tar_path = self.data_dir / "aclImdb_v1.tar.gz"
        
        if tar_path.exists():
            logger.info("Dataset file already exists. Verifying integrity...")
            if self._verify_file_integrity(tar_path):
                logger.info("Existing dataset file is valid.")
                return tar_path
            else:
                logger.warning("Existing dataset file is corrupted. Re-downloading...")
                tar_path.unlink()
        
        logger.info("Downloading IMDB dataset...")
        try:
            response = requests.get(self.url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(tar_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\rProgress: {percent:.1f}%", end="", flush=True)
            
            print()  # New line after progress
            logger.info("Download completed. Verifying integrity...")
            
            if self._verify_file_integrity(tar_path):
                logger.info("Dataset integrity verified.")
                return tar_path
            else:
                logger.error("Downloaded file failed integrity check.")
                tar_path.unlink()
                return None
                
        except requests.RequestException as e:
            logger.error(f"Download failed: {str(e)}")
            return None
    
    def _verify_file_integrity(self, file_path: Path) -> bool:
        """Verify file integrity using MD5 hash."""
        try:
            md5_hash = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    md5_hash.update(chunk)
            
            calculated_md5 = md5_hash.hexdigest()
            return calculated_md5 == self.expected_md5
        except Exception as e:
            logger.error(f"Error verifying file integrity: {str(e)}")
            return False
    
    def _extract_dataset(self, tar_path: Path) -> bool:
        """Extract dataset with error handling."""
        try:
            logger.info("Extracting dataset...")
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(self.imdb_dir)
            logger.info("Dataset extracted successfully.")
            return True
        except Exception as e:
            logger.error(f"Error extracting dataset: {str(e)}")
            return False

    def _convert_to_structured_format(self) -> bool:
        """Convert IMDB text files to structured CSV format with proper splits."""
        try:
            logger.info("Converting dataset to structured format...")
            
            imdb_extracted_dir = self.imdb_dir / "aclImdb"
            
            # Read all reviews
            train_reviews = self._read_reviews_from_directory(imdb_extracted_dir / "train")
            test_reviews = self._read_reviews_from_directory(imdb_extracted_dir / "test")
            
            if not train_reviews or not test_reviews:
                logger.error("Failed to read reviews from directories.")
                return False
            
            # Create DataFrames
            train_df = pd.DataFrame(train_reviews)
            test_df = pd.DataFrame(test_reviews)
            
            # Shuffle datasets
            train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
            test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # Create train/validation split from training data (80/20 split)
            train_data, val_data = train_test_split(
                train_df, test_size=0.2, random_state=42, stratify=train_df['label']
            )
            
            # Reset indices
            train_data = train_data.reset_index(drop=True)
            val_data = val_data.reset_index(drop=True)
            
            # Save datasets
            train_data.to_csv(self.imdb_dir / "train.csv", index=False)
            val_data.to_csv(self.imdb_dir / "validation.csv", index=False)
            test_df.to_csv(self.imdb_dir / "test.csv", index=False)
            
            # Log statistics
            logger.info(f"Dataset split completed:")
            logger.info(f"  Training samples: {len(train_data)} (pos: {sum(train_data['label'])}, neg: {len(train_data) - sum(train_data['label'])})")
            logger.info(f"  Validation samples: {len(val_data)} (pos: {sum(val_data['label'])}, neg: {len(val_data) - sum(val_data['label'])})")
            logger.info(f"  Test samples: {len(test_df)} (pos: {sum(test_df['label'])}, neg: {len(test_df) - sum(test_df['label'])})")
            
            return True
            
        except Exception as e:
            logger.error(f"Error converting to structured format: {str(e)}")
            return False
    
    def _read_reviews_from_directory(self, base_dir: Path) -> List[Dict]:
        """Read reviews from positive and negative directories."""
        reviews = []
        
        # Read positive reviews
        pos_dir = base_dir / "pos"
        if pos_dir.exists():
            for file_path in pos_dir.glob("*.txt"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                        if text:  # Only add non-empty reviews
                            reviews.append({'text': text, 'label': 1})
                except Exception as e:
                    logger.warning(f"Error reading {file_path}: {str(e)}")
        
        # Read negative reviews
        neg_dir = base_dir / "neg"
        if neg_dir.exists():
            for file_path in neg_dir.glob("*.txt"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                        if text:  # Only add non-empty reviews
                            reviews.append({'text': text, 'label': 0})
                except Exception as e:
                    logger.warning(f"Error reading {file_path}: {str(e)}")
        
        return reviews
    
    def _validate_data_quality(self) -> bool:
        """Perform comprehensive data quality validation."""
        try:
            logger.info("Performing data quality validation...")
            
            # Check if CSV files exist
            csv_files = ['train.csv', 'validation.csv', 'test.csv']
            for csv_file in csv_files:
                file_path = self.imdb_dir / csv_file
                if not file_path.exists():
                    logger.error(f"Missing CSV file: {csv_file}")
                    return False
            
            # Load and validate each dataset
            datasets = {}
            for csv_file in csv_files:
                file_path = self.imdb_dir / csv_file
                df = pd.read_csv(file_path)
                datasets[csv_file.replace('.csv', '')] = df
                
                # Basic validation
                if len(df) == 0:
                    logger.error(f"Empty dataset: {csv_file}")
                    return False
                
                # Check required columns
                if not all(col in df.columns for col in ['text', 'label']):
                    logger.error(f"Missing required columns in {csv_file}")
                    return False
                
                # Check for null values
                if df.isnull().any().any():
                    logger.warning(f"Found null values in {csv_file}")
                
                # Check label distribution
                label_counts = df['label'].value_counts()
                if len(label_counts) != 2:
                    logger.error(f"Invalid label distribution in {csv_file}: {label_counts}")
                    return False
                
                # Check text quality
                empty_texts = df['text'].str.strip().eq('').sum()
                if empty_texts > 0:
                    logger.warning(f"Found {empty_texts} empty texts in {csv_file}")
                
                # Log statistics
                avg_length = df['text'].str.len().mean()
                logger.info(f"{csv_file}: {len(df)} samples, avg length: {avg_length:.1f} chars")
            
            # Validate expected dataset sizes
            expected_sizes = {
                'train': 20000,  # 80% of 25k
                'validation': 5000,  # 20% of 25k
                'test': 25000
            }
            
            for dataset_name, expected_size in expected_sizes.items():
                actual_size = len(datasets[dataset_name])
                if actual_size != expected_size:
                    logger.warning(f"Unexpected size for {dataset_name}: {actual_size} (expected: {expected_size})")
            
            # Check class balance
            for dataset_name, df in datasets.items():
                pos_ratio = df['label'].mean()
                if not (0.4 <= pos_ratio <= 0.6):
                    logger.warning(f"Class imbalance in {dataset_name}: {pos_ratio:.3f} positive ratio")
            
            logger.info("Data quality validation completed successfully.")
            return True
            
        except Exception as e:
            logger.error(f"Error during data quality validation: {str(e)}")
            return False
    
    def get_dataset_info(self) -> Dict:
        """Get comprehensive dataset information."""
        info = {}
        
        csv_files = ['train.csv', 'validation.csv', 'test.csv']
        for csv_file in csv_files:
            file_path = self.imdb_dir / csv_file
            if file_path.exists():
                df = pd.read_csv(file_path)
                dataset_name = csv_file.replace('.csv', '')
                info[dataset_name] = {
                    'size': len(df),
                    'positive_samples': int(df['label'].sum()),
                    'negative_samples': int(len(df) - df['label'].sum()),
                    'avg_text_length': float(df['text'].str.len().mean()),
                    'max_text_length': int(df['text'].str.len().max()),
                    'min_text_length': int(df['text'].str.len().min())
                }
        
        return info


def main():
    """Main function to download and prepare IMDB dataset."""
    downloader = IMDBDatasetDownloader()
    
    if downloader.download_and_prepare():
        logger.info("Dataset preparation completed successfully!")
        
        # Print dataset information
        info = downloader.get_dataset_info()
        print("\n" + "="*50)
        print("DATASET INFORMATION")
        print("="*50)
        
        for dataset_name, stats in info.items():
            print(f"\n{dataset_name.upper()} SET:")
            print(f"  Total samples: {stats['size']:,}")
            print(f"  Positive samples: {stats['positive_samples']:,}")
            print(f"  Negative samples: {stats['negative_samples']:,}")
            print(f"  Average text length: {stats['avg_text_length']:.1f} characters")
            print(f"  Text length range: {stats['min_text_length']}-{stats['max_text_length']} characters")
        
        return True
    else:
        logger.error("Dataset preparation failed!")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)