"""
Text augmentation module for LSTM sentiment classifier.

This module provides various text augmentation techniques including synonym replacement,
random word operations, and quality control for data augmentation.
"""

import random
import re
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import requests
import json


class TextAugmenter:
    """
    Text augmentation class for sentiment analysis data enhancement.
    
    Provides various augmentation techniques including synonym replacement,
    random word insertion/deletion/swapping, and quality control.
    """
    
    def __init__(self, 
                 synonym_prob: float = 0.1,
                 insert_prob: float = 0.1, 
                 delete_prob: float = 0.1,
                 swap_prob: float = 0.1,
                 max_augmentations: int = 1,
                 preserve_length: bool = True):
        """
        Initialize the TextAugmenter.
        
        Args:
            synonym_prob: Probability of replacing a word with synonym
            insert_prob: Probability of inserting a random word
            delete_prob: Probability of deleting a word
            swap_prob: Probability of swapping adjacent words
            max_augmentations: Maximum number of augmentations per text
            preserve_length: Whether to preserve approximate text length
        """
        self.synonym_prob = synonym_prob
        self.insert_prob = insert_prob
        self.delete_prob = delete_prob
        self.swap_prob = swap_prob
        self.max_augmentations = max_augmentations
        self.preserve_length = preserve_length
        
        # Synonym dictionary (will be populated from various sources)
        self.synonym_dict = defaultdict(list)
        self.word_embeddings = {}
        
        # Common words for insertion (sentiment-neutral)
        self.common_words = [
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'throughout',
            'despite', 'towards', 'upon', 'concerning', 'under', 'within', 'without',
            'also', 'however', 'therefore', 'moreover', 'furthermore', 'nevertheless'
        ]
        
        # Initialize basic synonym mappings
        self._init_basic_synonyms()
    
    def _init_basic_synonyms(self):
        """Initialize basic synonym mappings for common words."""
        basic_synonyms = {
            'good': ['great', 'excellent', 'wonderful', 'fantastic', 'amazing', 'superb'],
            'bad': ['terrible', 'awful', 'horrible', 'dreadful', 'poor', 'disappointing'],
            'great': ['excellent', 'wonderful', 'fantastic', 'amazing', 'superb', 'outstanding'],
            'terrible': ['awful', 'horrible', 'dreadful', 'bad', 'poor', 'disappointing'],
            'love': ['adore', 'enjoy', 'like', 'appreciate', 'cherish'],
            'hate': ['despise', 'detest', 'dislike', 'loathe', 'abhor'],
            'beautiful': ['gorgeous', 'stunning', 'lovely', 'attractive', 'pretty'],
            'ugly': ['hideous', 'unattractive', 'unsightly', 'repulsive'],
            'big': ['large', 'huge', 'enormous', 'massive', 'giant'],
            'small': ['tiny', 'little', 'miniature', 'compact', 'petite'],
            'fast': ['quick', 'rapid', 'swift', 'speedy', 'hasty'],
            'slow': ['sluggish', 'gradual', 'leisurely', 'unhurried'],
            'smart': ['intelligent', 'clever', 'brilliant', 'wise', 'bright'],
            'stupid': ['foolish', 'dumb', 'ignorant', 'silly', 'senseless'],
            'funny': ['hilarious', 'amusing', 'entertaining', 'comical', 'humorous'],
            'boring': ['dull', 'tedious', 'monotonous', 'uninteresting', 'tiresome'],
            'happy': ['joyful', 'cheerful', 'delighted', 'pleased', 'content'],
            'sad': ['unhappy', 'depressed', 'melancholy', 'sorrowful', 'gloomy'],
            'amazing': ['incredible', 'astonishing', 'remarkable', 'extraordinary', 'phenomenal'],
            'awful': ['terrible', 'horrible', 'dreadful', 'appalling', 'atrocious']
        }
        
        for word, synonyms in basic_synonyms.items():
            self.synonym_dict[word.lower()] = [s.lower() for s in synonyms]
            # Add reverse mappings
            for synonym in synonyms:
                if synonym.lower() not in self.synonym_dict:
                    self.synonym_dict[synonym.lower()] = [word.lower()]
                elif word.lower() not in self.synonym_dict[synonym.lower()]:
                    self.synonym_dict[synonym.lower()].append(word.lower())
    
    def load_word_embeddings(self, embedding_dict: Dict[str, np.ndarray]):
        """
        Load word embeddings for similarity-based synonym replacement.
        
        Args:
            embedding_dict: Dictionary mapping words to embedding vectors
        """
        self.word_embeddings = embedding_dict
    
    def find_similar_words(self, word: str, top_k: int = 5, threshold: float = 0.6) -> List[str]:
        """
        Find similar words using word embeddings.
        
        Args:
            word: Target word
            top_k: Number of similar words to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of similar words
        """
        if word.lower() not in self.word_embeddings:
            return []
        
        target_embedding = self.word_embeddings[word.lower()]
        similarities = []
        
        for candidate_word, embedding in self.word_embeddings.items():
            if candidate_word == word.lower():
                continue
            
            # Calculate cosine similarity
            similarity = np.dot(target_embedding, embedding) / (
                np.linalg.norm(target_embedding) * np.linalg.norm(embedding)
            )
            
            if similarity >= threshold:
                similarities.append((candidate_word, similarity))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [word for word, _ in similarities[:top_k]]
    
    def synonym_replacement(self, tokens: List[str], n: int = 1) -> List[str]:
        """
        Replace n random words with their synonyms.
        
        Args:
            tokens: List of word tokens
            n: Number of words to replace
            
        Returns:
            Augmented token list
        """
        if len(tokens) == 0:
            return tokens
        
        new_tokens = tokens.copy()
        random_word_list = list(range(len(tokens)))
        random.shuffle(random_word_list)
        
        num_replaced = 0
        for random_idx in random_word_list:
            if num_replaced >= n:
                break
            
            word = tokens[random_idx].lower()
            
            # Try to find synonyms from dictionary first
            synonyms = self.synonym_dict.get(word, [])
            
            # If no synonyms in dictionary, try word embeddings
            if not synonyms and self.word_embeddings:
                synonyms = self.find_similar_words(word)
            
            if synonyms:
                synonym = random.choice(synonyms)
                # Preserve original case
                if tokens[random_idx].isupper():
                    synonym = synonym.upper()
                elif tokens[random_idx].istitle():
                    synonym = synonym.capitalize()
                
                new_tokens[random_idx] = synonym
                num_replaced += 1
        
        return new_tokens
    
    def random_insertion(self, tokens: List[str], n: int = 1) -> List[str]:
        """
        Randomly insert n words into the sentence.
        
        Args:
            tokens: List of word tokens
            n: Number of words to insert
            
        Returns:
            Augmented token list
        """
        if len(tokens) == 0:
            return tokens
        
        new_tokens = tokens.copy()
        
        for _ in range(n):
            # Choose a random word to insert
            new_word = random.choice(self.common_words)
            
            # Choose a random position to insert
            random_idx = random.randint(0, len(new_tokens))
            new_tokens.insert(random_idx, new_word)
        
        return new_tokens
    
    def random_deletion(self, tokens: List[str], p: float = 0.1) -> List[str]:
        """
        Randomly delete words from the sentence with probability p.
        
        Args:
            tokens: List of word tokens
            p: Probability of deleting each word
            
        Returns:
            Augmented token list
        """
        if len(tokens) == 1:
            return tokens
        
        new_tokens = []
        for token in tokens:
            if random.random() > p:
                new_tokens.append(token)
        
        # If all words are deleted, return original
        if len(new_tokens) == 0:
            return tokens
        
        return new_tokens
    
    def random_swap(self, tokens: List[str], n: int = 1) -> List[str]:
        """
        Randomly swap two words in the sentence n times.
        
        Args:
            tokens: List of word tokens
            n: Number of swaps to perform
            
        Returns:
            Augmented token list
        """
        if len(tokens) < 2:
            return tokens
        
        new_tokens = tokens.copy()
        
        for _ in range(n):
            random_idx_1 = random.randint(0, len(new_tokens) - 1)
            random_idx_2 = random_idx_1
            counter = 0
            
            while random_idx_2 == random_idx_1:
                random_idx_2 = random.randint(0, len(new_tokens) - 1)
                counter += 1
                if counter > 3:
                    break
            
            new_tokens[random_idx_1], new_tokens[random_idx_2] = \
                new_tokens[random_idx_2], new_tokens[random_idx_1]
        
        return new_tokens
    
    def add_noise(self, tokens: List[str], noise_prob: float = 0.05) -> List[str]:
        """
        Add character-level noise to words.
        
        Args:
            tokens: List of word tokens
            noise_prob: Probability of adding noise to each character
            
        Returns:
            Augmented token list with noise
        """
        new_tokens = []
        
        for token in tokens:
            if len(token) <= 2:  # Don't modify very short words
                new_tokens.append(token)
                continue
            
            new_token = ""
            for char in token:
                if random.random() < noise_prob and char.isalpha():
                    # Random character substitution
                    if random.random() < 0.5:
                        # Replace with random letter
                        new_char = random.choice('abcdefghijklmnopqrstuvwxyz')
                        new_token += new_char
                    # Skip character (deletion)
                    # else: skip adding the character
                else:
                    new_token += char
            
            # Ensure we don't create empty tokens
            if new_token:
                new_tokens.append(new_token)
            else:
                new_tokens.append(token)
        
        return new_tokens
    
    def augment_text(self, text: str, num_augmentations: int = 1) -> List[str]:
        """
        Apply multiple augmentation techniques to generate augmented versions.
        
        Args:
            text: Input text string
            num_augmentations: Number of augmented versions to generate
            
        Returns:
            List of augmented text strings
        """
        # Simple tokenization (can be replaced with more sophisticated tokenizer)
        tokens = text.lower().split()
        
        if len(tokens) == 0:
            return [text] * num_augmentations
        
        augmented_texts = []
        
        for _ in range(num_augmentations):
            augmented_tokens = tokens.copy()
            
            # Randomly choose which augmentations to apply
            augmentation_methods = []
            
            if random.random() < self.synonym_prob:
                augmentation_methods.append('synonym')
            if random.random() < self.insert_prob and not self.preserve_length:
                augmentation_methods.append('insert')
            if random.random() < self.delete_prob and not self.preserve_length:
                augmentation_methods.append('delete')
            if random.random() < self.swap_prob:
                augmentation_methods.append('swap')
            
            # Apply selected augmentations
            for method in augmentation_methods:
                if method == 'synonym':
                    n_replace = max(1, int(len(augmented_tokens) * 0.1))
                    augmented_tokens = self.synonym_replacement(augmented_tokens, n_replace)
                elif method == 'insert':
                    n_insert = random.randint(1, max(1, len(augmented_tokens) // 10))
                    augmented_tokens = self.random_insertion(augmented_tokens, n_insert)
                elif method == 'delete':
                    augmented_tokens = self.random_deletion(augmented_tokens, 0.1)
                elif method == 'swap':
                    n_swap = random.randint(1, max(1, len(augmented_tokens) // 10))
                    augmented_tokens = self.random_swap(augmented_tokens, n_swap)
            
            # Add slight character noise occasionally
            if random.random() < 0.1:
                augmented_tokens = self.add_noise(augmented_tokens, 0.02)
            
            augmented_text = ' '.join(augmented_tokens)
            augmented_texts.append(augmented_text)
        
        return augmented_texts
    
    def validate_augmentation_quality(self, original: str, augmented: str, 
                                    min_similarity: float = 0.7) -> bool:
        """
        Validate the quality of augmented text.
        
        Args:
            original: Original text
            augmented: Augmented text
            min_similarity: Minimum similarity threshold
            
        Returns:
            True if augmentation quality is acceptable
        """
        # Simple similarity check based on word overlap
        original_words = set(original.lower().split())
        augmented_words = set(augmented.lower().split())
        
        if len(original_words) == 0:
            return len(augmented_words) == 0
        
        # Calculate Jaccard similarity
        intersection = len(original_words.intersection(augmented_words))
        union = len(original_words.union(augmented_words))
        
        similarity = intersection / union if union > 0 else 0
        
        # Additional checks
        length_ratio = len(augmented.split()) / len(original.split()) if len(original.split()) > 0 else 1
        
        # Accept if similarity is high enough and length is reasonable
        return (similarity >= min_similarity and 
                0.5 <= length_ratio <= 2.0 and
                len(augmented.strip()) > 0)
    
    def augment_dataset(self, texts: List[str], labels: List[int], 
                       augmentation_factor: float = 0.5) -> Tuple[List[str], List[int]]:
        """
        Augment an entire dataset with quality control.
        
        Args:
            texts: List of text strings
            labels: List of corresponding labels
            augmentation_factor: Fraction of dataset to augment
            
        Returns:
            Tuple of (augmented_texts, augmented_labels)
        """
        augmented_texts = texts.copy()
        augmented_labels = labels.copy()
        
        num_to_augment = int(len(texts) * augmentation_factor)
        indices_to_augment = random.sample(range(len(texts)), num_to_augment)
        
        for idx in indices_to_augment:
            original_text = texts[idx]
            original_label = labels[idx]
            
            # Generate augmented versions
            augmented_versions = self.augment_text(original_text, self.max_augmentations)
            
            # Validate and add high-quality augmentations
            for aug_text in augmented_versions:
                if self.validate_augmentation_quality(original_text, aug_text):
                    augmented_texts.append(aug_text)
                    augmented_labels.append(original_label)
        
        return augmented_texts, augmented_labels


class BackTranslationAugmenter:
    """
    Back-translation augmentation using translation APIs.
    
    Note: This is a simplified implementation. In production, you would
    use services like Google Translate API, Azure Translator, or local models.
    """
    
    def __init__(self, intermediate_languages: List[str] = None):
        """
        Initialize back-translation augmenter.
        
        Args:
            intermediate_languages: Languages to use for back-translation
        """
        self.intermediate_languages = intermediate_languages or ['es', 'fr', 'de', 'it']
        self.translation_cache = {}
    
    def back_translate(self, text: str, intermediate_lang: str = 'es') -> str:
        """
        Perform back-translation through an intermediate language.
        
        Args:
            text: Original text
            intermediate_lang: Intermediate language code
            
        Returns:
            Back-translated text
        """
        # This is a placeholder implementation
        # In practice, you would use actual translation services
        
        # Simple simulation of back-translation effects
        words = text.split()
        
        # Simulate translation artifacts
        if len(words) > 1:
            # Occasionally change word order (common in translation)
            if random.random() < 0.2:
                idx1, idx2 = random.sample(range(len(words)), 2)
                words[idx1], words[idx2] = words[idx2], words[idx1]
            
            # Simulate synonym substitution from translation
            for i in range(len(words)):
                if random.random() < 0.1:  # 10% chance
                    word = words[i].lower()
                    # Simple synonym mapping that might occur in translation
                    translation_synonyms = {
                        'good': 'nice', 'bad': 'poor', 'great': 'excellent',
                        'movie': 'film', 'show': 'display', 'see': 'watch',
                        'think': 'believe', 'feel': 'sense', 'know': 'understand'
                    }
                    if word in translation_synonyms:
                        words[i] = translation_synonyms[word]
        
        return ' '.join(words)
    
    def augment_text(self, text: str, num_augmentations: int = 1) -> List[str]:
        """
        Generate back-translated augmentations.
        
        Args:
            text: Original text
            num_augmentations: Number of augmentations to generate
            
        Returns:
            List of back-translated texts
        """
        augmented_texts = []
        
        for _ in range(num_augmentations):
            # Choose random intermediate language
            intermediate_lang = random.choice(self.intermediate_languages)
            
            # Perform back-translation
            back_translated = self.back_translate(text, intermediate_lang)
            augmented_texts.append(back_translated)
        
        return augmented_texts


def create_balanced_dataset(texts: List[str], labels: List[int], 
                          target_balance: float = 0.5) -> Tuple[List[str], List[int]]:
    """
    Create a balanced dataset by augmenting the minority class.
    
    Args:
        texts: List of text strings
        labels: List of corresponding labels
        target_balance: Target ratio for minority class (0.5 = perfectly balanced)
        
    Returns:
        Tuple of (balanced_texts, balanced_labels)
    """
    # Count class distribution
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    # Find minority and majority classes
    sorted_classes = sorted(label_counts.items(), key=lambda x: x[1])
    minority_class, minority_count = sorted_classes[0]
    majority_class, majority_count = sorted_classes[-1]
    
    # Calculate how many samples we need to add to achieve target balance
    # For target_balance = 0.5, we want minority_count + samples_to_add = majority_count
    # More generally: (minority_count + samples_to_add) / (total + samples_to_add) = target_balance
    # Solving: samples_to_add = (target_balance * total - minority_count) / (1 - target_balance)
    
    total_samples = len(texts)
    if target_balance >= 1.0 or target_balance <= 0.0:
        return texts, labels
    
    # Calculate required minority samples for target balance
    required_minority_samples = (target_balance * total_samples) / (1 - target_balance + target_balance)
    samples_to_add = max(0, int(required_minority_samples - minority_count))
    
    if samples_to_add == 0:
        return texts, labels
    
    # Find indices of minority class samples
    minority_indices = [i for i, label in enumerate(labels) if label == minority_class]
    
    # Create augmenter
    augmenter = TextAugmenter(
        synonym_prob=0.3,
        insert_prob=0.1,
        delete_prob=0.1,
        swap_prob=0.2,
        max_augmentations=1,
        preserve_length=True
    )
    
    # Generate augmented samples
    balanced_texts = texts.copy()
    balanced_labels = labels.copy()
    
    added_samples = 0
    max_attempts = samples_to_add * 3  # Prevent infinite loops
    attempts = 0
    
    while added_samples < samples_to_add and attempts < max_attempts:
        # Randomly select a minority class sample to augment
        source_idx = random.choice(minority_indices)
        source_text = texts[source_idx]
        
        # Generate augmented version
        augmented_versions = augmenter.augment_text(source_text, 1)
        
        for aug_text in augmented_versions:
            if augmenter.validate_augmentation_quality(source_text, aug_text):
                balanced_texts.append(aug_text)
                balanced_labels.append(minority_class)
                added_samples += 1
                break
        
        attempts += 1
    
    return balanced_texts, balanced_labels