"""
Advanced text preprocessing module for LSTM sentiment classifier.

This module provides enhanced preprocessing capabilities including advanced tokenization,
lemmatization, stemming, named entity recognition, and preprocessing pipeline comparison.
"""

import re
import torch
import string
from typing import List, Dict, Tuple, Optional, Set, Any
from collections import Counter, defaultdict
import unicodedata

# Try to import advanced NLP libraries (optional dependencies)
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.tag import pos_tag
    from nltk.chunk import ne_chunk
    from nltk.tree import Tree
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

from .text_preprocessor import TextPreprocessor


class AdvancedTextPreprocessor(TextPreprocessor):
    """
    Advanced text preprocessing class with enhanced NLP capabilities.
    
    Extends the base TextPreprocessor with advanced tokenization, lemmatization,
    stemming, named entity recognition, and other NLP features.
    """
    
    def __init__(self, 
                 max_vocab_size: int = 10000,
                 min_freq: int = 2,
                 max_length: int = 500,
                 use_lemmatization: bool = True,
                 use_stemming: bool = False,
                 remove_stopwords: bool = False,
                 handle_negation: bool = True,
                 preserve_entities: bool = True,
                 tokenizer_type: str = 'advanced'):
        """
        Initialize the AdvancedTextPreprocessor.
        
        Args:
            max_vocab_size: Maximum vocabulary size
            min_freq: Minimum frequency threshold for vocabulary inclusion
            max_length: Maximum sequence length for padding/truncation
            use_lemmatization: Whether to apply lemmatization
            use_stemming: Whether to apply stemming
            remove_stopwords: Whether to remove stopwords
            handle_negation: Whether to handle negation specially
            preserve_entities: Whether to preserve named entities
            tokenizer_type: Type of tokenizer ('basic', 'advanced', 'spacy', 'nltk')
        """
        super().__init__(max_vocab_size, min_freq, max_length)
        
        self.use_lemmatization = use_lemmatization
        self.use_stemming = use_stemming
        self.remove_stopwords = remove_stopwords
        self.handle_negation = handle_negation
        self.preserve_entities = preserve_entities
        self.tokenizer_type = tokenizer_type
        
        # Initialize NLP tools
        self.stemmer = None
        self.lemmatizer = None
        self.stopwords_set = set()
        self.nlp_model = None
        
        # Negation handling
        self.negation_words = {
            'not', 'no', 'never', 'nothing', 'nowhere', 'nobody', 'none',
            'neither', 'nor', 'cannot', 'cant', 'couldnt', 'shouldnt',
            'wouldnt', 'dont', 'doesnt', 'didnt', 'isnt', 'arent', 'wasnt',
            'werent', 'hasnt', 'havent', 'hadnt', 'wont', 'without'
        }
        
        # Initialize tools based on availability
        self._initialize_nlp_tools()
    
    def _initialize_nlp_tools(self):
        """Initialize NLP tools based on available libraries."""
        
        # Initialize NLTK tools if available
        if NLTK_AVAILABLE:
            try:
                # Download required NLTK data
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('wordnet', quiet=True)
                nltk.download('averaged_perceptron_tagger', quiet=True)
                nltk.download('maxent_ne_chunker', quiet=True)
                nltk.download('words', quiet=True)
                
                if self.use_stemming:
                    self.stemmer = PorterStemmer()
                
                if self.use_lemmatization:
                    self.lemmatizer = WordNetLemmatizer()
                
                if self.remove_stopwords:
                    self.stopwords_set = set(stopwords.words('english'))
                    
            except Exception as e:
                print(f"Warning: Could not initialize NLTK tools: {e}")
        
        # Initialize spaCy model if available
        if SPACY_AVAILABLE and self.tokenizer_type == 'spacy':
            try:
                # Try to load English model
                self.nlp_model = spacy.load('en_core_web_sm')
            except OSError:
                try:
                    # Fallback to basic English model
                    self.nlp_model = spacy.load('en')
                except OSError:
                    print("Warning: spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
                    self.tokenizer_type = 'advanced'
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove or replace special characters
        text = re.sub(r'<[^>]+>', ' ', text)  # Remove HTML tags
        text = re.sub(r'http\S+|www\S+', ' URL ', text)  # Replace URLs
        text = re.sub(r'@\w+', ' MENTION ', text)  # Replace mentions
        text = re.sub(r'#\w+', ' HASHTAG ', text)  # Replace hashtags
        
        # Handle contractions more comprehensively
        contractions = {
            r"won't": "will not",
            r"can't": "cannot",
            r"n't": " not",
            r"'re": " are",
            r"'ve": " have",
            r"'ll": " will",
            r"'d": " would",
            r"'m": " am",
            r"let's": "let us",
            r"that's": "that is",
            r"who's": "who is",
            r"what's": "what is",
            r"where's": "where is",
            r"how's": "how is",
            r"it's": "it is",
            r"he's": "he is",
            r"she's": "she is"
        }
        
        for contraction, expansion in contractions.items():
            text = re.sub(contraction, expansion, text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def tokenize_advanced(self, text: str) -> List[str]:
        """
        Advanced tokenization with multiple strategies.
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens
        """
        if not isinstance(text, str) or not text.strip():
            return []
        
        # Clean text first
        text = self.clean_text(text)
        
        if self.tokenizer_type == 'spacy' and self.nlp_model:
            return self._tokenize_spacy(text)
        elif self.tokenizer_type == 'nltk' and NLTK_AVAILABLE:
            return self._tokenize_nltk(text)
        else:
            return self._tokenize_advanced_regex(text)
    
    def _tokenize_spacy(self, text: str) -> List[str]:
        """Tokenize using spaCy."""
        doc = self.nlp_model(text)
        tokens = []
        
        for token in doc:
            # Skip whitespace and punctuation if desired
            if token.is_space:
                continue
            
            # Handle named entities if preserve_entities is True
            if self.preserve_entities and token.ent_type_:
                tokens.append(f"ENT_{token.ent_type_}")
            else:
                # Apply lemmatization or stemming
                if self.use_lemmatization:
                    tokens.append(token.lemma_)
                elif self.use_stemming and self.stemmer:
                    tokens.append(self.stemmer.stem(token.text))
                else:
                    tokens.append(token.text)
        
        return self._post_process_tokens(tokens)
    
    def _tokenize_nltk(self, text: str) -> List[str]:
        """Tokenize using NLTK."""
        tokens = word_tokenize(text)
        
        # Handle named entities
        if self.preserve_entities:
            tokens = self._handle_named_entities_nltk(tokens)
        
        # Apply stemming or lemmatization
        processed_tokens = []
        for token in tokens:
            if token.isalpha():  # Only process alphabetic tokens
                if self.use_lemmatization and self.lemmatizer:
                    processed_tokens.append(self.lemmatizer.lemmatize(token))
                elif self.use_stemming and self.stemmer:
                    processed_tokens.append(self.stemmer.stem(token))
                else:
                    processed_tokens.append(token)
            else:
                processed_tokens.append(token)
        
        return self._post_process_tokens(processed_tokens)
    
    def _tokenize_advanced_regex(self, text: str) -> List[str]:
        """Advanced regex-based tokenization."""
        # Enhanced regex pattern for better tokenization
        pattern = r'''
            (?:[A-Z]\.)+                    # Abbreviations (U.S.A., etc.)
            |(?:\$?\d+(?:\.\d+)?%?)         # Numbers, currency, percentages
            |(?:\w+(?:[-']\w+)*)            # Words with hyphens and apostrophes
            |(?:[.!?]+)                     # Sentence endings
            |(?:[,;:])                      # Other punctuation
            |(?:\S)                         # Any other non-whitespace
        '''
        
        tokens = re.findall(pattern, text, re.VERBOSE)
        
        # Apply basic stemming/lemmatization if tools are available
        processed_tokens = []
        for token in tokens:
            if token.isalpha():
                if self.use_lemmatization and self.lemmatizer:
                    processed_tokens.append(self.lemmatizer.lemmatize(token))
                elif self.use_stemming and self.stemmer:
                    processed_tokens.append(self.stemmer.stem(token))
                else:
                    processed_tokens.append(token)
            else:
                processed_tokens.append(token)
        
        return self._post_process_tokens(processed_tokens)
    
    def _handle_named_entities_nltk(self, tokens: List[str]) -> List[str]:
        """Handle named entities using NLTK."""
        try:
            # POS tagging
            pos_tags = pos_tag(tokens)
            
            # Named entity chunking
            chunks = ne_chunk(pos_tags)
            
            processed_tokens = []
            for chunk in chunks:
                if isinstance(chunk, Tree):
                    # This is a named entity
                    entity_type = chunk.label()
                    processed_tokens.append(f"ENT_{entity_type}")
                else:
                    # Regular token
                    processed_tokens.append(chunk[0])
            
            return processed_tokens
        except Exception:
            # Fallback to original tokens if NER fails
            return tokens
    
    def _post_process_tokens(self, tokens: List[str]) -> List[str]:
        """Post-process tokens (remove stopwords, handle negation, etc.)."""
        processed_tokens = []
        
        for i, token in enumerate(tokens):
            # Skip empty tokens
            if not token.strip():
                continue
            
            # Remove stopwords if requested (but preserve negation context)
            if self.remove_stopwords and token.lower() in self.stopwords_set:
                # Don't remove if it's a negation word or follows a negation
                if not (token.lower() in self.negation_words or 
                       (i > 0 and tokens[i-1].lower() in self.negation_words)):
                    continue
            
            # Handle negation
            if self.handle_negation and i > 0 and tokens[i-1].lower() in self.negation_words:
                processed_tokens.append(f"NOT_{token}")
            else:
                processed_tokens.append(token)
        
        return processed_tokens
    
    def tokenize(self, text: str) -> List[str]:
        """
        Main tokenization method (overrides parent class).
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens
        """
        if self.tokenizer_type in ['advanced', 'spacy', 'nltk']:
            return self.tokenize_advanced(text)
        else:
            # Fallback to parent class method
            return super().tokenize(text)
    
    def extract_features(self, text: str) -> Dict[str, Any]:
        """
        Extract various text features for analysis.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Basic statistics
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(re.split(r'[.!?]+', text))
        
        # Tokenize for further analysis
        tokens = self.tokenize(text)
        features['token_count'] = len(tokens)
        
        # Lexical diversity
        unique_tokens = set(tokens)
        features['unique_token_count'] = len(unique_tokens)
        features['lexical_diversity'] = len(unique_tokens) / len(tokens) if tokens else 0
        
        # Punctuation and capitalization
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['uppercase_count'] = sum(1 for c in text if c.isupper())
        features['uppercase_ratio'] = features['uppercase_count'] / len(text) if text else 0
        
        # Sentiment-related features
        features['negation_count'] = sum(1 for token in tokens 
                                       if token.lower() in self.negation_words)
        
        # Average word length
        word_lengths = [len(token) for token in tokens if token.isalpha()]
        features['avg_word_length'] = sum(word_lengths) / len(word_lengths) if word_lengths else 0
        
        return features
    
    def compare_preprocessing_methods(self, texts: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Compare different preprocessing methods on a sample of texts.
        
        Args:
            texts: List of sample texts
            
        Returns:
            Dictionary comparing different preprocessing approaches
        """
        methods = {
            'basic': TextPreprocessor(),
            'advanced_regex': AdvancedTextPreprocessor(tokenizer_type='advanced'),
        }
        
        if NLTK_AVAILABLE:
            methods['nltk'] = AdvancedTextPreprocessor(tokenizer_type='nltk')
        
        if SPACY_AVAILABLE and self.nlp_model:
            methods['spacy'] = AdvancedTextPreprocessor(tokenizer_type='spacy')
        
        comparison = {}
        
        for method_name, preprocessor in methods.items():
            method_stats = {
                'total_tokens': 0,
                'unique_tokens': set(),
                'avg_tokens_per_text': 0,
                'processing_time': 0
            }
            
            import time
            start_time = time.time()
            
            for text in texts[:100]:  # Limit to first 100 texts for comparison
                tokens = preprocessor.tokenize(text)
                method_stats['total_tokens'] += len(tokens)
                method_stats['unique_tokens'].update(tokens)
            
            method_stats['processing_time'] = time.time() - start_time
            method_stats['avg_tokens_per_text'] = method_stats['total_tokens'] / min(len(texts), 100)
            method_stats['unique_token_count'] = len(method_stats['unique_tokens'])
            
            # Remove the set for JSON serialization
            del method_stats['unique_tokens']
            
            comparison[method_name] = method_stats
        
        return comparison
    
    def get_preprocessing_stats(self, texts: List[str]) -> Dict[str, Any]:
        """
        Get comprehensive preprocessing statistics.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Dictionary of preprocessing statistics
        """
        stats = {
            'total_texts': len(texts),
            'total_characters': sum(len(text) for text in texts),
            'total_words': sum(len(text.split()) for text in texts),
            'avg_text_length': 0,
            'avg_word_count': 0,
            'vocabulary_size': 0,
            'token_distribution': Counter(),
            'feature_summary': {}
        }
        
        if texts:
            stats['avg_text_length'] = stats['total_characters'] / len(texts)
            stats['avg_word_count'] = stats['total_words'] / len(texts)
        
        # Analyze a sample for detailed statistics
        sample_size = min(1000, len(texts))
        sample_texts = texts[:sample_size]
        
        all_tokens = []
        all_features = []
        
        for text in sample_texts:
            tokens = self.tokenize(text)
            all_tokens.extend(tokens)
            
            features = self.extract_features(text)
            all_features.append(features)
        
        stats['vocabulary_size'] = len(set(all_tokens))
        stats['token_distribution'] = Counter(all_tokens)
        
        # Aggregate feature statistics
        if all_features:
            feature_keys = all_features[0].keys()
            for key in feature_keys:
                values = [f[key] for f in all_features]
                stats['feature_summary'][key] = {
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values)
                }
        
        return stats


def create_preprocessing_pipeline(config: Dict[str, Any]) -> AdvancedTextPreprocessor:
    """
    Create a preprocessing pipeline based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured AdvancedTextPreprocessor instance
    """
    return AdvancedTextPreprocessor(
        max_vocab_size=config.get('max_vocab_size', 10000),
        min_freq=config.get('min_freq', 2),
        max_length=config.get('max_length', 500),
        use_lemmatization=config.get('use_lemmatization', True),
        use_stemming=config.get('use_stemming', False),
        remove_stopwords=config.get('remove_stopwords', False),
        handle_negation=config.get('handle_negation', True),
        preserve_entities=config.get('preserve_entities', True),
        tokenizer_type=config.get('tokenizer_type', 'advanced')
    )


def evaluate_preprocessing_pipeline(preprocessor: AdvancedTextPreprocessor,
                                  texts: List[str],
                                  labels: List[int]) -> Dict[str, Any]:
    """
    Evaluate the effectiveness of a preprocessing pipeline.
    
    Args:
        preprocessor: The preprocessor to evaluate
        texts: List of texts
        labels: List of corresponding labels
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Process texts
    processed_sequences = []
    for text in texts:
        tokens = preprocessor.tokenize(text)
        sequence = preprocessor.text_to_sequence(text)
        processed_sequences.append(sequence)
    
    # Calculate metrics
    vocab_info = preprocessor.get_vocab_info()
    
    evaluation = {
        'vocabulary_size': vocab_info['vocab_size'],
        'avg_sequence_length': sum(len(seq) for seq in processed_sequences) / len(processed_sequences),
        'max_sequence_length': max(len(seq) for seq in processed_sequences),
        'min_sequence_length': min(len(seq) for seq in processed_sequences),
        'sequences_at_max_length': sum(1 for seq in processed_sequences 
                                     if len(seq) >= preprocessor.max_length),
        'empty_sequences': sum(1 for seq in processed_sequences if len(seq) == 0),
        'preprocessing_config': {
            'use_lemmatization': preprocessor.use_lemmatization,
            'use_stemming': preprocessor.use_stemming,
            'remove_stopwords': preprocessor.remove_stopwords,
            'handle_negation': preprocessor.handle_negation,
            'preserve_entities': preprocessor.preserve_entities,
            'tokenizer_type': preprocessor.tokenizer_type
        }
    }
    
    return evaluation