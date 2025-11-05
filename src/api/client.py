"""
Client Library for LSTM Sentiment Classifier API

This module provides a Python client for interacting with the sentiment
analysis API, including error handling, retries, and batch processing.
"""

import requests
import time
import logging
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import json


@dataclass
class SentimentResult:
    """Result of sentiment prediction."""
    sentiment: str
    confidence: float
    probability: Optional[float] = None
    processing_time_ms: Optional[float] = None


@dataclass
class BatchSentimentResult:
    """Result of batch sentiment prediction."""
    predictions: List[SentimentResult]
    statistics: Optional[Dict] = None
    total_processing_time_ms: Optional[float] = None


@dataclass
class HealthStatus:
    """API health status."""
    status: str
    timestamp: str
    model_loaded: bool
    uptime_seconds: float
    memory_usage_mb: float
    cpu_usage_percent: float


@dataclass
class APIMetrics:
    """API performance metrics."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time_ms: float
    cache_hit_rate: float
    uptime_seconds: float
    memory_usage_mb: float
    cpu_usage_percent: float


class SentimentAPIClient:
    """
    Client for the LSTM Sentiment Classifier API.
    
    Provides methods for sentiment prediction, health checks, and metrics
    with automatic retries, error handling, and connection management.
    """
    
    def __init__(self,
                 base_url: str = "http://localhost:8000",
                 timeout: float = 30.0,
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 enable_logging: bool = True):
        """
        Initialize the API client.
        
        Args:
            base_url: Base URL of the API server
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            enable_logging: Whether to enable request logging
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Set up session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'SentimentAPIClient/1.0'
        })
        
        # Set up logging
        if enable_logging:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.disabled = True
    
    def _make_request(self,
                     method: str,
                     endpoint: str,
                     data: Optional[Dict] = None,
                     params: Optional[Dict] = None) -> requests.Response:
        """
        Make HTTP request with retries and error handling.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            data: Request data for POST requests
            params: Query parameters for GET requests
            
        Returns:
            Response object
            
        Raises:
            requests.RequestException: If request fails after all retries
        """
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(self.max_retries + 1):
            try:
                if method.upper() == 'GET':
                    response = self.session.get(url, params=params, timeout=self.timeout)
                elif method.upper() == 'POST':
                    response = self.session.post(url, json=data, timeout=self.timeout)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                # Check for HTTP errors
                response.raise_for_status()
                
                self.logger.debug(f"Request successful: {method} {url}")
                return response
                
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    self.logger.error(f"Request failed after {self.max_retries + 1} attempts")
                    raise
    
    def predict_sentiment(self,
                         text: str,
                         threshold: float = 0.5,
                         include_probability: bool = False) -> SentimentResult:
        """
        Predict sentiment for a single text.
        
        Args:
            text: Text to analyze
            threshold: Classification threshold (0.0-1.0)
            include_probability: Whether to include raw probability
            
        Returns:
            Sentiment prediction result
            
        Raises:
            ValueError: If input validation fails
            requests.RequestException: If API request fails
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text must be a non-empty string")
        
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        
        data = {
            'text': text.strip(),
            'threshold': threshold,
            'include_probability': include_probability
        }
        
        self.logger.info(f"Predicting sentiment for text: {text[:50]}...")
        
        response = self._make_request('POST', '/predict', data=data)
        result_data = response.json()
        
        return SentimentResult(
            sentiment=result_data['sentiment'],
            confidence=result_data['confidence'],
            probability=result_data.get('probability'),
            processing_time_ms=result_data.get('processing_time_ms')
        )
    
    def predict_batch_sentiment(self,
                               texts: List[str],
                               threshold: float = 0.5,
                               include_probability: bool = False,
                               include_statistics: bool = False) -> BatchSentimentResult:
        """
        Predict sentiment for multiple texts.
        
        Args:
            texts: List of texts to analyze
            threshold: Classification threshold (0.0-1.0)
            include_probability: Whether to include raw probabilities
            include_statistics: Whether to include batch statistics
            
        Returns:
            Batch sentiment prediction result
            
        Raises:
            ValueError: If input validation fails
            requests.RequestException: If API request fails
        """
        if not isinstance(texts, list) or not texts:
            raise ValueError("Texts must be a non-empty list")
        
        if len(texts) > 100:
            raise ValueError("Maximum batch size is 100 texts")
        
        # Validate individual texts
        cleaned_texts = []
        for i, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Text at index {i} must be a non-empty string")
            cleaned_texts.append(text.strip())
        
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        
        data = {
            'texts': cleaned_texts,
            'threshold': threshold,
            'include_probability': include_probability,
            'include_statistics': include_statistics
        }
        
        self.logger.info(f"Predicting sentiment for {len(texts)} texts")
        
        response = self._make_request('POST', '/predict/batch', data=data)
        result_data = response.json()
        
        # Parse predictions
        predictions = []
        for pred_data in result_data['predictions']:
            predictions.append(SentimentResult(
                sentiment=pred_data['sentiment'],
                confidence=pred_data['confidence'],
                probability=pred_data.get('probability'),
                processing_time_ms=pred_data.get('processing_time_ms')
            ))
        
        return BatchSentimentResult(
            predictions=predictions,
            statistics=result_data.get('statistics'),
            total_processing_time_ms=result_data.get('total_processing_time_ms')
        )
    
    def get_health_status(self) -> HealthStatus:
        """
        Get API health status.
        
        Returns:
            Health status information
            
        Raises:
            requests.RequestException: If API request fails
        """
        self.logger.debug("Getting health status")
        
        response = self._make_request('GET', '/health')
        data = response.json()
        
        return HealthStatus(
            status=data['status'],
            timestamp=data['timestamp'],
            model_loaded=data['model_loaded'],
            uptime_seconds=data['uptime_seconds'],
            memory_usage_mb=data['memory_usage_mb'],
            cpu_usage_percent=data['cpu_usage_percent']
        )
    
    def get_metrics(self) -> APIMetrics:
        """
        Get API performance metrics.
        
        Returns:
            Performance metrics
            
        Raises:
            requests.RequestException: If API request fails
        """
        self.logger.debug("Getting performance metrics")
        
        response = self._make_request('GET', '/metrics')
        data = response.json()
        
        return APIMetrics(
            total_requests=data['total_requests'],
            successful_requests=data['successful_requests'],
            failed_requests=data['failed_requests'],
            average_response_time_ms=data['average_response_time_ms'],
            cache_hit_rate=data['cache_hit_rate'],
            uptime_seconds=data['uptime_seconds'],
            memory_usage_mb=data['memory_usage_mb'],
            cpu_usage_percent=data['cpu_usage_percent']
        )
    
    def get_model_info(self) -> Dict:
        """
        Get model information.
        
        Returns:
            Model configuration and statistics
            
        Raises:
            requests.RequestException: If API request fails
        """
        self.logger.debug("Getting model information")
        
        response = self._make_request('GET', '/model/info')
        return response.json()
    
    def clear_cache(self) -> bool:
        """
        Clear prediction cache.
        
        Returns:
            True if cache was cleared successfully
            
        Raises:
            requests.RequestException: If API request fails
        """
        self.logger.info("Clearing prediction cache")
        
        response = self._make_request('POST', '/cache/clear')
        return response.status_code == 200
    
    def is_healthy(self) -> bool:
        """
        Check if API is healthy and ready to serve requests.
        
        Returns:
            True if API is healthy, False otherwise
        """
        try:
            health = self.get_health_status()
            return health.status == "healthy" and health.model_loaded
        except Exception:
            return False
    
    def wait_for_ready(self, timeout: float = 60.0, check_interval: float = 2.0) -> bool:
        """
        Wait for API to become ready.
        
        Args:
            timeout: Maximum time to wait in seconds
            check_interval: Time between health checks in seconds
            
        Returns:
            True if API becomes ready, False if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.is_healthy():
                self.logger.info("API is ready")
                return True
            
            self.logger.debug("Waiting for API to become ready...")
            time.sleep(check_interval)
        
        self.logger.warning(f"API not ready after {timeout} seconds")
        return False
    
    def close(self):
        """Close the client session."""
        self.session.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Convenience functions
def predict_sentiment(text: str,
                     api_url: str = "http://localhost:8000",
                     threshold: float = 0.5,
                     include_probability: bool = False) -> SentimentResult:
    """
    Convenience function for single sentiment prediction.
    
    Args:
        text: Text to analyze
        api_url: API server URL
        threshold: Classification threshold
        include_probability: Whether to include raw probability
        
    Returns:
        Sentiment prediction result
    """
    with SentimentAPIClient(api_url) as client:
        return client.predict_sentiment(text, threshold, include_probability)


def predict_batch_sentiment(texts: List[str],
                           api_url: str = "http://localhost:8000",
                           threshold: float = 0.5,
                           include_probability: bool = False,
                           include_statistics: bool = False) -> BatchSentimentResult:
    """
    Convenience function for batch sentiment prediction.
    
    Args:
        texts: List of texts to analyze
        api_url: API server URL
        threshold: Classification threshold
        include_probability: Whether to include raw probabilities
        include_statistics: Whether to include batch statistics
        
    Returns:
        Batch sentiment prediction result
    """
    with SentimentAPIClient(api_url) as client:
        return client.predict_batch_sentiment(texts, threshold, include_probability, include_statistics)


def check_api_health(api_url: str = "http://localhost:8000") -> bool:
    """
    Convenience function to check API health.
    
    Args:
        api_url: API server URL
        
    Returns:
        True if API is healthy, False otherwise
    """
    with SentimentAPIClient(api_url) as client:
        return client.is_healthy()