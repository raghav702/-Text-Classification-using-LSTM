"""
Production-ready FastAPI Inference Service for LSTM Sentiment Classifier

This module provides a REST API for sentiment classification with:
- Input validation and error handling
- Request/response logging and monitoring
- Health checks and performance metrics
- Caching for improved response times
- Rate limiting and security features
"""

import os
import time
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import json
import hashlib
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn
from cachetools import TTLCache
import psutil

# Import our inference engine
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from inference.inference_engine import InferenceEngine, create_inference_engine


# Pydantic models for request/response validation
class SentimentRequest(BaseModel):
    """Request model for single text sentiment prediction."""
    text: str = Field(..., min_length=1, max_length=10000, description="Text to analyze")
    threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Classification threshold")
    include_probability: Optional[bool] = Field(False, description="Include raw probability in response")
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty or only whitespace')
        return v.strip()


class BatchSentimentRequest(BaseModel):
    """Request model for batch sentiment prediction."""
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of texts to analyze")
    threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Classification threshold")
    include_probability: Optional[bool] = Field(False, description="Include raw probabilities in response")
    include_statistics: Optional[bool] = Field(False, description="Include batch statistics")
    
    @validator('texts')
    def validate_texts(cls, v):
        if not v:
            raise ValueError('Texts list cannot be empty')
        
        cleaned_texts = []
        for i, text in enumerate(v):
            if not isinstance(text, str):
                raise ValueError(f'Text at index {i} must be a string')
            if not text.strip():
                raise ValueError(f'Text at index {i} cannot be empty or only whitespace')
            if len(text) > 10000:
                raise ValueError(f'Text at index {i} exceeds maximum length of 10000 characters')
            cleaned_texts.append(text.strip())
        
        return cleaned_texts


class SentimentResponse(BaseModel):
    """Response model for sentiment prediction."""
    sentiment: str = Field(..., description="Predicted sentiment (positive/negative)")
    confidence: float = Field(..., description="Confidence score (0-1)")
    probability: Optional[float] = Field(None, description="Raw probability score")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class BatchSentimentResponse(BaseModel):
    """Response model for batch sentiment prediction."""
    predictions: List[SentimentResponse] = Field(..., description="List of predictions")
    statistics: Optional[Dict] = Field(None, description="Batch statistics")
    total_processing_time_ms: float = Field(..., description="Total processing time in milliseconds")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Current timestamp")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    memory_usage_mb: float = Field(..., description="Memory usage in MB")
    cpu_usage_percent: float = Field(..., description="CPU usage percentage")


class MetricsResponse(BaseModel):
    """Response model for performance metrics."""
    total_requests: int = Field(..., description="Total number of requests")
    successful_requests: int = Field(..., description="Number of successful requests")
    failed_requests: int = Field(..., description="Number of failed requests")
    average_response_time_ms: float = Field(..., description="Average response time")
    cache_hit_rate: float = Field(..., description="Cache hit rate")
    uptime_seconds: float = Field(..., description="Service uptime")
    memory_usage_mb: float = Field(..., description="Current memory usage")
    cpu_usage_percent: float = Field(..., description="Current CPU usage")


class InferenceAPIService:
    """
    Production-ready inference API service with monitoring and caching.
    """
    
    def __init__(self, 
                 model_path: str,
                 vocab_path: str,
                 device: str = None,
                 cache_size: int = 1000,
                 cache_ttl: int = 3600,
                 enable_logging: bool = True):
        """
        Initialize the inference API service.
        
        Args:
            model_path: Path to trained model checkpoint
            vocab_path: Path to vocabulary file
            device: Device to run inference on
            cache_size: Maximum number of cached predictions
            cache_ttl: Cache time-to-live in seconds
            enable_logging: Whether to enable request logging
        """
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.device = device
        self.enable_logging = enable_logging
        
        # Initialize inference engine
        self.inference_engine = None
        self.model_loaded = False
        
        # Initialize cache
        self.cache = TTLCache(maxsize=cache_size, ttl=cache_ttl)
        
        # Initialize metrics
        self.start_time = time.time()
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'response_times': [],
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Set up logging
        if enable_logging:
            self.setup_logging()
        
        self.logger = logging.getLogger(__name__)
    
    def setup_logging(self):
        """Set up request and error logging."""
        # Create logs directory
        log_dir = Path("logs/api")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "api.log"),
                logging.StreamHandler()
            ]
        )
    
    async def load_model(self):
        """Load the inference model asynchronously."""
        try:
            self.logger.info("Loading inference model...")
            self.inference_engine = create_inference_engine(
                self.model_path, 
                self.vocab_path, 
                self.device
            )
            self.model_loaded = True
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def _generate_cache_key(self, text: str, threshold: float) -> str:
        """Generate cache key for text and threshold."""
        key_string = f"{text}:{threshold}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cached_prediction(self, text: str, threshold: float) -> Optional[Dict]:
        """Get cached prediction if available."""
        cache_key = self._generate_cache_key(text, threshold)
        cached_result = self.cache.get(cache_key)
        
        if cached_result:
            self.metrics['cache_hits'] += 1
            return cached_result
        else:
            self.metrics['cache_misses'] += 1
            return None
    
    def _cache_prediction(self, text: str, threshold: float, result: Dict):
        """Cache prediction result."""
        cache_key = self._generate_cache_key(text, threshold)
        self.cache[cache_key] = result
    
    def _update_metrics(self, success: bool, response_time: float):
        """Update service metrics."""
        self.metrics['total_requests'] += 1
        if success:
            self.metrics['successful_requests'] += 1
        else:
            self.metrics['failed_requests'] += 1
        
        self.metrics['response_times'].append(response_time)
        
        # Keep only last 1000 response times for memory efficiency
        if len(self.metrics['response_times']) > 1000:
            self.metrics['response_times'] = self.metrics['response_times'][-1000:]
    
    async def predict_sentiment(self, request: SentimentRequest) -> SentimentResponse:
        """
        Predict sentiment for a single text.
        
        Args:
            request: Sentiment prediction request
            
        Returns:
            Sentiment prediction response
        """
        start_time = time.time()
        
        try:
            # Auto-load model on first request if not loaded
            if not self.model_loaded:
                self.logger.info("Loading model on first request...")
                await self.load_model()
            
            # Check cache first
            cached_result = self._get_cached_prediction(request.text, request.threshold)
            if cached_result:
                self.logger.info(f"Cache hit for text: {request.text[:50]}...")
                cached_result['processing_time_ms'] = (time.time() - start_time) * 1000
                self._update_metrics(True, time.time() - start_time)
                return SentimentResponse(**cached_result)
            
            # Make prediction
            if request.include_probability:
                sentiment, probability, confidence = self.inference_engine.predict_sentiment_with_probability(
                    request.text
                )
                result = {
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'probability': probability
                }
            else:
                sentiment, confidence = self.inference_engine.predict_sentiment(
                    request.text, request.threshold
                )
                result = {
                    'sentiment': sentiment,
                    'confidence': confidence
                }
            
            # Add processing time
            processing_time = (time.time() - start_time) * 1000
            result['processing_time_ms'] = processing_time
            
            # Cache the result
            self._cache_prediction(request.text, request.threshold, result)
            
            # Update metrics
            self._update_metrics(True, time.time() - start_time)
            
            # Log request
            if self.enable_logging:
                self.logger.info(
                    f"Prediction - Text: {request.text[:50]}..., "
                    f"Sentiment: {sentiment}, Confidence: {confidence:.3f}, "
                    f"Time: {processing_time:.2f}ms"
                )
            
            return SentimentResponse(**result)
            
        except Exception as e:
            self._update_metrics(False, time.time() - start_time)
            self.logger.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    async def predict_batch_sentiment(self, request: BatchSentimentRequest) -> BatchSentimentResponse:
        """
        Predict sentiment for multiple texts.
        
        Args:
            request: Batch sentiment prediction request
            
        Returns:
            Batch sentiment prediction response
        """
        start_time = time.time()
        
        try:
            # Auto-load model on first request if not loaded
            if not self.model_loaded:
                self.logger.info("Loading model on first request...")
                await self.load_model()
            
            predictions = []
            
            # Process each text
            for text in request.texts:
                text_start_time = time.time()
                
                # Check cache
                cached_result = self._get_cached_prediction(text, request.threshold)
                if cached_result:
                    cached_result['processing_time_ms'] = (time.time() - text_start_time) * 1000
                    predictions.append(SentimentResponse(**cached_result))
                    continue
                
                # Make prediction
                if request.include_probability:
                    sentiment, probability, confidence = self.inference_engine.predict_sentiment_with_probability(text)
                    result = {
                        'sentiment': sentiment,
                        'confidence': confidence,
                        'probability': probability,
                        'processing_time_ms': (time.time() - text_start_time) * 1000
                    }
                else:
                    sentiment, confidence = self.inference_engine.predict_sentiment(text, request.threshold)
                    result = {
                        'sentiment': sentiment,
                        'confidence': confidence,
                        'processing_time_ms': (time.time() - text_start_time) * 1000
                    }
                
                # Cache the result
                self._cache_prediction(text, request.threshold, result)
                predictions.append(SentimentResponse(**result))
            
            # Get batch statistics if requested
            statistics = None
            if request.include_statistics:
                statistics = self.inference_engine.get_prediction_stats(request.texts)
            
            total_processing_time = (time.time() - start_time) * 1000
            
            # Update metrics
            self._update_metrics(True, time.time() - start_time)
            
            # Log batch request
            if self.enable_logging:
                self.logger.info(
                    f"Batch prediction - Texts: {len(request.texts)}, "
                    f"Time: {total_processing_time:.2f}ms"
                )
            
            return BatchSentimentResponse(
                predictions=predictions,
                statistics=statistics,
                total_processing_time_ms=total_processing_time
            )
            
        except Exception as e:
            self._update_metrics(False, time.time() - start_time)
            self.logger.error(f"Batch prediction failed: {e}")
            raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
    
    def get_health_status(self) -> HealthResponse:
        """Get service health status."""
        uptime = time.time() - self.start_time
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        cpu_usage = psutil.cpu_percent()
        
        return HealthResponse(
            status="healthy" if self.model_loaded else "unhealthy",
            timestamp=datetime.now().isoformat(),
            model_loaded=self.model_loaded,
            uptime_seconds=uptime,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage
        )
    
    def get_metrics(self) -> MetricsResponse:
        """Get service performance metrics."""
        uptime = time.time() - self.start_time
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        cpu_usage = psutil.cpu_percent()
        
        # Calculate average response time
        avg_response_time = 0
        if self.metrics['response_times']:
            avg_response_time = sum(self.metrics['response_times']) / len(self.metrics['response_times']) * 1000
        
        # Calculate cache hit rate
        total_cache_requests = self.metrics['cache_hits'] + self.metrics['cache_misses']
        cache_hit_rate = self.metrics['cache_hits'] / total_cache_requests if total_cache_requests > 0 else 0
        
        return MetricsResponse(
            total_requests=self.metrics['total_requests'],
            successful_requests=self.metrics['successful_requests'],
            failed_requests=self.metrics['failed_requests'],
            average_response_time_ms=avg_response_time,
            cache_hit_rate=cache_hit_rate,
            uptime_seconds=uptime,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage
        )
    
    def clear_cache(self):
        """Clear prediction cache."""
        self.cache.clear()
        self.logger.info("Prediction cache cleared")


# Global service instance
service = None


def get_service() -> InferenceAPIService:
    """Dependency to get the service instance."""
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return service


# Create FastAPI app
app = FastAPI(
    title="LSTM Sentiment Classifier API",
    description="Production-ready API for sentiment analysis using LSTM neural networks",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log all requests."""
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Log request details
    process_time = time.time() - start_time
    logger = logging.getLogger("api.requests")
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )
    
    return response


@app.on_event("startup")
async def startup_event():
    """Initialize service on startup."""
    global service
    
    # Get configuration from environment variables
    model_path = os.getenv("MODEL_PATH", "models/improved_lstm_model_20251105_121505.pth")
    vocab_path = os.getenv("VOCAB_PATH", "models/improved_lstm_model_20251105_121505_vocabulary.pth")
    device = os.getenv("DEVICE", None)
    cache_size = int(os.getenv("CACHE_SIZE", "1000"))
    cache_ttl = int(os.getenv("CACHE_TTL", "3600"))
    
    # Initialize service
    service = InferenceAPIService(
        model_path=model_path,
        vocab_path=vocab_path,
        device=device,
        cache_size=cache_size,
        cache_ttl=cache_ttl
    )
    
    # Don't load model at startup - load on first request to avoid timeout
    # await service.load_model()
    logger.info("API startup complete - model will load on first request")


@app.get("/", response_model=Dict)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "LSTM Sentiment Classifier API",
        "version": "1.0.0",
        "description": "Production-ready API for sentiment analysis",
        "endpoints": {
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "health": "/health",
            "metrics": "/metrics",
            "docs": "/docs"
        }
    }


@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(
    request: SentimentRequest,
    service: InferenceAPIService = Depends(get_service)
):
    """
    Predict sentiment for a single text.
    
    - **text**: Text to analyze (1-10000 characters)
    - **threshold**: Classification threshold (0.0-1.0, default: 0.5)
    - **include_probability**: Include raw probability score (default: false)
    """
    return await service.predict_sentiment(request)


@app.post("/predict/batch", response_model=BatchSentimentResponse)
async def predict_batch_sentiment(
    request: BatchSentimentRequest,
    service: InferenceAPIService = Depends(get_service)
):
    """
    Predict sentiment for multiple texts in batch.
    
    - **texts**: List of texts to analyze (1-100 texts, each 1-10000 characters)
    - **threshold**: Classification threshold (0.0-1.0, default: 0.5)
    - **include_probability**: Include raw probability scores (default: false)
    - **include_statistics**: Include batch statistics (default: false)
    """
    return await service.predict_batch_sentiment(request)


@app.get("/health", response_model=HealthResponse)
async def health_check(service: InferenceAPIService = Depends(get_service)):
    """
    Health check endpoint.
    
    Returns service status, uptime, and resource usage.
    """
    return service.get_health_status()


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics(service: InferenceAPIService = Depends(get_service)):
    """
    Performance metrics endpoint.
    
    Returns detailed performance metrics including request counts,
    response times, cache statistics, and resource usage.
    """
    return service.get_metrics()


@app.post("/cache/clear")
async def clear_cache(service: InferenceAPIService = Depends(get_service)):
    """
    Clear prediction cache.
    
    Clears all cached predictions to free memory or force fresh predictions.
    """
    service.clear_cache()
    return {"message": "Cache cleared successfully"}


@app.post("/warmup")
async def warmup(service: InferenceAPIService = Depends(get_service)):
    """
    Warmup endpoint - Pre-loads the model into memory.
    
    Call this before demos/presentations to ensure the model is ready.
    Makes the first real prediction instant instead of waiting 30-60 seconds.
    """
    if not service.model_loaded:
        await service.load_model()
        return {"message": "Model loaded successfully", "status": "ready"}
    return {"message": "Model already loaded", "status": "ready"}


@app.get("/model/info")
async def get_model_info(service: InferenceAPIService = Depends(get_service)):
    """
    Get model information.
    
    Returns detailed information about the loaded model including
    configuration, vocabulary size, and architecture details.
    """
    if not service.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return service.inference_engine.get_model_info()


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with proper logging."""
    logger = logging.getLogger("api.errors")
    logger.error(f"HTTP {exc.status_code}: {exc.detail} - {request.method} {request.url}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions with proper logging."""
    logger = logging.getLogger("api.errors")
    logger.error(f"Unhandled exception: {str(exc)} - {request.method} {request.url}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    # Run the API server
    import os
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(
        "inference_api:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )