#!/usr/bin/env python3
"""
Test Script for LSTM Sentiment Classifier API

This script demonstrates how to use the sentiment analysis API
and tests various endpoints and functionality.
"""

import sys
import os
import time
import argparse
from typing import List

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from api.client import SentimentAPIClient, predict_sentiment, predict_batch_sentiment, check_api_health


def test_single_prediction(client: SentimentAPIClient):
    """Test single text prediction."""
    print("\n" + "="*50)
    print("Testing Single Prediction")
    print("="*50)
    
    test_texts = [
        "This movie was absolutely fantastic! I loved every minute of it.",
        "The film was terrible and boring. Complete waste of time.",
        "It was an okay movie, nothing special but not bad either.",
        "Amazing cinematography and outstanding performances by all actors.",
        "I fell asleep halfway through. Very disappointing."
    ]
    
    for text in test_texts:
        try:
            result = client.predict_sentiment(text, include_probability=True)
            print(f"\nText: {text}")
            print(f"Sentiment: {result.sentiment}")
            print(f"Confidence: {result.confidence:.3f}")
            print(f"Probability: {result.probability:.3f}")
            print(f"Processing Time: {result.processing_time_ms:.2f}ms")
        except Exception as e:
            print(f"Error predicting sentiment for text: {e}")


def test_batch_prediction(client: SentimentAPIClient):
    """Test batch text prediction."""
    print("\n" + "="*50)
    print("Testing Batch Prediction")
    print("="*50)
    
    test_texts = [
        "Great movie with excellent acting!",
        "Boring and predictable storyline.",
        "The special effects were incredible.",
        "Poor dialogue and weak character development.",
        "A masterpiece of modern cinema.",
        "Not worth the ticket price.",
        "Engaging plot with surprising twists.",
        "The worst movie I've ever seen."
    ]
    
    try:
        result = client.predict_batch_sentiment(
            test_texts,
            include_probability=True,
            include_statistics=True
        )
        
        print(f"\nBatch Results ({len(result.predictions)} predictions):")
        print(f"Total Processing Time: {result.total_processing_time_ms:.2f}ms")
        
        for i, (text, prediction) in enumerate(zip(test_texts, result.predictions)):
            print(f"\n{i+1}. {text}")
            print(f"   Sentiment: {prediction.sentiment}")
            print(f"   Confidence: {prediction.confidence:.3f}")
            print(f"   Probability: {prediction.probability:.3f}")
        
        if result.statistics:
            stats = result.statistics
            print(f"\nBatch Statistics:")
            print(f"  Positive predictions: {stats['positive_predictions']}")
            print(f"  Negative predictions: {stats['negative_predictions']}")
            print(f"  Positive ratio: {stats['positive_ratio']:.2%}")
            print(f"  Average confidence: {stats['average_confidence']:.3f}")
            print(f"  High confidence count: {stats['high_confidence_count']}")
            
    except Exception as e:
        print(f"Error in batch prediction: {e}")


def test_health_and_metrics(client: SentimentAPIClient):
    """Test health check and metrics endpoints."""
    print("\n" + "="*50)
    print("Testing Health and Metrics")
    print("="*50)
    
    try:
        # Health check
        health = client.get_health_status()
        print(f"\nHealth Status:")
        print(f"  Status: {health.status}")
        print(f"  Model Loaded: {health.model_loaded}")
        print(f"  Uptime: {health.uptime_seconds:.1f} seconds")
        print(f"  Memory Usage: {health.memory_usage_mb:.1f} MB")
        print(f"  CPU Usage: {health.cpu_usage_percent:.1f}%")
        
        # Metrics
        metrics = client.get_metrics()
        print(f"\nPerformance Metrics:")
        print(f"  Total Requests: {metrics.total_requests}")
        print(f"  Successful Requests: {metrics.successful_requests}")
        print(f"  Failed Requests: {metrics.failed_requests}")
        print(f"  Average Response Time: {metrics.average_response_time_ms:.2f}ms")
        print(f"  Cache Hit Rate: {metrics.cache_hit_rate:.2%}")
        
    except Exception as e:
        print(f"Error getting health/metrics: {e}")


def test_model_info(client: SentimentAPIClient):
    """Test model information endpoint."""
    print("\n" + "="*50)
    print("Testing Model Information")
    print("="*50)
    
    try:
        model_info = client.get_model_info()
        print(f"\nModel Information:")
        
        if 'model_config' in model_info:
            config = model_info['model_config']
            print(f"  Vocabulary Size: {config.get('vocab_size', 'N/A')}")
            print(f"  Embedding Dimension: {config.get('embedding_dim', 'N/A')}")
            print(f"  Hidden Dimension: {config.get('hidden_dim', 'N/A')}")
            print(f"  Number of Layers: {config.get('n_layers', 'N/A')}")
            print(f"  Bidirectional: {config.get('bidirectional', 'N/A')}")
        
        if 'model_parameters' in model_info:
            params = model_info['model_parameters']
            print(f"  Total Parameters: {params.get('total_parameters', 'N/A'):,}")
            print(f"  Trainable Parameters: {params.get('trainable_parameters', 'N/A'):,}")
        
        if 'vocab_info' in model_info:
            vocab = model_info['vocab_info']
            print(f"  Vocabulary Size: {vocab.get('vocab_size', 'N/A')}")
            print(f"  Max Sequence Length: {vocab.get('max_length', 'N/A')}")
        
    except Exception as e:
        print(f"Error getting model info: {e}")


def test_cache_functionality(client: SentimentAPIClient):
    """Test cache functionality."""
    print("\n" + "="*50)
    print("Testing Cache Functionality")
    print("="*50)
    
    test_text = "This is a test text for cache functionality."
    
    try:
        # First prediction (cache miss)
        print("Making first prediction (should be cache miss)...")
        start_time = time.time()
        result1 = client.predict_sentiment(test_text)
        time1 = time.time() - start_time
        print(f"First prediction time: {time1*1000:.2f}ms")
        
        # Second prediction (should be cache hit)
        print("Making second prediction (should be cache hit)...")
        start_time = time.time()
        result2 = client.predict_sentiment(test_text)
        time2 = time.time() - start_time
        print(f"Second prediction time: {time2*1000:.2f}ms")
        
        # Verify results are the same
        if (result1.sentiment == result2.sentiment and 
            abs(result1.confidence - result2.confidence) < 0.001):
            print("✓ Cache working correctly - results are identical")
            if time2 < time1:
                print("✓ Cache improved response time")
        else:
            print("✗ Cache issue - results differ")
        
        # Clear cache
        print("Clearing cache...")
        client.clear_cache()
        print("✓ Cache cleared")
        
    except Exception as e:
        print(f"Error testing cache: {e}")


def test_error_handling(client: SentimentAPIClient):
    """Test error handling with invalid inputs."""
    print("\n" + "="*50)
    print("Testing Error Handling")
    print("="*50)
    
    # Test empty text
    try:
        client.predict_sentiment("")
        print("✗ Empty text should have failed")
    except ValueError:
        print("✓ Empty text correctly rejected")
    except Exception as e:
        print(f"✗ Unexpected error for empty text: {e}")
    
    # Test invalid threshold
    try:
        client.predict_sentiment("Test text", threshold=1.5)
        print("✗ Invalid threshold should have failed")
    except ValueError:
        print("✓ Invalid threshold correctly rejected")
    except Exception as e:
        print(f"✗ Unexpected error for invalid threshold: {e}")
    
    # Test oversized batch
    try:
        large_batch = ["Test text"] * 101
        client.predict_batch_sentiment(large_batch)
        print("✗ Oversized batch should have failed")
    except ValueError:
        print("✓ Oversized batch correctly rejected")
    except Exception as e:
        print(f"✗ Unexpected error for oversized batch: {e}")
    
    # Test empty batch
    try:
        client.predict_batch_sentiment([])
        print("✗ Empty batch should have failed")
    except ValueError:
        print("✓ Empty batch correctly rejected")
    except Exception as e:
        print(f"✗ Unexpected error for empty batch: {e}")


def test_performance_benchmark(client: SentimentAPIClient):
    """Run a simple performance benchmark."""
    print("\n" + "="*50)
    print("Performance Benchmark")
    print("="*50)
    
    test_texts = [
        "This movie was great!",
        "Terrible film, waste of time.",
        "Average movie, nothing special.",
        "Excellent acting and direction.",
        "Boring and predictable plot."
    ] * 10  # 50 texts total
    
    try:
        # Single predictions
        print("Benchmarking single predictions...")
        start_time = time.time()
        for text in test_texts[:10]:  # Test 10 texts
            client.predict_sentiment(text)
        single_time = time.time() - start_time
        print(f"10 single predictions: {single_time:.2f}s ({single_time/10*1000:.2f}ms per prediction)")
        
        # Batch prediction
        print("Benchmarking batch prediction...")
        start_time = time.time()
        client.predict_batch_sentiment(test_texts[:10])
        batch_time = time.time() - start_time
        print(f"1 batch of 10 predictions: {batch_time:.2f}s ({batch_time/10*1000:.2f}ms per prediction)")
        
        # Compare performance
        if batch_time < single_time:
            speedup = single_time / batch_time
            print(f"✓ Batch processing is {speedup:.1f}x faster")
        else:
            print("✗ Batch processing is not faster (possibly due to cache)")
        
    except Exception as e:
        print(f"Error in performance benchmark: {e}")


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test LSTM Sentiment Classifier API")
    
    parser.add_argument(
        '--api-url', '-u',
        default='http://localhost:8000',
        help='API server URL (default: http://localhost:8000)'
    )
    
    parser.add_argument(
        '--test', '-t',
        choices=['all', 'single', 'batch', 'health', 'model', 'cache', 'errors', 'performance'],
        default='all',
        help='Which test to run (default: all)'
    )
    
    parser.add_argument(
        '--wait-for-ready',
        action='store_true',
        help='Wait for API to become ready before testing'
    )
    
    parser.add_argument(
        '--timeout',
        type=float,
        default=60.0,
        help='Timeout for waiting for API to be ready (default: 60s)'
    )
    
    args = parser.parse_args()
    
    print("LSTM Sentiment Classifier API Test")
    print(f"API URL: {args.api_url}")
    
    # Check if API is available
    print("\nChecking API availability...")
    if not check_api_health(args.api_url):
        if args.wait_for_ready:
            print("API not ready, waiting...")
            with SentimentAPIClient(args.api_url) as client:
                if not client.wait_for_ready(timeout=args.timeout):
                    print("API did not become ready within timeout")
                    sys.exit(1)
        else:
            print("API is not available. Make sure the server is running.")
            print("Use --wait-for-ready to wait for the API to become available.")
            sys.exit(1)
    
    print("✓ API is available and ready")
    
    # Run tests
    with SentimentAPIClient(args.api_url) as client:
        if args.test in ['all', 'single']:
            test_single_prediction(client)
        
        if args.test in ['all', 'batch']:
            test_batch_prediction(client)
        
        if args.test in ['all', 'health']:
            test_health_and_metrics(client)
        
        if args.test in ['all', 'model']:
            test_model_info(client)
        
        if args.test in ['all', 'cache']:
            test_cache_functionality(client)
        
        if args.test in ['all', 'errors']:
            test_error_handling(client)
        
        if args.test in ['all', 'performance']:
            test_performance_benchmark(client)
    
    print("\n" + "="*50)
    print("All tests completed!")
    print("="*50)


if __name__ == '__main__':
    main()