#!/usr/bin/env python3
"""
Production API Server Runner for LSTM Sentiment Classifier

This script provides a production-ready way to run the sentiment analysis API
with proper configuration, logging, and monitoring.
"""

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path
import uvicorn
from typing import Dict, Any

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def setup_logging(config: Dict[str, Any]):
    """Set up logging based on configuration."""
    log_config = config.get('logging', {})
    
    # Create log directory
    log_file = log_config.get('file', 'logs/api/api.log')
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    level = getattr(logging, log_config.get('level', 'INFO').upper())
    format_str = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def setup_environment(config: Dict[str, Any]):
    """Set up environment variables from configuration."""
    model_config = config.get('model', {})
    cache_config = config.get('cache', {})
    
    # Set model paths
    if 'model_path' in model_config:
        os.environ['MODEL_PATH'] = model_config['model_path']
    if 'vocab_path' in model_config:
        os.environ['VOCAB_PATH'] = model_config['vocab_path']
    if 'device' in model_config:
        os.environ['DEVICE'] = model_config['device']
    
    # Set cache configuration
    if 'size' in cache_config:
        os.environ['CACHE_SIZE'] = str(cache_config['size'])
    if 'ttl_seconds' in cache_config:
        os.environ['CACHE_TTL'] = str(cache_config['ttl_seconds'])


def validate_model_files(config: Dict[str, Any]) -> bool:
    """Validate that required model files exist."""
    model_config = config.get('model', {})
    
    model_path = model_config.get('model_path')
    vocab_path = model_config.get('vocab_path')
    
    if not model_path or not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return False
    
    if not vocab_path or not os.path.exists(vocab_path):
        print(f"Error: Vocabulary file not found: {vocab_path}")
        return False
    
    return True


def create_uvicorn_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create uvicorn configuration from config file."""
    server_config = config.get('server', {})
    
    uvicorn_config = {
        'app': 'src.api.inference_api:app',
        'host': server_config.get('host', '0.0.0.0'),
        'port': server_config.get('port', 8000),
        'workers': server_config.get('workers', 1),
        'reload': server_config.get('reload', False),
        'log_level': server_config.get('log_level', 'info'),
        'access_log': server_config.get('access_log', True),
    }
    
    return uvicorn_config


def print_startup_info(config: Dict[str, Any]):
    """Print startup information."""
    server_config = config.get('server', {})
    model_config = config.get('model', {})
    
    print("="*60)
    print("LSTM Sentiment Classifier API Server")
    print("="*60)
    print(f"Server: http://{server_config.get('host', '0.0.0.0')}:{server_config.get('port', 8000)}")
    print(f"Model: {model_config.get('model_path', 'Not specified')}")
    print(f"Vocabulary: {model_config.get('vocab_path', 'Not specified')}")
    print(f"Device: {model_config.get('device', 'auto')}")
    print(f"Workers: {server_config.get('workers', 1)}")
    print(f"Log Level: {server_config.get('log_level', 'info')}")
    print("="*60)
    print("API Endpoints:")
    print("  - GET  /           - API information")
    print("  - POST /predict    - Single text prediction")
    print("  - POST /predict/batch - Batch text prediction")
    print("  - GET  /health     - Health check")
    print("  - GET  /metrics    - Performance metrics")
    print("  - GET  /docs       - API documentation")
    print("="*60)


def main():
    """Main function to run the API server."""
    parser = argparse.ArgumentParser(
        description="LSTM Sentiment Classifier API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--config', '-c',
        default='api_config.yaml',
        help='Path to configuration file (default: api_config.yaml)'
    )
    
    parser.add_argument(
        '--host',
        help='Host to bind to (overrides config)'
    )
    
    parser.add_argument(
        '--port', '-p',
        type=int,
        help='Port to bind to (overrides config)'
    )
    
    parser.add_argument(
        '--workers', '-w',
        type=int,
        help='Number of worker processes (overrides config)'
    )
    
    parser.add_argument(
        '--reload',
        action='store_true',
        help='Enable auto-reload for development'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['debug', 'info', 'warning', 'error'],
        help='Log level (overrides config)'
    )
    
    parser.add_argument(
        '--model-path',
        help='Path to model file (overrides config)'
    )
    
    parser.add_argument(
        '--vocab-path',
        help='Path to vocabulary file (overrides config)'
    )
    
    parser.add_argument(
        '--device',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to use for inference (overrides config)'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate configuration and model files, do not start server'
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        print(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
        
        # Override config with command line arguments
        if args.host:
            config.setdefault('server', {})['host'] = args.host
        if args.port:
            config.setdefault('server', {})['port'] = args.port
        if args.workers:
            config.setdefault('server', {})['workers'] = args.workers
        if args.reload:
            config.setdefault('server', {})['reload'] = True
        if args.log_level:
            config.setdefault('server', {})['log_level'] = args.log_level
        if args.model_path:
            config.setdefault('model', {})['model_path'] = args.model_path
        if args.vocab_path:
            config.setdefault('model', {})['vocab_path'] = args.vocab_path
        if args.device:
            config.setdefault('model', {})['device'] = args.device
        
        # Set up logging
        setup_logging(config)
        logger = logging.getLogger(__name__)
        
        # Validate model files
        if not validate_model_files(config):
            sys.exit(1)
        
        logger.info("Model files validated successfully")
        
        if args.validate_only:
            print("Configuration and model files are valid.")
            return
        
        # Set up environment
        setup_environment(config)
        
        # Print startup information
        print_startup_info(config)
        
        # Create uvicorn configuration
        uvicorn_config = create_uvicorn_config(config)
        
        # Start the server
        logger.info("Starting API server...")
        uvicorn.run(**uvicorn_config)
        
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()