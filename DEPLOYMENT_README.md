# LSTM Sentiment Classifier - Deployment and Optimization Guide

This guide covers the deployment and optimization features implemented for the LSTM Sentiment Classifier, including model optimization techniques and production-ready API deployment.

## 🚀 Features Implemented

### Model Optimization (Task 15.1)
- **Dynamic Quantization**: INT8 and FP16 quantization for reduced memory usage
- **Static Quantization**: Calibration-based quantization for maximum performance
- **Model Pruning**: Structured and unstructured pruning to remove unnecessary parameters
- **ONNX Export**: Cross-platform deployment with ONNX Runtime support
- **Performance Benchmarking**: Comprehensive benchmarking and profiling tools

### Production API (Task 15.2)
- **FastAPI Service**: High-performance REST API with automatic documentation
- **Input Validation**: Robust request validation using Pydantic models
- **Error Handling**: Comprehensive error handling with proper HTTP status codes
- **Caching**: TTL-based caching for improved response times
- **Monitoring**: Health checks, performance metrics, and request logging
- **Rate Limiting**: Built-in protection against abuse
- **Batch Processing**: Efficient batch prediction capabilities

## 📁 File Structure

```
├── src/
│   ├── optimization/
│   │   └── model_optimizer.py      # Model optimization toolkit
│   └── api/
│       ├── inference_api.py        # FastAPI service implementation
│       └── client.py              # Python client library
├── benchmark_inference.py          # Performance benchmarking script
├── run_api.py                      # Production API server runner
├── test_api.py                     # API testing and demonstration
├── api_config.yaml                 # API configuration file
└── requirements_api.txt            # API dependencies
```

## 🔧 Model Optimization

### Quick Start

```python
from src.optimization.model_optimizer import create_model_optimizer

# Create optimizer from trained model
optimizer = create_model_optimizer(
    model_path="models/your_model.pth",
    vocab_size=10000,
    device="cuda"
)

# Apply dynamic quantization
quantized_model = optimizer.quantize_model_dynamic(torch.qint8)

# Apply pruning
pruned_model = optimizer.prune_model(pruning_ratio=0.2)

# Export to ONNX
optimizer.export_to_onnx(quantized_model, "optimized_model.onnx")

# Benchmark performance
results = optimizer.benchmark_model(quantized_model)
print(f"Inference time: {results['mean_inference_time']:.4f}s")
```

### Comprehensive Optimization

```bash
# Run comprehensive optimization with benchmarking
python benchmark_inference.py \
    --model-path models/your_model.pth \
    --mode comprehensive \
    --export-onnx \
    --output-dir benchmark_results
```

### Available Optimization Techniques

1. **Dynamic Quantization**
   - INT8: ~4x memory reduction, 2-3x speedup
   - FP16: ~2x memory reduction, 1.5-2x speedup

2. **Static Quantization**
   - Requires calibration data
   - Best performance gains
   - INT8: Up to 4x speedup

3. **Model Pruning**
   - Unstructured: Remove individual weights
   - Structured: Remove entire neurons/channels
   - 10-30% pruning typically maintains accuracy

4. **ONNX Export**
   - Cross-platform deployment
   - Optimized runtime execution
   - CPU and GPU acceleration

## 🌐 Production API Deployment

### Quick Start

1. **Install Dependencies**
```bash
pip install -r requirements_api.txt
```

2. **Configure API**
Edit `api_config.yaml`:
```yaml
model:
  model_path: "models/your_model.pth"
  vocab_path: "models/your_vocabulary.pth"
  device: "auto"

server:
  host: "0.0.0.0"
  port: 8000
  workers: 1
```

3. **Start API Server**
```bash
python run_api.py --config api_config.yaml
```

4. **Test API**
```bash
python test_api.py --api-url http://localhost:8000
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/predict` | POST | Single text prediction |
| `/predict/batch` | POST | Batch text prediction |
| `/health` | GET | Health check |
| `/metrics` | GET | Performance metrics |
| `/model/info` | GET | Model information |
| `/cache/clear` | POST | Clear prediction cache |
| `/docs` | GET | Interactive API documentation |

### Using the Python Client

```python
from src.api.client import SentimentAPIClient

# Create client
client = SentimentAPIClient("http://localhost:8000")

# Single prediction
result = client.predict_sentiment(
    "This movie was fantastic!",
    include_probability=True
)
print(f"Sentiment: {result.sentiment}, Confidence: {result.confidence}")

# Batch prediction
texts = ["Great movie!", "Terrible film.", "Average story."]
results = client.predict_batch_sentiment(
    texts,
    include_statistics=True
)

# Check health
health = client.get_health_status()
print(f"API Status: {health.status}")
```

### Using cURL

```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "This movie was amazing!", "include_probability": true}'

# Batch prediction
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Great film!", "Boring movie."], "include_statistics": true}'

# Health check
curl "http://localhost:8000/health"
```

## 📊 Performance Benchmarking

### Benchmarking Script Usage

```bash
# Benchmark original model
python benchmark_inference.py -m models/your_model.pth --mode original

# Test quantization techniques
python benchmark_inference.py -m models/your_model.pth --mode quantization

# Test pruning with different ratios
python benchmark_inference.py -m models/your_model.pth --mode pruning --pruning-ratios 0.1 0.2 0.3

# Export and benchmark ONNX models
python benchmark_inference.py -m models/your_model.pth --mode onnx --export-onnx

# Run all optimizations
python benchmark_inference.py -m models/your_model.pth --mode all --output-dir results
```

### Expected Performance Improvements

| Optimization | Memory Reduction | Speed Improvement | Accuracy Impact |
|--------------|------------------|-------------------|-----------------|
| Dynamic INT8 | ~75% | 2-3x | Minimal |
| Dynamic FP16 | ~50% | 1.5-2x | Minimal |
| Static INT8 | ~75% | 3-4x | Minimal |
| Pruning 20% | ~20% | 1.2-1.5x | <1% |
| ONNX Export | Variable | 1.2-2x | None |

## 🔧 Configuration Options

### API Configuration (`api_config.yaml`)

```yaml
# Server settings
server:
  host: "0.0.0.0"
  port: 8000
  workers: 1
  log_level: "info"

# Model settings
model:
  model_path: "models/model.pth"
  vocab_path: "models/vocab.pth"
  device: "auto"  # auto, cpu, cuda

# Cache settings
cache:
  size: 1000
  ttl_seconds: 3600

# Security settings
security:
  max_request_size: 1048576  # 1MB
  request_timeout: 30
  cors_origins: ["*"]

# Performance settings
performance:
  max_batch_size: 100
  max_text_length: 10000
```

### Environment Variables

```bash
# Model configuration
export MODEL_PATH="models/your_model.pth"
export VOCAB_PATH="models/your_vocabulary.pth"
export DEVICE="cuda"

# Cache configuration
export CACHE_SIZE="1000"
export CACHE_TTL="3600"

# Server configuration
export API_HOST="0.0.0.0"
export API_PORT="8000"
```

## 🐳 Docker Deployment

### Dockerfile Example

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements_api.txt .
RUN pip install -r requirements_api.txt

# Copy application
COPY src/ src/
COPY models/ models/
COPY api_config.yaml .
COPY run_api.py .

# Expose port
EXPOSE 8000

# Run API
CMD ["python", "run_api.py", "--config", "api_config.yaml"]
```

### Docker Compose Example

```yaml
version: '3.8'
services:
  sentiment-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=models/model.pth
      - VOCAB_PATH=models/vocab.pth
      - DEVICE=cpu
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## 📈 Monitoring and Logging

### Health Monitoring

The API provides comprehensive health monitoring:

```python
# Check if API is healthy
health = client.get_health_status()
print(f"Status: {health.status}")
print(f"Uptime: {health.uptime_seconds}s")
print(f"Memory: {health.memory_usage_mb}MB")
print(f"CPU: {health.cpu_usage_percent}%")
```

### Performance Metrics

```python
# Get performance metrics
metrics = client.get_metrics()
print(f"Total requests: {metrics.total_requests}")
print(f"Success rate: {metrics.successful_requests/metrics.total_requests:.2%}")
print(f"Avg response time: {metrics.average_response_time_ms:.2f}ms")
print(f"Cache hit rate: {metrics.cache_hit_rate:.2%}")
```

### Logging

Logs are automatically generated for:
- All API requests and responses
- Error conditions and exceptions
- Performance metrics
- Health check results

Log files are stored in `logs/api/` directory.

## 🔒 Security Considerations

### Production Deployment Checklist

- [ ] Configure proper CORS origins (not `["*"]`)
- [ ] Set up HTTPS/TLS encryption
- [ ] Implement authentication/authorization
- [ ] Configure rate limiting
- [ ] Set up monitoring and alerting
- [ ] Use environment variables for sensitive config
- [ ] Enable request size limits
- [ ] Configure trusted host middleware
- [ ] Set up log rotation
- [ ] Implement input sanitization

### Rate Limiting

```python
# Example rate limiting configuration
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/predict")
@limiter.limit("10/minute")
async def predict_sentiment(request: Request, ...):
    # Prediction logic
```

## 🚀 Deployment Strategies

### 1. Single Server Deployment
- Simple setup for development/testing
- Use `run_api.py` with configuration file
- Suitable for low-traffic applications

### 2. Load Balanced Deployment
- Multiple API instances behind load balancer
- Horizontal scaling for high traffic
- Use container orchestration (Kubernetes, Docker Swarm)

### 3. Serverless Deployment
- Deploy as AWS Lambda, Google Cloud Functions
- Auto-scaling based on demand
- Cost-effective for variable workloads

### 4. Edge Deployment
- Deploy optimized ONNX models at edge locations
- Reduced latency for global users
- Use CDN with compute capabilities

## 🔍 Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Check model and vocabulary file paths
   - Verify file permissions
   - Ensure compatible PyTorch version

2. **Memory Issues**
   - Reduce batch size
   - Use quantized models
   - Enable model pruning

3. **Performance Issues**
   - Enable caching
   - Use ONNX runtime
   - Optimize batch processing

4. **API Connection Issues**
   - Check firewall settings
   - Verify port availability
   - Test with health endpoint

### Debug Mode

```bash
# Run API in debug mode
python run_api.py --log-level debug --reload

# Test with verbose client
python test_api.py --api-url http://localhost:8000 -v
```

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [ONNX Runtime Documentation](https://onnxruntime.ai/)
- [PyTorch Quantization Guide](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html)
- [Model Pruning Tutorial](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)

## 🤝 Contributing

When contributing to the deployment and optimization features:

1. Follow the existing code structure
2. Add comprehensive tests for new features
3. Update documentation and examples
4. Ensure backward compatibility
5. Test performance impact of changes

## 📄 License

This deployment and optimization implementation is part of the LSTM Sentiment Classifier project and follows the same license terms.