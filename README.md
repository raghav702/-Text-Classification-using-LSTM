# 🎬 LSTM Sentiment Classifier for Movie Reviews

Hey there! This is a deep learning project that can tell whether a movie review is positive or negative. I built it using PyTorch and trained it on thousands of IMDB reviews. Whether you want to analyze customer feedback, social media posts, or just play around with sentiment analysis, this project has you covered.

## 🌟 What Makes This Special?

I've spent quite a bit of time making this both powerful and easy to use. Here's what you get:

- **Smart LSTM Model** - Uses bidirectional LSTM to understand context from both directions (like reading a sentence forward and backward)
- **Pre-trained Word Embeddings** - Leverages GloVe embeddings so the model already "knows" words before training
- **Production-Ready API** - FastAPI server with caching, health checks, and proper error handling
- **Easy Web Interface** - Simple Streamlit UI if you prefer clicking buttons over typing commands
- **Deploy Anywhere** - Dockerized and ready for Google Cloud Run (or any cloud provider)
- **Actually Useful Documentation** - Real examples that you can copy-paste and run

## 🚀 Get Started in 5 Minutes

### Installation

```bash
# Clone this repo
git clone https://github.com/raghav702/-Text-Classification-using-LSTM.git
cd -Text-Classification-using-LSTM

# Create a virtual environment (trust me, do this)
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Mac/Linux

# Install dependencies
pip install -r requirements.txt
pip install -r requirements_api.txt
```

### Try It Out Right Away

The easiest way to see it in action is using the pre-trained model:

```bash
# Run the web UI (recommended for first-timers)
streamlit run app.py
```

This opens a browser where you can type any review and get instant predictions. It's pretty satisfying to watch!

Or if you prefer APIs:

```bash
# Start the API server
python run_api.py

# Then visit http://localhost:8000/docs for the interactive API docs
```

### Quick Predictions

```bash
# Predict a single review
python predict.py \
  -m models/improved_lstm_model_20251106_003134.pth \
  -v models/improved_lstm_model_20251106_003134_vocabulary.pth \
  -t "This movie was absolutely fantastic! Loved every minute."

# Output: positive (confidence: 0.95)
```

## 📁 What's Inside?

Here's how everything is organized (I tried to keep it simple):

```
lstm_model/
├── app.py                  # Streamlit web interface - easiest way to use it
├── run_api.py              # FastAPI server - for production use
├── train.py                # Training script - if you want to train your own
├── predict.py              # Command-line predictions
├── evaluate.py             # See how well the model performs
├── src/                    # All the code
│   ├── api/               # API implementation
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # LSTM architecture
│   ├── training/          # Training pipeline
│   └── inference/         # Prediction engine
├── models/                # Pre-trained models live here
├── data/                  # Your datasets go here
│   ├── imdb/             # Movie reviews
│   └── glove/            # Word embeddings (auto-downloaded)
├── configs/              # Configuration files
└── tests/                # Unit tests (they all pass!)
```

## 🎯 Use Cases

Here are some real ways you might use this:

**For Learning:**
- Understanding how LSTMs work with text
- Learning about word embeddings (GloVe)
- Getting hands-on with PyTorch
- Seeing a complete ML project from start to finish

**For Projects:**
- Analyze product reviews
- Monitor social media sentiment
- Classify customer feedback
- Filter spam/negative comments
- Build a sentiment dashboard

**For Production:**
- API is production-ready with proper error handling
- Supports batch processing for efficiency
- Includes request caching to save compute
- Docker container for easy deployment
- Works great on Google Cloud Run (stays within free tier!)

## 🧠 How It Works (The Simple Version)

1. **Text → Numbers**: Reviews are converted to sequences of numbers (each word becomes an ID)
2. **Word Embeddings**: Each word ID gets mapped to a 300-dimensional vector (using GloVe)
3. **LSTM Processing**: The LSTM reads the sequence, understanding context and relationships
4. **Classification**: A final layer outputs positive/negative with a confidence score

The model was trained on 25,000 IMDB reviews and achieves about 88-90% accuracy on unseen data.

## 📊 Model Performance

On the IMDB test set:
- **Accuracy**: ~88-90%
- **Precision**: 0.89
- **Recall**: 0.88
- **F1 Score**: 0.88

Real-world performance:
- Handles misspellings reasonably well (thanks to embeddings)
- Understands negations ("not good" → negative)
- Picks up on sarcasm about 60% of the time (still working on this!)
- Fast: Can process 1000+ reviews per second

## 🚂 Training Your Own Model

If you want to train from scratch (takes 2-4 hours on a decent GPU):

```bash
# Download the IMDB dataset first
python download_imdb_data.py

# Download GloVe embeddings (optional but recommended)
python download_glove.py

# Start training
python train.py \
  --data-dir data/imdb \
  --epochs 20 \
  --batch-size 64 \
  --use-glove \
  --output-dir models/my_model
```

The training script will:
- Automatically use GPU if available
- Save checkpoints every few epochs
- Stop early if the model stops improving
- Show you progress bars and metrics
- Save the best model automatically

Tips for better results:
- More epochs = better results (but diminishing returns after 20-30)
- Larger batch sizes train faster but need more GPU memory
- Using GloVe embeddings gives a ~3-5% accuracy boost
- Try different learning rates if training is unstable

## 🔌 Using the API

The FastAPI server is perfect for integrating with other applications:

### Start the Server

```bash
python run_api.py
```

The server starts at `http://localhost:8000` with:
- Interactive docs at `/docs` (really useful, check it out!)
- Health check at `/health`
- Predictions at `/predict` and `/predict/batch`

### Example API Calls

**Single Prediction:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was incredible!"}'

# Response:
# {
#   "sentiment": "positive",
#   "confidence": 0.94,
#   "probability": 0.97
# }
```

**Batch Predictions:**
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Loved it!",
      "Waste of time",
      "Pretty good movie"
    ]
  }'
```

**Using Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "Amazing film with great acting!"}
)
print(response.json())
# {'sentiment': 'positive', 'confidence': 0.91, ...}
```

The API includes built-in caching, so repeated requests are super fast!

## ☁️ Deploying to the Cloud

I've included everything you need to deploy on Google Cloud Run (stays within free tier for most use cases):

### Quick Deploy

```powershell
# Update deploy.ps1 with your GCP project ID
# Then run:
.\deploy.ps1
```

That's it! Your API will be live at a public URL in about 5-10 minutes.

### Cost Optimization

The deployment is configured to:
- Scale to zero when not in use (no idle costs!)
- Use minimal resources (512MB RAM, 0.5 CPU)
- Stay within the free tier for <10K requests/month
- Expected cost: **$0-2/month** for typical usage

See `DEPLOYMENT_GCP.md` for detailed instructions.

## 🛠️ Configuration

I've included several pre-made configs in `configs/examples/`:

- `quick_training.yaml` - Fast training for testing (5 epochs)
- `production_training.yaml` - Full training for best results (50 epochs)
- `better_training.yaml` - Balanced approach (20 epochs)

You can edit these or create your own. The important settings:

```yaml
# Model size (bigger = better but slower)
hidden_dim: 128           # Try 256 for better quality
n_layers: 2               # 2-3 is usually best

# Training
batch_size: 64            # Reduce if you run out of memory
learning_rate: 0.001      # Lower if training is unstable
epochs: 20                # More epochs = better results (to a point)

# Embeddings
use_glove: true           # Highly recommended!
glove_dim: "300d"         # 100d/200d/300d available
```

## 📈 Evaluation

Want to see how well the model performs?

```bash
python evaluate.py \
  -m models/improved_lstm_model_20251106_003134.pth \
  -v models/improved_lstm_model_20251106_003134_vocabulary.pth \
  -d data/imdb
```

This generates:
- Confusion matrix
- Precision/recall curves
- ROC curves
- Detailed metrics report
- All saved to `evaluation_results/`

## 🧪 Testing

I've included unit tests for all major components:

```bash
# Run all tests
pytest tests/ -v

# Run specific tests
pytest tests/test_training_integration.py -v

# Check coverage
pytest tests/ --cov=src --cov-report=html
```

## 🐛 Troubleshooting

**"CUDA out of memory"**
- Reduce batch size in config
- Use CPU instead: add `--device cpu` when training

**"Model file not found"**
- Make sure you're using the full path to the `.pth` files
- Check that both model and vocabulary files exist

**"Slow predictions"**
- First prediction is always slower (model loading)
- Use batch predictions for multiple texts
- Enable caching in `api_config.yaml`

**API not starting**
- Check if port 8000 is already in use
- Try a different port: `python run_api.py --port 8080`
- Check logs in `logs/api/`

## 💡 Tips & Tricks

1. **Speed up training**: Use a GPU if you have one (10-20x faster)
2. **Better accuracy**: Use GloVe embeddings (adds ~5% accuracy)
3. **Save money**: The API auto-sleeps on Cloud Run when not in use
4. **Debug issues**: Check `logs/` directory for detailed error messages
5. **Test locally first**: Always test with `streamlit run app.py` before deploying

## 🤝 Contributing

Found a bug? Have an idea? Contributions are welcome!

1. Fork the repo
2. Create a branch: `git checkout -b cool-feature`
3. Make your changes
4. Test them: `pytest tests/ -v`
5. Submit a pull request

## 📚 What I Learned Building This

This project taught me a lot about:
- LSTMs and how they handle sequential data
- The importance of pre-trained embeddings
- Why batch normalization matters
- How to build production-ready ML APIs
- Docker and cloud deployments
- Writing code that others can actually use

## 📄 License

MIT License - feel free to use this however you want!

## 🙏 Credits & Acknowledgments

- **IMDB Dataset**: Andrew Maas et al. - Thanks for the 50K reviews!
- **GloVe**: Stanford NLP Group - Amazing word embeddings
- **PyTorch**: Facebook AI - Great framework
- **You**: For checking out this project!

## 📬 Questions?

If something's not working or you're confused about anything:
1. Check the troubleshooting section above
2. Look at the examples in `examples/`
3. Read the deployment docs in `DEPLOYMENT_GCP.md`
4. Open an issue on GitHub

---

Built with ❤️ and lots of ☕ | Last updated: December 2025