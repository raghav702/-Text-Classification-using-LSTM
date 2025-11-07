#!/usr/bin/env python3
"""
Simple test script for the LSTM model with hardcoded reviews.
"""

import os
import sys
import numpy as np

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from inference.inference_engine import create_inference_engine

def get_test_reviews():
    """Get 20 test reviews with known sentiments."""
    reviews = [
        # Clearly Positive (10 reviews)
        ("This movie is absolutely fantastic! Amazing acting and great story.", "positive"),
        ("I loved every minute of this film. Brilliant cinematography!", "positive"),
        ("Outstanding movie with incredible performances. Highly recommended!", "positive"),
        ("This is one of the best films I've ever seen. Perfect!", "positive"),
        ("Excellent movie with wonderful acting and beautiful visuals.", "positive"),
        ("Great film with an amazing storyline and perfect execution.", "positive"),
        ("I really enjoyed this movie. It was entertaining and well-made.", "positive"),
        ("Fantastic acting, great plot, and amazing direction. Must watch!", "positive"),
        ("Beautiful film with excellent performances and stunning scenes.", "positive"),
        ("This movie exceeded my expectations. Truly outstanding work!", "positive"),
        
        # Clearly Negative (10 reviews)
        ("This movie is terrible. Awful acting and boring plot.", "negative"),
        ("I hated this film. Complete waste of time and money.", "negative"),
        ("Worst movie ever! Bad acting, terrible story, avoid it.", "negative"),
        ("This film is absolutely horrible. Couldn't finish watching.", "negative"),
        ("Terrible acting and ridiculous plot. Completely disappointing.", "negative"),
        ("This movie is a disaster. Poor direction and awful performances.", "negative"),
        ("I regret watching this film. Boring and badly executed.", "negative"),
        ("Horrible movie with terrible acting and nonsensical story.", "negative"),
        ("This film is unwatchable. Bad in every possible way.", "negative"),
        ("Awful movie. Poor quality and completely uninteresting.", "negative"),
    ]
    
    texts = [review[0] for review in reviews]
    labels = [review[1] for review in reviews]
    
    return texts, labels

def test_model():
    """Test the model with sample reviews."""
    print("🧪 Testing LSTM Sentiment Classifier")
    print("=" * 50)
    
    # Find the latest improved model
    model_files = [f for f in os.listdir('models') 
                   if f.startswith('improved_lstm_model') 
                   and f.endswith('.pth') 
                   and 'vocabulary' not in f]
    
    if not model_files:
        print("❌ No improved model found. Please train the model first.")
        return
    
    # Get the latest model
    latest_model = sorted(model_files)[-1]
    model_path = os.path.join('models', latest_model)
    vocab_path = model_path.replace('.pth', '_vocabulary.pth')
    
    print(f"📁 Using model: {latest_model}")
    
    # Load model
    try:
        engine = create_inference_engine(model_path, vocab_path, device='cpu')
        print("✅ Model loaded successfully!")
        
        model_info = engine.get_model_info()
        print(f"📊 Vocabulary size: {model_info['vocab_info']['vocab_size']:,}")
        print(f"🔧 Model parameters: {model_info['model_parameters']['total_parameters']:,}")
        print()
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Get test reviews
    texts, true_labels = get_test_reviews()
    print(f"📝 Testing with {len(texts)} reviews:")
    print(f"   • Positive reviews: {sum(1 for l in true_labels if l == 'positive')}")
    print(f"   • Negative reviews: {sum(1 for l in true_labels if l == 'negative')}")
    print()
    
    # Get predictions
    print("🔮 Getting predictions...")
    try:
        results = engine.batch_predict_with_probabilities(texts)
        predictions = [result[0] for result in results]
        probabilities = [result[1] for result in results]
        confidences = [result[2] for result in results]
        print("✅ Predictions completed!")
        print()
    except Exception as e:
        print(f"❌ Error getting predictions: {e}")
        return
    
    # Analyze results
    print("📊 RESULTS ANALYSIS")
    print("=" * 50)
    
    # Calculate accuracy
    correct = sum(1 for true, pred in zip(true_labels, predictions) if true == pred)
    accuracy = correct / len(true_labels)
    
    print(f"🎯 Accuracy: {accuracy:.1%} ({correct}/{len(true_labels)})")
    
    # Check if model is working properly (not giving constant predictions)
    prob_range = max(probabilities) - min(probabilities)
    print(f"📈 Probability range: {prob_range:.4f}")
    print(f"   • Min probability: {min(probabilities):.4f}")
    print(f"   • Max probability: {max(probabilities):.4f}")
    
    if prob_range < 0.1:
        print("⚠️  WARNING: Model showing constant predictions (like old broken model)")
    else:
        print("✅ Model shows varied predictions (good sign!)")
    
    print()
    
    # Show individual predictions
    print("🔍 INDIVIDUAL PREDICTIONS")
    print("=" * 50)
    
    for i, (text, true_label, pred, prob, conf) in enumerate(zip(texts, true_labels, predictions, probabilities, confidences)):
        status = "✅" if true_label == pred else "❌"
        print(f"{status} Review {i+1:2d}: {pred.upper()} ({prob:.3f}) | True: {true_label.upper()}")
        print(f"    Text: {text[:80]}{'...' if len(text) > 80 else ''}")
        print()
    
    # Summary
    print("📋 SUMMARY")
    print("=" * 50)
    
    if accuracy >= 0.8:
        performance = "🎉 EXCELLENT"
    elif accuracy >= 0.6:
        performance = "✅ GOOD"
    elif accuracy >= 0.4:
        performance = "⚠️  FAIR"
    else:
        performance = "❌ POOR"
    
    print(f"Performance: {performance}")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"Average confidence: {np.mean(confidences):.3f}")
    
    # Check improvement from old model
    if prob_range > 0.2:
        print("🚀 MAJOR IMPROVEMENT: Model now gives varied predictions!")
        print("   (Old model gave ~0.698 for everything)")
    elif prob_range > 0.1:
        print("📈 IMPROVEMENT: Model shows some variation in predictions")
    else:
        print("⚠️  Model may still have issues with constant predictions")
    
    print(f"\n🕐 Test completed!")

if __name__ == "__main__":
    test_model()