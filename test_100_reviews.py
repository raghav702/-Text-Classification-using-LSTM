#!/usr/bin/env python3
"""
Test the improved LSTM model on 100 sample reviews to analyze performance.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from inference.inference_engine import create_inference_engine

def load_test_samples(n_samples=100):
    """Load n_samples from test data for evaluation."""
    test_path = "data/imdb/test.csv"
    
    if not os.path.exists(test_path):
        print("Test data not found. Using hardcoded sample reviews.")
        return get_hardcoded_samples()
    
    try:
        test_df = pd.read_csv(test_path)
        
        # Sample n_samples reviews (balanced if possible)
        if len(test_df) >= n_samples:
            # Try to get balanced samples
            text_col = 'review' if 'review' in test_df.columns else 'text'
            label_col = 'sentiment' if 'sentiment' in test_df.columns else 'label'
            
            # Get equal numbers of positive and negative if possible
            if label_col in test_df.columns:
                pos_samples = test_df[test_df[label_col] == 'positive'].head(n_samples//2)
                neg_samples = test_df[test_df[label_col] == 'negative'].head(n_samples//2)
                sample_df = pd.concat([pos_samples, neg_samples]).sample(frac=1).reset_index(drop=True)
            else:
                sample_df = test_df.sample(n=n_samples).reset_index(drop=True)
            
            texts = sample_df[text_col].tolist()
            true_labels = sample_df[label_col].tolist() if label_col in sample_df.columns else None
            
            return texts, true_labels
        else:
            print(f"Not enough samples in test data ({len(test_df)}). Using hardcoded samples.")
            return get_hardcoded_samples()
            
    except Exception as e:
        print(f"Error loading test data: {e}. Using hardcoded samples.")
        return get_hardcoded_samples()

def get_hardcoded_samples():
    """Get hardcoded sample reviews for testing."""
    samples = [
        # Clearly Positive Reviews
        ("This movie is absolutely fantastic! The acting is superb and the plot is engaging.", "positive"),
        ("I loved every minute of this film. It's a masterpiece!", "positive"),
        ("Brilliant cinematography and outstanding performances. Highly recommended!", "positive"),
        ("One of the best movies I've ever seen. Amazing story and great characters.", "positive"),
        ("Excellent film with wonderful acting and beautiful visuals.", "positive"),
        ("This is a great movie with an incredible storyline and perfect execution.", "positive"),
        ("Fantastic acting, great plot, and amazing direction. A must-watch!", "positive"),
        ("I really enjoyed this movie. It was entertaining and well-made.", "positive"),
        ("Beautiful film with excellent performances and stunning cinematography.", "positive"),
        ("This movie exceeded my expectations. Truly outstanding!", "positive"),
        
        # Clearly Negative Reviews
        ("This movie is terrible. The acting is awful and the plot makes no sense.", "negative"),
        ("I hated this film. It was boring and poorly made.", "negative"),
        ("Worst movie ever! Bad acting, terrible story, complete waste of time.", "negative"),
        ("This film is absolutely horrible. I couldn't even finish watching it.", "negative"),
        ("Terrible acting and a ridiculous plot. Avoid this movie at all costs.", "negative"),
        ("This movie is a disaster. Poor direction and awful performances.", "negative"),
        ("I regret watching this film. It was boring and badly executed.", "negative"),
        ("Horrible movie with terrible acting and a nonsensical story.", "negative"),
        ("This film is unwatchable. Bad in every possible way.", "negative"),
        ("Awful movie. Poor quality and completely uninteresting.", "negative"),
        
        # Moderately Positive
        ("The movie was good. Not great, but definitely worth watching.", "positive"),
        ("I liked this film. It had some good moments and decent acting.", "positive"),
        ("Pretty good movie with solid performances and an interesting story.", "positive"),
        ("This film was enjoyable. Good entertainment value.", "positive"),
        ("Nice movie with good acting and a decent plot.", "positive"),
        
        # Moderately Negative
        ("The movie was disappointing. Expected much better.", "negative"),
        ("Not a great film. Some good parts but overall mediocre.", "negative"),
        ("This movie was okay but nothing special. Rather forgettable.", "negative"),
        ("Mediocre film with average acting and a predictable plot.", "negative"),
        ("The movie was boring and failed to hold my interest.", "negative"),
        
        # Mixed/Neutral (should be challenging)
        ("The movie had good parts and bad parts. Mixed feelings about it.", "positive"),  # Slightly positive
        ("Some great acting but the story was weak. Average overall.", "negative"),  # Slightly negative
        ("Beautiful visuals but the plot was confusing and slow.", "negative"),
        ("Great concept but poor execution. Could have been much better.", "negative"),
        ("The first half was excellent but the second half was disappointing.", "negative"),
        
        # Short reviews
        ("Great movie!", "positive"),
        ("Loved it!", "positive"),
        ("Terrible film.", "negative"),
        ("Hated it.", "negative"),
        ("Amazing!", "positive"),
        ("Awful.", "negative"),
        ("Perfect!", "positive"),
        ("Boring.", "negative"),
        
        # Longer, more complex reviews
        ("This movie starts strong with excellent character development and engaging dialogue, but unfortunately loses momentum in the second act with unnecessary subplots and pacing issues. The cinematography is beautiful throughout, and the lead actors deliver convincing performances. While it has its flaws, the emotional core of the story resonates well, making it a worthwhile watch despite its shortcomings.", "positive"),
        ("Despite having a promising premise and a talented cast, this film fails to deliver on multiple fronts. The script is poorly written with dialogue that feels forced and unnatural. The pacing is inconsistent, with long stretches of boring exposition followed by rushed action sequences. The director seems to have lost control of the narrative, resulting in a confusing and unsatisfying experience. Even the usually reliable lead actor seems disengaged. A disappointing effort that wastes its potential.", "negative"),
        
        # Reviews with specific movie elements
        ("The special effects were incredible and the action sequences were thrilling. Great entertainment!", "positive"),
        ("Poor special effects and unconvincing action scenes ruined the experience.", "negative"),
        ("The soundtrack was amazing and really enhanced the emotional impact of the story.", "positive"),
        ("The music was intrusive and didn't fit the scenes at all. Very distracting.", "negative"),
        ("Excellent character development and meaningful dialogue throughout the film.", "positive"),
        ("Shallow characters and terrible dialogue made this unwatchable.", "negative"),
        
        # Genre-specific reviews
        ("This horror movie was genuinely scary with great atmosphere and suspense.", "positive"),
        ("This horror movie wasn't scary at all. Predictable and boring.", "negative"),
        ("Hilarious comedy with perfect timing and great jokes throughout.", "positive"),
        ("This comedy wasn't funny at all. Forced humor and bad timing.", "negative"),
        ("Beautiful romantic story with chemistry between the leads.", "positive"),
        ("Clichéd romance with no chemistry and predictable plot.", "negative"),
        
        # Reviews mentioning specific aspects
        ("The plot was engaging and kept me guessing until the end.", "positive"),
        ("The plot was confusing and full of holes. Made no sense.", "negative"),
        ("Outstanding performances from the entire cast.", "positive"),
        ("Terrible acting from everyone involved. Very unconvincing.", "negative"),
        ("The direction was masterful and every scene was perfectly crafted.", "positive"),
        ("Poor direction with no clear vision. Felt like amateur work.", "negative"),
    ]
    
    # Extend to 100 samples by adding variations
    extended_samples = samples.copy()
    
    # Add more variations to reach 100
    additional_positive = [
        "This film is a work of art with stunning visuals and powerful storytelling.",
        "Incredible movie that left me speechless. Absolutely brilliant!",
        "Perfect blend of action, drama, and emotion. Highly entertaining.",
        "This movie touched my heart and made me think. Truly special.",
        "Exceptional filmmaking with attention to every detail. Masterful work.",
        "I was completely absorbed in this film from start to finish.",
        "Outstanding movie with memorable characters and great dialogue.",
        "This film deserves all the praise it gets. Simply amazing.",
        "Wonderful story with excellent pacing and beautiful cinematography.",
        "This movie is a gem. Everything about it works perfectly.",
        "Captivating film that keeps you engaged throughout. Excellent work.",
        "This is filmmaking at its finest. Absolutely recommended.",
        "Great movie with strong performances and compelling story.",
        "This film exceeded all my expectations. Truly outstanding.",
        "Perfect entertainment with great acting and exciting plot.",
        "This movie is a classic in the making. Absolutely loved it.",
        "Brilliant film with innovative storytelling and great execution.",
        "This movie is pure magic. Everything clicks perfectly.",
        "Outstanding film that showcases the best of cinema.",
        "This movie is a triumph in every aspect. Highly recommended."
    ]
    
    additional_negative = [
        "This film is a complete disaster with no redeeming qualities.",
        "Absolutely terrible movie that wasted my time completely.",
        "This film fails on every level. Poorly made and boring.",
        "Horrible movie with bad acting and worse writing.",
        "This film is unwatchable garbage. Avoid at all costs.",
        "Terrible movie that makes no sense and goes nowhere.",
        "This film is a mess with poor direction and awful performances.",
        "Completely disappointing movie that fails to deliver anything good.",
        "This film is boring, confusing, and poorly executed throughout.",
        "Awful movie with no entertainment value whatsoever.",
        "This film is a waste of talent and resources. Very disappointing.",
        "Terrible movie that insults the intelligence of viewers.",
        "This film is poorly written with unconvincing performances.",
        "Horrible movie that drags on without any purpose.",
        "This film is a failure in storytelling and execution.",
        "Completely boring movie that puts you to sleep.",
        "This film is badly made with no artistic merit.",
        "Terrible movie that should never have been made.",
        "This film is a disaster from beginning to end.",
        "Awful movie that fails to engage on any level."
    ]
    
    # Add the additional samples
    for text in additional_positive:
        extended_samples.append((text, "positive"))
    
    for text in additional_negative:
        extended_samples.append((text, "negative"))
    
    # Take first 100 samples
    final_samples = extended_samples[:100]
    
    texts = [sample[0] for sample in final_samples]
    labels = [sample[1] for sample in final_samples]
    
    return texts, labels

def analyze_predictions(texts, true_labels, predictions, probabilities):
    """Analyze the model predictions and provide detailed insights."""
    
    print("\n" + "="*80)
    print("DETAILED PREDICTION ANALYSIS")
    print("="*80)
    
    # Convert string labels to binary for analysis
    true_binary = [1 if label == 'positive' else 0 for label in true_labels]
    pred_binary = [1 if pred == 'positive' else 0 for pred in predictions]
    
    # Calculate metrics
    correct = sum(1 for t, p in zip(true_binary, pred_binary) if t == p)
    accuracy = correct / len(true_binary)
    
    # Confusion matrix components
    tp = sum(1 for t, p in zip(true_binary, pred_binary) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(true_binary, pred_binary) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(true_binary, pred_binary) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(true_binary, pred_binary) if t == 1 and p == 0)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Overall Accuracy: {accuracy:.4f} ({correct}/{len(true_binary)})")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print()
    
    print("Confusion Matrix:")
    print(f"True Positive:  {tp:3d}  |  False Positive: {fp:3d}")
    print(f"False Negative: {fn:3d}  |  True Negative:  {tn:3d}")
    print()
    
    # Probability analysis
    pos_probs = [prob for i, prob in enumerate(probabilities) if true_binary[i] == 1]
    neg_probs = [prob for i, prob in enumerate(probabilities) if true_binary[i] == 0]
    
    print("Probability Distribution Analysis:")
    print(f"Positive reviews - Avg probability: {np.mean(pos_probs):.4f} (std: {np.std(pos_probs):.4f})")
    print(f"Negative reviews - Avg probability: {np.mean(neg_probs):.4f} (std: {np.std(neg_probs):.4f})")
    print()
    
    # Confidence analysis
    confidences = [abs(prob - 0.5) * 2 for prob in probabilities]
    high_conf = sum(1 for conf in confidences if conf > 0.7)
    low_conf = sum(1 for conf in confidences if conf < 0.3)
    
    print("Confidence Analysis:")
    print(f"High confidence predictions (>0.7): {high_conf}/{len(confidences)} ({high_conf/len(confidences)*100:.1f}%)")
    print(f"Low confidence predictions (<0.3): {low_conf}/{len(confidences)} ({low_conf/len(confidences)*100:.1f}%)")
    print(f"Average confidence: {np.mean(confidences):.4f}")
    print()
    
    # Show some examples
    print("SAMPLE PREDICTIONS:")
    print("-" * 80)
    
    # Show correct high-confidence predictions
    print("✅ CORRECT HIGH-CONFIDENCE PREDICTIONS:")
    correct_high_conf = []
    for i, (text, true_label, pred, prob, conf) in enumerate(zip(texts, true_labels, predictions, probabilities, confidences)):
        if true_label == pred and conf > 0.7:
            correct_high_conf.append((text, true_label, pred, prob, conf))
    
    for text, true_label, pred, prob, conf in correct_high_conf[:3]:
        print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"True: {true_label}, Predicted: {pred}, Probability: {prob:.4f}, Confidence: {conf:.4f}")
        print()
    
    # Show incorrect predictions
    print("❌ INCORRECT PREDICTIONS:")
    incorrect = []
    for i, (text, true_label, pred, prob, conf) in enumerate(zip(texts, true_labels, predictions, probabilities, confidences)):
        if true_label != pred:
            incorrect.append((text, true_label, pred, prob, conf))
    
    # Sort by confidence (most confident errors first)
    incorrect.sort(key=lambda x: x[4], reverse=True)
    
    for text, true_label, pred, prob, conf in incorrect[:5]:
        print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"True: {true_label}, Predicted: {pred}, Probability: {prob:.4f}, Confidence: {conf:.4f}")
        print()
    
    # Show low confidence predictions
    print("⚠️  LOW CONFIDENCE PREDICTIONS:")
    low_conf_examples = []
    for i, (text, true_label, pred, prob, conf) in enumerate(zip(texts, true_labels, predictions, probabilities, confidences)):
        if conf < 0.3:
            low_conf_examples.append((text, true_label, pred, prob, conf))
    
    for text, true_label, pred, prob, conf in low_conf_examples[:3]:
        print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"True: {true_label}, Predicted: {pred}, Probability: {prob:.4f}, Confidence: {conf:.4f}")
        print()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn},
        'avg_confidence': np.mean(confidences),
        'high_confidence_count': high_conf,
        'low_confidence_count': low_conf
    }

def main():
    """Main testing function."""
    print("Testing LSTM Sentiment Classifier on 100 Reviews")
    print("=" * 60)
    
    # Find the latest improved model (exclude vocabulary files)
    model_files = [f for f in os.listdir('models') 
                   if f.startswith('improved_lstm_model') 
                   and f.endswith('.pth') 
                   and 'vocabulary' not in f]
    
    if not model_files:
        print("No improved model found. Please train the model first.")
        return
    
    # Get the latest model (by timestamp in filename)
    latest_model = sorted(model_files)[-1]
    model_path = os.path.join('models', latest_model)
    
    # Find corresponding vocabulary file
    base_name = latest_model.replace('.pth', '')
    vocab_path = os.path.join('models', f"{base_name}_vocabulary.pth")
    
    if not os.path.exists(vocab_path):
        # List all vocabulary files to find the matching one
        vocab_files = [f for f in os.listdir('models') if 'vocabulary' in f and f.endswith('.pth')]
        if vocab_files:
            # Find vocabulary file with matching timestamp
            timestamp = base_name.split('_')[-2] + '_' + base_name.split('_')[-1]  # Extract timestamp
            matching_vocab = [f for f in vocab_files if timestamp in f]
            if matching_vocab:
                vocab_path = os.path.join('models', matching_vocab[0])
            else:
                # Use the latest vocabulary file as fallback
                latest_vocab = sorted(vocab_files)[-1]
                vocab_path = os.path.join('models', latest_vocab)
                print(f"Using fallback vocabulary file: {latest_vocab}")
        else:
            print("No vocabulary file found!")
            return
    
    if not os.path.exists(vocab_path):
        print(f"Vocabulary file not found: {vocab_path}")
        return
    
    print(f"Using model: {latest_model}")
    print(f"Model path: {model_path}")
    print(f"Vocab path: {vocab_path}")
    print()
    
    # Load model
    print("Loading model and vocabulary...")
    try:
        engine = create_inference_engine(model_path, vocab_path, device='cpu')
        model_info = engine.get_model_info()
        print(f"✅ Model loaded successfully!")
        print(f"Device: {model_info['device']}")
        print(f"Vocabulary size: {model_info['vocab_info']['vocab_size']}")
        print(f"Model parameters: {model_info['model_parameters']['total_parameters']:,}")
        print()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load test samples
    print("Loading test samples...")
    texts, true_labels = load_test_samples(100)
    print(f"✅ Loaded {len(texts)} test samples")
    
    # Count positive/negative samples
    pos_count = sum(1 for label in true_labels if label == 'positive')
    neg_count = len(true_labels) - pos_count
    print(f"Positive samples: {pos_count}, Negative samples: {neg_count}")
    print()
    
    # Get predictions
    print("Getting predictions...")
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
    analysis = analyze_predictions(texts, true_labels, predictions, probabilities)
    
    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    
    if analysis['accuracy'] >= 0.90:
        performance = "🎉 EXCELLENT"
    elif analysis['accuracy'] >= 0.80:
        performance = "✅ GOOD"
    elif analysis['accuracy'] >= 0.70:
        performance = "⚠️  FAIR"
    else:
        performance = "❌ POOR"
    
    print(f"Model Performance: {performance}")
    print(f"Accuracy: {analysis['accuracy']:.1%}")
    print(f"F1-Score: {analysis['f1_score']:.4f}")
    print(f"Average Confidence: {analysis['avg_confidence']:.4f}")
    print(f"High Confidence Predictions: {analysis['high_confidence_count']}/100")
    
    # Compare with the old problematic model behavior
    print("\n🔍 IMPROVEMENT CHECK:")
    prob_range = max(probabilities) - min(probabilities)
    if prob_range < 0.1:
        print("❌ WARNING: Model still showing constant predictions (like the old model)")
        print(f"   Probability range: {prob_range:.4f} (should be much higher)")
    else:
        print("✅ Model shows varied predictions (improvement from old constant behavior)")
        print(f"   Probability range: {prob_range:.4f}")
        print(f"   Min probability: {min(probabilities):.4f}")
        print(f"   Max probability: {max(probabilities):.4f}")
    
    print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()