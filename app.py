#!/usr/bin/env python3
"""
Streamlit Web UI for LSTM Sentiment Classifier

A user-friendly web interface for sentiment analysis using the trained LSTM model.
"""

import streamlit as st
import sys
import os
import glob
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from inference.inference_engine import create_inference_engine
    from models.lstm_model import LSTMClassifier
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please make sure the src directory is properly set up.")
    st.stop()


def load_available_models():
    """Load list of available trained models."""
    model_files = []
    
    # Search for model files in models directory
    for pattern in ['models/**/*.pth', 'models/*.pth']:
        model_files.extend(glob.glob(pattern, recursive=True))
    
    # Sort by modification time (newest first)
    model_files.sort(key=os.path.getmtime, reverse=True)
    
    # Filter out vocabulary files
    model_files = [f for f in model_files if 'vocabulary' not in f]
    
    return model_files


def find_vocab_file(model_path):
    """Find corresponding vocabulary file for a model."""
    base_name = model_path.replace('.pth', '')
    vocab_candidates = [
        f"{base_name}_vocabulary.pth",
        f"{base_name}_vocab.pth",
        os.path.join(os.path.dirname(model_path), "vocabulary.pth"),
        os.path.join(os.path.dirname(model_path), "vocab.pth")
    ]
    
    for vocab_path in vocab_candidates:
        if os.path.exists(vocab_path):
            return vocab_path
    
    return None


@st.cache_resource
def load_model(model_path, vocab_path):
    """Load and cache the inference engine."""
    try:
        engine = create_inference_engine(model_path, vocab_path, device='cpu')
        return engine
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


def predict_sentiment_with_details(engine, text, threshold=0.5):
    """Get detailed prediction results."""
    try:
        # Get basic prediction
        sentiment, confidence = engine.predict_sentiment(text, threshold)
        
        # Get probability
        prob_sentiment, probability, prob_confidence = engine.predict_sentiment_with_probability(text)
        
        # Get threshold analysis
        analysis = engine.predict_with_threshold_analysis(text)
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'probability': probability,
            'raw_probability': analysis['raw_probability'],
            'threshold_analysis': analysis['threshold_analysis']
        }
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None


def create_confidence_gauge(confidence):
    """Create a confidence gauge chart."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Score"},
        gauge = {
            'axis': {'range': [None, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.3], 'color': "lightgray"},
                {'range': [0.3, 0.7], 'color': "yellow"},
                {'range': [0.7, 1], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.5
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig


def create_threshold_analysis_chart(threshold_analysis):
    """Create threshold analysis chart."""
    thresholds = list(threshold_analysis.keys())
    confidences = [analysis['confidence'] for analysis in threshold_analysis.values()]
    sentiments = [analysis['sentiment'] for analysis in threshold_analysis.values()]
    
    colors = ['red' if s == 'negative' else 'green' for s in sentiments]
    
    fig = go.Figure(data=go.Bar(
        x=thresholds,
        y=confidences,
        marker_color=colors,
        text=sentiments,
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Sentiment Prediction Across Different Thresholds",
        xaxis_title="Threshold",
        yaxis_title="Confidence",
        height=400
    )
    
    return fig


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="LSTM Sentiment Analyzer",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🎬 LSTM Sentiment Analyzer")
    st.markdown("Analyze movie review sentiment using deep learning!")
    
    # Sidebar for model selection
    st.sidebar.header("Model Configuration")
    
    # Load available models
    available_models = load_available_models()
    
    if not available_models:
        st.error("No trained models found! Please train a model first.")
        st.markdown("""
        To train a model, run:
        ```bash
        python train.py --config configs/examples/quick_training.yaml
        ```
        """)
        st.stop()
    
    # Model selection
    selected_model = st.sidebar.selectbox(
        "Select Model",
        available_models,
        format_func=lambda x: os.path.basename(x)
    )
    
    # Find vocabulary file
    vocab_file = find_vocab_file(selected_model)
    
    if not vocab_file:
        st.error(f"Vocabulary file not found for model: {selected_model}")
        st.stop()
    
    st.sidebar.success(f"Model: {os.path.basename(selected_model)}")
    st.sidebar.success(f"Vocabulary: {os.path.basename(vocab_file)}")
    
    # Load model
    with st.spinner("Loading model..."):
        engine = load_model(selected_model, vocab_file)
    
    if not engine:
        st.stop()
    
    # Model info
    model_info = engine.get_model_info()
    
    with st.sidebar.expander("Model Details"):
        st.write(f"**Device:** {model_info['device']}")
        st.write(f"**Vocabulary Size:** {model_info['vocab_info']['vocab_size']:,}")
        st.write(f"**Max Sequence Length:** {model_info['vocab_info']['max_length']}")
        st.write(f"**Model Parameters:** {model_info['model_parameters']['total_parameters']:,}")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Enter Movie Review")
        
        # Text input options
        input_method = st.radio(
            "Input Method",
            ["Type Text", "Upload File", "Use Examples"],
            horizontal=True
        )
        
        text_to_analyze = ""
        
        if input_method == "Type Text":
            text_to_analyze = st.text_area(
                "Enter your movie review:",
                height=150,
                placeholder="Type your movie review here... e.g., 'This movie was absolutely amazing! The acting was superb and the plot kept me engaged throughout.'"
            )
        
        elif input_method == "Upload File":
            uploaded_file = st.file_uploader(
                "Upload a text file",
                type=['txt'],
                help="Upload a .txt file containing movie reviews (one per line)"
            )
            
            if uploaded_file:
                text_content = uploaded_file.read().decode('utf-8')
                text_to_analyze = st.text_area(
                    "File content:",
                    value=text_content,
                    height=150
                )
        
        elif input_method == "Use Examples":
            examples = {
                "Positive Review 1": "This movie was absolutely fantastic! The cinematography was breathtaking, the acting was superb, and the storyline kept me engaged from start to finish. I would definitely recommend this to anyone looking for a great film experience.",
                "Positive Review 2": "Amazing film with incredible performances. The director did an outstanding job bringing this story to life. Every scene was beautifully crafted and the emotional depth was remarkable.",
                "Negative Review 1": "This movie was a complete waste of time. The plot was confusing, the acting was terrible, and I couldn't wait for it to end. I would not recommend this to anyone.",
                "Negative Review 2": "Disappointing film with poor character development. The story dragged on and the ending was unsatisfying. The special effects couldn't save this poorly written script.",
                "Mixed Review": "The movie had some good moments but overall it was just okay. The acting was decent but the plot had some holes. It's watchable but not memorable."
            }
            
            selected_example = st.selectbox("Choose an example:", list(examples.keys()))
            text_to_analyze = st.text_area(
                "Example review:",
                value=examples[selected_example],
                height=150
            )
    
    with col2:
        st.header("Prediction Settings")
        
        threshold = st.slider(
            "Decision Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
            help="Threshold for classifying as positive vs negative"
        )
        
        show_details = st.checkbox("Show Detailed Analysis", value=True)
        show_threshold_analysis = st.checkbox("Show Threshold Analysis", value=False)
    
    # Prediction
    if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
        if not text_to_analyze.strip():
            st.warning("Please enter some text to analyze!")
        else:
            with st.spinner("Analyzing sentiment..."):
                results = predict_sentiment_with_details(engine, text_to_analyze, threshold)
            
            if results:
                # Main results
                st.header("📊 Analysis Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    sentiment_color = "green" if results['sentiment'] == 'positive' else "red"
                    st.markdown(f"""
                    <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: {sentiment_color}20; border: 2px solid {sentiment_color};">
                        <h2 style="color: {sentiment_color}; margin: 0;">
                            {'😊 POSITIVE' if results['sentiment'] == 'positive' else '😞 NEGATIVE'}
                        </h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.metric(
                        "Confidence Score",
                        f"{results['confidence']:.3f}",
                        delta=f"{results['confidence'] - 0.5:.3f}" if results['confidence'] != 0.5 else None
                    )
                
                with col3:
                    st.metric(
                        "Raw Probability",
                        f"{results['probability']:.3f}",
                        help="Raw model output probability (0=negative, 1=positive)"
                    )
                
                # Confidence gauge
                if show_details:
                    st.subheader("Confidence Visualization")
                    fig_gauge = create_confidence_gauge(results['confidence'])
                    st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Threshold analysis
                if show_threshold_analysis:
                    st.subheader("Threshold Analysis")
                    fig_threshold = create_threshold_analysis_chart(results['threshold_analysis'])
                    st.plotly_chart(fig_threshold, use_container_width=True)
                    
                    # Threshold table
                    threshold_df = pd.DataFrame([
                        {
                            'Threshold': float(t),
                            'Sentiment': analysis['sentiment'].title(),
                            'Confidence': f"{analysis['confidence']:.3f}"
                        }
                        for t, analysis in results['threshold_analysis'].items()
                    ])
                    st.dataframe(threshold_df, use_container_width=True)
    
    # Batch processing
    st.header("📝 Batch Processing")
    
    with st.expander("Process Multiple Reviews"):
        batch_text = st.text_area(
            "Enter multiple reviews (one per line):",
            height=100,
            placeholder="Review 1\nReview 2\nReview 3..."
        )
        
        if st.button("Process Batch"):
            if batch_text.strip():
                lines = [line.strip() for line in batch_text.split('\n') if line.strip()]
                
                if lines:
                    progress_bar = st.progress(0)
                    results_data = []
                    
                    for i, line in enumerate(lines):
                        result = predict_sentiment_with_details(engine, line, threshold)
                        if result:
                            results_data.append({
                                'Review': line[:100] + '...' if len(line) > 100 else line,
                                'Sentiment': result['sentiment'].title(),
                                'Confidence': result['confidence'],
                                'Probability': result['probability']
                            })
                        progress_bar.progress((i + 1) / len(lines))
                    
                    if results_data:
                        df = pd.DataFrame(results_data)
                        st.dataframe(df, use_container_width=True)
                        
                        # Summary statistics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            positive_count = len(df[df['Sentiment'] == 'Positive'])
                            st.metric("Positive Reviews", positive_count)
                        
                        with col2:
                            negative_count = len(df[df['Sentiment'] == 'Negative'])
                            st.metric("Negative Reviews", negative_count)
                        
                        with col3:
                            avg_confidence = df['Confidence'].mean()
                            st.metric("Average Confidence", f"{avg_confidence:.3f}")
                        
                        # Download results
                        csv = df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Results as CSV",
                            csv,
                            "sentiment_analysis_results.csv",
                            "text/csv"
                        )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray;">
        🎬 LSTM Sentiment Analyzer | Built with Streamlit and PyTorch
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()