import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import pickle
import os
import sys
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import create_training_data, split_data, FeatureExtractor, TextPreprocessor
from models import IntentClassifierFactory, evaluate_model
from training import IntentDetectionTrainer
from evaluation import ModelEvaluator

# Page configuration
st.set_page_config(
    page_title="Open Intent Detection",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_intent_data():
    """Load intent data from JSON file"""
    with open('data/intents.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    responses = {}
    for intent in data['intents']:
        responses[intent['tag']] = intent['responses']
    
    return data, responses

@st.cache_data
def load_training_data():
    """Load and cache training data"""
    return create_training_data('data/intents.json')

@st.cache_resource
def load_model(model_type='svm'):
    """Load trained model"""
    model_path = f'models/{model_type}_model.pkl'
    feature_extractor_path = f'models/{model_type}_feature_extractor.pkl'
    
    if os.path.exists(model_path) and os.path.exists(feature_extractor_path):
        # Load model
        model = IntentClassifierFactory.create_classifier(model_type)
        model.load_model(model_path)
        
        # Load feature extractor
        with open(feature_extractor_path, 'rb') as f:
            feature_extractor = pickle.load(f)
        
        return model, feature_extractor
    return None, None

def predict_intent(text, model, feature_extractor=None, preprocessor=None):
    """Predict intent for a given text"""
    if preprocessor is None:
        preprocessor = TextPreprocessor()
    
    # Preprocess text
    processed_text = preprocessor.preprocess_text(text)
    
    # Transform features if using traditional ML model
    if feature_extractor is not None:
        X = feature_extractor.transform_texts([processed_text])
    else:
        X = [processed_text]
    
    # Predict
    intent = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0] if hasattr(model, 'predict_proba') else None
    
    return intent, probabilities, processed_text

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Open Intent Detection System</h1>', unsafe_allow_html=True)
    
    # Load data
    intent_data, responses = load_intent_data()
    df = load_training_data()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "🔍 Intent Detection", "📊 Model Training", "📈 Analytics", "ℹ️ About"]
    )
    
    if page == "🏠 Home":
        show_home_page(df, intent_data)
    elif page == "🔍 Intent Detection":
        show_intent_detection_page(responses)
    elif page == "📊 Model Training":
        show_training_page(df)
    elif page == "📈 Analytics":
        show_analytics_page(df)
    elif page == "ℹ️ About":
        show_about_page()

def show_home_page(df, intent_data):
    """Home page with overview"""
    st.header("Welcome to Open Intent Detection System")
    
    # Overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", len(df))
    
    with col2:
        st.metric("Intent Classes", df['label'].nunique())
    
    with col3:
        st.metric("Available Models", len([f for f in os.listdir('models') if f.endswith('_model.pkl')]))
    
    with col4:
        avg_length = df['text'].str.len().mean()
        st.metric("Avg Text Length", f"{avg_length:.1f}")
    
    # Dataset overview
    st.subheader("📋 Dataset Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Intent distribution
        intent_counts = df['label'].value_counts()
        fig = px.pie(
            values=intent_counts.values,
            names=intent_counts.index,
            title="Intent Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Text length distribution
        text_lengths = df['text'].str.len()
        fig = px.histogram(
            x=text_lengths,
            title="Text Length Distribution",
            labels={'x': 'Text Length', 'y': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Sample data
    st.subheader("📝 Sample Training Data")
    st.dataframe(df.head(10), use_container_width=True)

def show_intent_detection_page(responses):
    """Intent detection page"""
    st.header("🔍 Intent Detection")
    
    # Model selection
    model_type = st.selectbox(
        "Select Model:",
        ["svm", "random_forest", "lstm"],
        help="Choose the model to use for intent detection"
    )
    
    # Load model
    model, feature_extractor = load_model(model_type)
    
    if model is None:
        st.error(f"No trained {model_type.upper()} model found. Please train models first.")
        if st.button("Train Models Now"):
            with st.spinner("Training models..."):
                trainer = IntentDetectionTrainer()
                trainer.load_and_prepare_data()
                trainer.train_all_models()
            st.success("Models trained successfully! Please refresh the page.")
        return
    
    st.success(f"✅ Loaded {model_type.upper()} model successfully!")
    
    # Input section
    st.subheader("Enter your text:")
    
    # Text input
    user_input = st.text_area(
        "Type your message here:",
        placeholder="e.g., Hello! How are you doing today?",
        height=100
    )
    
    # Predict button
    if st.button("🔍 Detect Intent", type="primary"):
        if user_input.strip():
            with st.spinner("Analyzing..."):
                intent, probabilities, processed_text = predict_intent(
                    user_input, model, feature_extractor
                )
            
            # Results
            st.subheader("🎯 Detection Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.write(f"**Detected Intent:** {intent.upper()}")
                if probabilities is not None:
                    confidence = max(probabilities)
                    st.write(f"**Confidence:** {confidence:.3f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                # Get response
                if intent in responses:
                    import random
                    response = random.choice(responses[intent])
                    st.write(f"**Response:** {response}")
                else:
                    st.write("**Response:** I'm not sure how to respond to that.")
            
            # Show processed text
            with st.expander("🔧 Text Processing Details"):
                st.write(f"**Original:** {user_input}")
                st.write(f"**Processed:** {processed_text}")
            
            # Show probabilities if available
            if probabilities is not None:
                st.subheader("📊 Confidence Scores")
                
                # Get class names
                class_names = model.label_encoder.classes_
                
                # Create probability chart
                prob_df = pd.DataFrame({
                    'Intent': class_names,
                    'Probability': probabilities
                }).sort_values('Probability', ascending=True)
                
                fig = px.bar(
                    prob_df,
                    x='Probability',
                    y='Intent',
                    orientation='h',
                    title="Intent Probabilities"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please enter some text to analyze.")

def show_training_page(df):
    """Model training page"""
    st.header("📊 Model Training")
    
    # Training options
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Train Models")
        
        if st.button("🚀 Train All Models", type="primary"):
            with st.spinner("Training models... This may take a few minutes."):
                trainer = IntentDetectionTrainer()
                trainer.load_and_prepare_data()
                trainer.train_all_models()
                
                # Compare models
                comparison_df = trainer.compare_models()
                
                st.success("✅ All models trained successfully!")
                
                # Show results
                st.subheader("📈 Training Results")
                st.dataframe(comparison_df, use_container_width=True)
    
    with col2:
        st.subheader("Model Status")
        
        model_files = [f for f in os.listdir('models') if f.endswith('_model.pkl')]
        
        if model_files:
            st.success(f"✅ {len(model_files)} trained models available")
            for model_file in model_files:
                st.write(f"• {model_file}")
        else:
            st.warning("⚠️ No trained models found")
    
    # Dataset information
    st.subheader("📋 Training Dataset")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Total samples:** {len(df)}")
        st.write(f"**Training samples:** {int(len(df) * 0.8)}")
        st.write(f"**Test samples:** {int(len(df) * 0.2)}")
    
    with col2:
        st.write(f"**Intent classes:** {df['label'].nunique()}")
        st.write(f"**Average text length:** {df['text'].str.len().mean():.1f}")
        st.write(f"**Unique words:** {len(set(' '.join(df['text']).split()))}")

def show_analytics_page(df):
    """Analytics page"""
    st.header("📈 Analytics")
    
    # Intent distribution
    st.subheader("Intent Distribution")
    
    intent_counts = df['label'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(
            values=intent_counts.values,
            names=intent_counts.index,
            title="Intent Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            x=intent_counts.index,
            y=intent_counts.values,
            title="Intent Counts",
            labels={'x': 'Intent', 'y': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Text analysis
    st.subheader("Text Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Text length by intent
        df['text_length'] = df['text'].str.len()
        fig = px.box(
            df,
            x='label',
            y='text_length',
            title="Text Length by Intent"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Word count by intent
        df['word_count'] = df['text'].str.split().str.len()
        fig = px.box(
            df,
            x='label',
            y='word_count',
            title="Word Count by Intent"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Sample texts
    st.subheader("Sample Texts by Intent")
    
    for intent in df['label'].unique():
        with st.expander(f"📝 {intent.upper()}"):
            samples = df[df['label'] == intent]['text'].head(5).tolist()
            for i, sample in enumerate(samples, 1):
                st.write(f"{i}. {sample}")

def show_about_page():
    """About page"""
    st.header("ℹ️ About")
    
    st.markdown("""
    ## Open Intent Detection System
    
    This is a comprehensive machine learning-based intent detection system that can classify user intents from text input.
    
    ### Features
    
    - **Multiple Model Approaches**: Traditional ML (SVM, Random Forest) and Deep Learning (LSTM)
    - **Interactive Web Interface**: Beautiful Streamlit-based UI for easy testing
    - **Data Preprocessing**: Text cleaning, tokenization, and feature extraction
    - **Model Evaluation**: Comprehensive metrics and visualization
    - **Real-time Prediction**: Live intent classification
    - **Model Persistence**: Save and load trained models
    
    ### Supported Intent Types
    
    - **Greeting**: Hello, hi, good morning
    - **Farewell**: Goodbye, bye, see you later
    - **Thanks**: Thank you, thanks, appreciate it
    - **Help**: Help, support, assistance
    - **Weather**: Weather forecast, temperature
    - **Time**: Current time, what time is it
    - **Jokes**: Tell me a joke, funny story
    - **Music**: Play music, song recommendation
    - **News**: Latest news, current events
    - **Unknown**: Unrecognized intents
    
    ### Technology Stack
    
    - **Python**: Core programming language
    - **Scikit-learn**: Traditional machine learning models
    - **TensorFlow/Keras**: Deep learning models
    - **NLTK**: Natural language processing
    - **Streamlit**: Web interface
    - **Plotly**: Interactive visualizations
    - **Pandas**: Data manipulation
    - **NumPy**: Numerical computing
    
    ### Usage
    
    1. **Install Dependencies**: `pip install -r requirements.txt`
    2. **Run Web App**: `streamlit run app.py`
    3. **Train Models**: Use the training page or run `python main.py`
    4. **Test Intent Detection**: Use the interactive demo
    
    ### Project Structure
    
    ```
    ├── data/                   # Dataset files
    ├── models/                 # Trained model files
    ├── src/                    # Source code
    │   ├── preprocessing.py    # Text preprocessing utilities
    │   ├── models.py          # Model implementations
    │   ├── training.py        # Training pipeline
    │   └── evaluation.py      # Model evaluation
    ├── app.py                 # Streamlit web application
    ├── main.py                # Main execution script
    └── requirements.txt       # Python dependencies
    ```
    
    ### Contributing
    
    This is an open-source project. Feel free to contribute by:
    - Adding new intent types
    - Improving model performance
    - Enhancing the user interface
    - Adding new features
    
    ### License
    
    MIT License - see LICENSE file for details
    """)

if __name__ == "__main__":
    main() 