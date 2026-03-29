#!/usr/bin/env python3
"""
Main execution script for Open Intent Detection Project
"""

import os
import sys
import json
import pickle
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import create_training_data, split_data, FeatureExtractor, TextPreprocessor
from models import IntentClassifierFactory, evaluate_model
from training import IntentDetectionTrainer
from evaluation import ModelEvaluator

def load_intent_responses():
    """Load intent responses from JSON file"""
    with open('data/intents.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    responses = {}
    for intent in data['intents']:
        responses[intent['tag']] = intent['responses']
    
    return responses

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
    
    return intent, probabilities

def interactive_demo():
    """Interactive demo of the intent detection system"""
    print("="*60)
    print("OPEN INTENT DETECTION SYSTEM - INTERACTIVE DEMO")
    print("="*60)
    
    # Check if trained models exist
    model_files = [f for f in os.listdir('models') if f.endswith('_model.pkl')]
    
    if not model_files:
        print("No trained models found. Training models first...")
        trainer = IntentDetectionTrainer()
        trainer.load_and_prepare_data()
        trainer.train_all_models()
        model_files = [f for f in os.listdir('models') if f.endswith('_model.pkl')]
    
    # Load the best model (SVM for demo)
    best_model_file = 'svm_model.pkl'
    if best_model_file in model_files:
        model_path = os.path.join('models', best_model_file)
        feature_extractor_path = os.path.join('models', 'svm_feature_extractor.pkl')
        
        # Load model
        model = IntentClassifierFactory.create_classifier('svm')
        model.load_model(model_path)
        
        # Load feature extractor
        with open(feature_extractor_path, 'rb') as f:
            feature_extractor = pickle.load(f)
        
        # Load responses
        responses = load_intent_responses()
        
        print(f"Loaded model: {best_model_file}")
        print("Type 'quit' to exit the demo")
        print("-"*60)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("Goodbye! Thanks for trying the intent detection system.")
                    break
                
                if not user_input:
                    continue
                
                # Predict intent
                intent, probabilities = predict_intent(user_input, model, feature_extractor)
                
                # Get response
                if intent in responses:
                    import random
                    response = random.choice(responses[intent])
                else:
                    response = "I'm not sure how to respond to that."
                
                print(f"Detected Intent: {intent}")
                if probabilities is not None:
                    print(f"Confidence: {max(probabilities):.3f}")
                print(f"Response: {response}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                print("Please try again.")

def train_and_evaluate():
    """Train models and evaluate performance"""
    print("="*60)
    print("TRAINING AND EVALUATION PIPELINE")
    print("="*60)
    
    # Initialize trainer
    trainer = IntentDetectionTrainer()
    
    # Load and prepare data
    trainer.load_and_prepare_data()
    
    # Train all models
    trainer.train_all_models()
    
    # Compare models
    comparison_df = trainer.compare_models()
    
    # Plot results
    trainer.plot_results()
    
    # Save training summary
    trainer.save_training_summary()
    
    print("\nTraining and evaluation completed!")
    print("Check the 'models' directory for saved models and results.")

def test_specific_model(model_type='svm'):
    """Test a specific model type"""
    print(f"Testing {model_type.upper()} model...")
    
    # Load data
    df = create_training_data('data/intents.json')
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Prepare features
    feature_extractor = FeatureExtractor()
    feature_extractor.fit_vectorizer(X_train)
    
    X_train_features = feature_extractor.transform_texts(X_train)
    X_test_features = feature_extractor.transform_texts(X_test)
    
    # Create and train model
    model = IntentClassifierFactory.create_classifier(model_type)
    
    if model_type.lower() == 'lstm':
        model.fit(X_train, y_train, epochs=10)  # Fewer epochs for testing
        results = evaluate_model(model, X_test, y_test)
    else:
        model.fit(X_train_features, y_train)
        results = evaluate_model(model, X_test_features, y_test)
    
    print(f"\n{model_type.upper()} Model Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['classification_report']['weighted avg']['precision']:.4f}")
    print(f"Recall: {results['classification_report']['weighted avg']['recall']:.4f}")
    print(f"F1-Score: {results['classification_report']['weighted avg']['f1-score']:.4f}")
    
    return model, results

def show_dataset_info():
    """Display information about the dataset"""
    print("="*60)
    print("DATASET INFORMATION")
    print("="*60)
    
    # Load data
    df = create_training_data('data/intents.json')
    
    print(f"Total samples: {len(df)}")
    print(f"Number of intent classes: {df['label'].nunique()}")
    print(f"Classes: {list(df['label'].unique())}")
    
    print("\nIntent distribution:")
    intent_counts = df['label'].value_counts()
    for intent, count in intent_counts.items():
        print(f"  {intent}: {count} samples")
    
    print("\nSample texts:")
    for intent in df['label'].unique():
        sample = df[df['label'] == intent]['text'].iloc[0]
        print(f"  {intent}: '{sample}'")

def main():
    """Main function with menu options"""
    while True:
        print("\n" + "="*60)
        print("OPEN INTENT DETECTION PROJECT")
        print("="*60)
        print("1. Interactive Demo")
        print("2. Train and Evaluate All Models")
        print("3. Test Specific Model")
        print("4. Show Dataset Information")
        print("5. Exit")
        print("-"*60)
        
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == '1':
            interactive_demo()
        elif choice == '2':
            train_and_evaluate()
        elif choice == '3':
            print("\nAvailable models: svm, random_forest, lstm")
            model_type = input("Enter model type: ").strip().lower()
            if model_type in ['svm', 'random_forest', 'lstm']:
                test_specific_model(model_type)
            else:
                print("Invalid model type!")
        elif choice == '4':
            show_dataset_info()
        elif choice == '5':
            print("Goodbye!")
            break
        else:
            print("Invalid choice! Please enter a number between 1-5.")

if __name__ == "__main__":
    main() 