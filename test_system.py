#!/usr/bin/env python3
"""
Simple test script for the Open Intent Detection System
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test if all modules can be imported"""
    print("Testing imports...")
    
    try:
        from preprocessing import TextPreprocessor, FeatureExtractor, create_training_data
        print("✅ Preprocessing module imported successfully")
    except Exception as e:
        print(f"❌ Error importing preprocessing: {e}")
        return False
    
    try:
        from models import IntentClassifierFactory, SVMIntentClassifier
        print("✅ Models module imported successfully")
    except Exception as e:
        print(f"❌ Error importing models: {e}")
        return False
    
    try:
        from training import IntentDetectionTrainer
        print("✅ Training module imported successfully")
    except Exception as e:
        print(f"❌ Error importing training: {e}")
        return False
    
    return True

def test_preprocessing():
    """Test text preprocessing"""
    print("\nTesting text preprocessing...")
    
    try:
        preprocessor = TextPreprocessor()
        test_text = "Hello! How are you doing today? I need some help."
        
        cleaned = preprocessor.clean_text(test_text)
        processed = preprocessor.preprocess_text(test_text)
        sentiment = preprocessor.get_sentiment(test_text)
        features = preprocessor.extract_features(test_text)
        
        print(f"✅ Original text: {test_text}")
        print(f"✅ Cleaned text: {cleaned}")
        print(f"✅ Processed text: {processed}")
        print(f"✅ Sentiment: {sentiment}")
        print(f"✅ Features: {features}")
        
        return True
    except Exception as e:
        print(f"❌ Error in preprocessing: {e}")
        return False

def test_data_loading():
    """Test data loading"""
    print("\nTesting data loading...")
    
    try:
        df = create_training_data('data/intents.json')
        print(f"✅ Loaded {len(df)} training examples")
        print(f"✅ Number of intent classes: {df['label'].nunique()}")
        print(f"✅ Intent classes: {list(df['label'].unique())}")
        
        return True
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

def test_model_creation():
    """Test model creation"""
    print("\nTesting model creation...")
    
    try:
        # Test SVM
        svm_model = IntentClassifierFactory.create_classifier('svm')
        print("✅ SVM model created successfully")
        
        # Test Random Forest
        rf_model = IntentClassifierFactory.create_classifier('random_forest')
        print("✅ Random Forest model created successfully")
        
        return True
    except Exception as e:
        print(f"❌ Error creating models: {e}")
        return False

def test_simple_prediction():
    """Test simple prediction with SVM"""
    print("\nTesting simple prediction...")
    
    try:
        # Load data
        df = create_training_data('data/intents.json')
        
        # Create feature extractor
        feature_extractor = FeatureExtractor()
        feature_extractor.fit_vectorizer(df['cleaned_text'])
        
        # Create and train SVM model
        model = IntentClassifierFactory.create_classifier('svm')
        X_features = feature_extractor.transform_texts(df['cleaned_text'])
        model.fit(X_features, df['label'])
        
        # Test prediction
        test_text = "Hello there!"
        preprocessor = TextPreprocessor()
        processed_text = preprocessor.preprocess_text(test_text)
        X_test = feature_extractor.transform_texts([processed_text])
        
        prediction = model.predict(X_test)[0]
        probabilities = model.predict_proba(X_test)[0]
        
        print(f"✅ Test text: {test_text}")
        print(f"✅ Predicted intent: {prediction}")
        print(f"✅ Confidence: {max(probabilities):.3f}")
        
        return True
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("OPEN INTENT DETECTION SYSTEM - TEST SCRIPT")
    print("="*60)
    
    tests = [
        ("Import Test", test_imports),
        ("Preprocessing Test", test_preprocessing),
        ("Data Loading Test", test_data_loading),
        ("Model Creation Test", test_model_creation),
        ("Simple Prediction Test", test_simple_prediction)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed!")
    
    print("\n" + "="*60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is working correctly.")
        print("\nNext steps:")
        print("1. Run 'python main.py' for interactive demo")
        print("2. Run 'streamlit run app.py' for web interface")
        print("3. Check the README.md for more information")
    else:
        print("⚠️ Some tests failed. Please check the error messages above.")
    
    print("="*60)

if __name__ == "__main__":
    main() 