#!/usr/bin/env python3
"""
Setup script for Open Intent Detection Project
"""

import os
import sys
import subprocess
import platform

def print_banner():
    """Print welcome banner"""
    print("="*60)
    print("🤖 OPEN INTENT DETECTION PROJECT SETUP")
    print("="*60)
    print("This script will help you set up the intent detection system.")
    print()

def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("❌ Python 3.7 or higher is required!")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    else:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True

def install_dependencies():
    """Install required dependencies"""
    print("\nInstalling dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def download_nltk_data():
    """Download required NLTK data"""
    print("\nDownloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        print("✅ NLTK data downloaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error downloading NLTK data: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\nCreating directories...")
    directories = ['models', 'evaluation_results']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"✅ Directory already exists: {directory}")

def test_imports():
    """Test if all modules can be imported"""
    print("\nTesting imports...")
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        
        from preprocessing import TextPreprocessor, FeatureExtractor
        from models import IntentClassifierFactory
        from training import IntentDetectionTrainer
        
        print("✅ All modules imported successfully!")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def run_quick_test():
    """Run a quick test to verify everything works"""
    print("\nRunning quick test...")
    try:
        # Test data loading
        from preprocessing import create_training_data
        df = create_training_data('data/intents.json')
        print(f"✅ Loaded {len(df)} training examples")
        
        # Test preprocessing
        preprocessor = TextPreprocessor()
        test_text = "Hello! How are you?"
        processed = preprocessor.preprocess_text(test_text)
        print(f"✅ Text preprocessing works: '{test_text}' -> '{processed}'")
        
        # Test model creation
        model = IntentClassifierFactory.create_classifier('svm')
        print("✅ Model creation works")
        
        print("✅ Quick test passed!")
        return True
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

def show_next_steps():
    """Show next steps for the user"""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nNext steps:")
    print("1. Test the system:")
    print("   python test_system.py")
    print()
    print("2. Run interactive demo:")
    print("   python main.py")
    print()
    print("3. Launch web interface:")
    print("   streamlit run app.py")
    print()
    print("4. Explore with Jupyter notebook:")
    print("   jupyter notebook notebooks/intent_detection_exploration.ipynb")
    print()
    print("5. Train models:")
    print("   python main.py (choose option 2)")
    print()
    print("For more information, check the README.md file.")
    print("="*60)

def main():
    """Main setup function"""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        return
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed! Please check the error messages above.")
        return
    
    # Download NLTK data
    if not download_nltk_data():
        print("\n❌ Setup failed! Please check the error messages above.")
        return
    
    # Create directories
    create_directories()
    
    # Test imports
    if not test_imports():
        print("\n❌ Setup failed! Please check the error messages above.")
        return
    
    # Run quick test
    if not run_quick_test():
        print("\n❌ Setup failed! Please check the error messages above.")
        return
    
    # Show next steps
    show_next_steps()

if __name__ == "__main__":
    main() 