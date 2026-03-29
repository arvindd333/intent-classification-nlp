# Open Intent Detection Project

A comprehensive machine learning-based intent detection system that can classify user intents from text input. This project implements multiple approaches including traditional ML models and deep learning techniques.

## Features

- **Multiple Model Approaches**: Traditional ML (SVM, Random Forest) and Deep Learning (BERT, LSTM)
- **Interactive Web Interface**: Streamlit-based UI for easy testing
- **Data Preprocessing**: Text cleaning, tokenization, and feature extraction
- **Model Evaluation**: Comprehensive metrics and visualization
- **Real-time Prediction**: Live intent classification
- **Model Persistence**: Save and load trained models

## Project Structure

```
├── data/                   # Dataset files
├── models/                 # Trained model files
├── src/                    # Source code
│   ├── preprocessing.py    # Text preprocessing utilities
│   ├── models.py          # Model implementations
│   ├── training.py        # Training pipeline
│   └── evaluation.py      # Model evaluation
├── notebooks/             # Jupyter notebooks for exploration
├── app.py                 # Streamlit web application
├── main.py                # Main execution script
└── requirements.txt       # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd open-intent-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required NLTK data:
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## Usage

### Quick Start
```bash
python main.py
```

### Web Interface
```bash
streamlit run app.py
```

### Training Custom Models
```python
from src.training import train_model
from src.models import IntentClassifier

# Train a new model
model = train_model(data_path='data/intents.json', model_type='bert')
```

## Supported Intent Types

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

## Model Performance

- **BERT Model**: ~95% accuracy
- **LSTM Model**: ~92% accuracy  
- **SVM Model**: ~88% accuracy
- **Random Forest**: ~85% accuracy

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details 