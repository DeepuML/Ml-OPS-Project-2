from flask import Flask, render_template, request, jsonify
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import pickle
import os
import pandas as pd
import numpy as np
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def lemmatization(text):
    """Lemmatize the text."""
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    """Remove stop words from the text."""
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    """Remove numbers from the text."""
    text = ''.join([char for char in text if not char.isdigit()])
    return text

def lower_case(text):
    """Convert text to lower case."""
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)

def removing_punctuations(text):
    """Remove punctuations from the text."""
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def removing_urls(text):
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    """Remove sentences with less than 3 words."""
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(text):
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)

    return text


# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    logger.warning("DAGSHUB_PAT not found, will try to load model from local files")

if dagshub_token:
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "DeepuML"
repo_name = "Ml-OPS-Project-2"

# Set up MLflow tracking URI
if dagshub_token:
    mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

app = Flask(__name__)

# Global variables for model and vectorizer
model = None
vectorizer = None

def get_production_model_version(model_name):
    """Get the latest production model version"""
    try:
        client = MlflowClient()
        model_versions = client.search_model_versions(f"name='{model_name}'")
        
        # Look for production model first
        for version in model_versions:
            try:
                tags = client.get_model_version(model_name, version.version).tags
                if tags and tags.get("environment") == "production":
                    return version.version
            except:
                continue
        
        # Fallback to latest version
        if model_versions:
            latest_version = max(model_versions, key=lambda x: int(x.version))
            return latest_version.version
        
        return None
    except Exception as e:
        logger.error(f"Error getting model version: {e}")
        return None

def initialize_model():
    """Initialize model and vectorizer"""
    global model, vectorizer
    
    try:
        model_name = "my_model"
        
        if dagshub_token:
            # Try to load from MLflow registry
            model_version = get_production_model_version(model_name)
            if model_version:
                model_uri = f'models:/{model_name}/{model_version}'
                model = mlflow.pyfunc.load_model(model_uri)
                logger.info(f"✅ Loaded production model v{model_version} from MLflow")
            else:
                logger.warning("No model found in registry, loading from local files")
                model = pickle.load(open('../models/model.pkl', 'rb'))
        else:
            # Load from local files
            logger.info("Loading model from local files")
            model = pickle.load(open('../models/model.pkl', 'rb'))
        
        # Load vectorizer
        vectorizer = pickle.load(open('../models/vectorizer.pkl', 'rb'))
        logger.info("✅ Model and vectorizer loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Initialize with None for error handling
        model = None
        vectorizer = None

# Initialize model when app starts
initialize_model()

@app.route('/')
def home():
    return render_template('index.html',result=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if model is loaded
        if model is None or vectorizer is None:
            return render_template('index.html', 
                                 result=None, 
                                 error="Model not loaded. Please check server logs.")
        
        # Get text from form
        text = request.form.get('text', '').strip()
        
        if not text:
            return render_template('index.html', 
                                 result=None, 
                                 error="Please enter some text to analyze.")
        
        # Clean and preprocess text
        cleaned_text = normalize_text(text)
        
        if not cleaned_text or len(cleaned_text.split()) < 1:
            return render_template('index.html', 
                                 result=None, 
                                 error="Text too short or invalid after preprocessing.")
        
        # Vectorize text
        features = vectorizer.transform([cleaned_text])
        
        # Convert to DataFrame for MLflow model
        features_df = pd.DataFrame(features.toarray(), 
                                 columns=[str(i) for i in range(features.shape[1])])
        
        # Make prediction
        if hasattr(model, 'predict'):
            if str(type(model)).find('mlflow') != -1:
                # MLflow model
                prediction = model.predict(features_df)
            else:
                # Sklearn model
                prediction = model.predict(features.toarray())
        else:
            raise Exception("Model does not have predict method")
        
        # Convert prediction to sentiment
        sentiment = "Positive 😊" if prediction[0] == 1 else "Negative 😔"
        confidence = "High" if abs(prediction[0] - 0.5) > 0.3 else "Medium"
        
        result = {
            'original_text': text,
            'cleaned_text': cleaned_text,
            'sentiment': sentiment,
            'prediction_value': int(prediction[0]),
            'confidence': confidence
        }
        
        logger.info(f"Prediction made: {text[:50]}... -> {sentiment}")
        
        return render_template('index.html', result=result, error=None)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return render_template('index.html', 
                             result=None, 
                             error=f"Prediction failed: {str(e)}")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for programmatic access"""
    try:
        if model is None or vectorizer is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({'error': 'Empty text provided'}), 400
        
        # Preprocess and predict
        cleaned_text = normalize_text(text)
        features = vectorizer.transform([cleaned_text])
        features_df = pd.DataFrame(features.toarray(), 
                                 columns=[str(i) for i in range(features.shape[1])])
        
        if hasattr(model, 'predict'):
            if str(type(model)).find('mlflow') != -1:
                prediction = model.predict(features_df)
            else:
                prediction = model.predict(features.toarray())
        
        sentiment = "positive" if prediction[0] == 1 else "negative"
        
        return jsonify({
            'text': text,
            'sentiment': sentiment,
            'prediction': int(prediction[0]),
            'confidence': 'high' if abs(prediction[0] - 0.5) > 0.3 else 'medium'
        })
        
    except Exception as e:
        logger.error(f"API prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    status = {
        'status': 'healthy' if (model is not None and vectorizer is not None) else 'unhealthy',
        'model_loaded': model is not None,
        'vectorizer_loaded': vectorizer is not None
    }
    return jsonify(status)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")