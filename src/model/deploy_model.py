#!/usr/bin/env python3
"""
Simple Model Deployment Script
Demonstrates how to deploy a production model for inference
"""

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import os
import pickle
import pandas as pd
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalysisService:
    """Production sentiment analysis service"""
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.setup_mlflow()
        self.load_production_model()
    
    def setup_mlflow(self):
        """Setup MLflow connection"""
        load_dotenv()
        
        dagshub_token = os.getenv("DAGSHUB_PAT")
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "DeepuML"
        repo_name = "Ml-OPS-Project-2"

        mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")
    
    def load_production_model(self):
        """Load the production model from MLflow registry"""
        try:
            client = MlflowClient()
            model_name = "my_model"
            
            # Get all model versions
            model_versions = client.search_model_versions(f"name='{model_name}'")
            
            # Find production model
            production_model = None
            for version in model_versions:
                tags = {tag.key: tag.value for tag in version.tags}
                if tags.get("environment") == "production":
                    production_model = version
                    break
            
            if production_model:
                model_uri = f"models:/{model_name}/{production_model.version}"
                self.model = mlflow.pyfunc.load_model(model_uri)
                logger.info(f"✅ Loaded production model v{production_model.version}")
            else:
                # Fallback to local files
                logger.warning("No production model found, using local files")
                self.model = pickle.load(open('models/model.pkl', 'rb'))
            
            # Load vectorizer
            self.vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))
            logger.info("✅ Model service ready for inference")
            
        except Exception as e:
            logger.error(f"Failed to load production model: {e}")
            raise
    
    def predict(self, text_input):
        """Predict sentiment for text input"""
        try:
            # Handle single string or list of strings
            if isinstance(text_input, str):
                texts = [text_input]
            else:
                texts = text_input
            
            # Vectorize input
            X = self.vectorizer.transform(texts)
            
            # Make prediction
            if hasattr(self.model, 'predict'):
                if str(type(self.model)).find('mlflow') != -1:
                    # MLflow model expects DataFrame
                    X_df = pd.DataFrame(X.toarray(), columns=[str(i) for i in range(X.shape[1])])
                    predictions = self.model.predict(X_df)
                else:
                    # Sklearn model
                    predictions = self.model.predict(X.toarray())
            
            # Convert to sentiment labels
            results = []
            for i, pred in enumerate(predictions):
                sentiment = "Positive" if pred == 1 else "Negative"
                confidence = "High"  # You could add confidence scores here
                results.append({
                    "text": texts[i],
                    "sentiment": sentiment,
                    "prediction": int(pred),
                    "confidence": confidence
                })
            
            return results[0] if isinstance(text_input, str) else results
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

def main():
    """Demo deployment service"""
    try:
        # Initialize service
        service = SentimentAnalysisService()
        
        # Test predictions
        test_texts = [
            "I love this product! It's amazing!",
            "This is terrible, I hate it.",
            "The weather is okay today.",
            "I'm so excited about this new feature!"
        ]
        
        logger.info("🚀 Production Model Deployment Demo")
        logger.info("=" * 50)
        
        for text in test_texts:
            result = service.predict(text)
            logger.info(f"Text: '{result['text']}'")
            logger.info(f"Sentiment: {result['sentiment']} ({result['confidence']} confidence)")
            logger.info("-" * 30)
        
        logger.info("✅ Deployment successful! Model is serving predictions.")
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()