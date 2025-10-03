#!/usr/bin/env python3
"""
Test script to verify DagsHub and MLflow connectivity
"""

import os
import sys
from dotenv import load_dotenv
import mlflow
from mlflow.tracking import MlflowClient

def test_dagshub_connection():
    """Test DagsHub connection and MLflow setup"""
    
    # Load environment variables
    load_dotenv()
    
    # Check if DAGSHUB_PAT is set
    dagshub_token = os.getenv("DAGSHUB_PAT")
    if not dagshub_token:
        print("❌ Error: DAGSHUB_PAT environment variable is not set")
        return False
    
    print("✅ DAGSHUB_PAT is configured")
    
    # Set up MLflow credentials
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
    
    # Set MLflow tracking URI
    dagshub_url = "https://dagshub.com"
    repo_owner = "DeepuML"
    repo_name = "Ml-OPS-Project-2"
    
    tracking_uri = f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow"
    mlflow.set_tracking_uri(tracking_uri)
    
    print(f"✅ MLflow tracking URI set to: {tracking_uri}")
    
    try:
        # Test connection by trying to list experiments
        client = MlflowClient()
        experiments = client.search_experiments()
        print(f"✅ Successfully connected to DagsHub. Found {len(experiments)} experiments.")
        return True
        
    except Exception as e:
        print(f"❌ Error connecting to DagsHub: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_dagshub_connection()
    sys.exit(0 if success else 1)