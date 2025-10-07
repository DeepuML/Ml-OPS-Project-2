#!/usr/bin/env python3
"""
Model Promotion Script
Automatically promotes models from staging to production based on performance criteria
"""

import mlflow
from mlflow.tracking import MlflowClient
import os
import json
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_mlflow():
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

def get_model_metrics(run_id):
    """Get metrics for a specific model run"""
    client = MlflowClient()
    try:
        run = client.get_run(run_id)
        return run.data.metrics
    except Exception as e:
        logger.error(f"Could not get metrics for run {run_id}: {e}")
        return {}

def check_promotion_criteria(metrics):
    """Check if model meets criteria for production promotion"""
    criteria = {
        'accuracy': 0.75,      # Must be at least 75% accurate
        'precision': 0.75,     # Must be at least 75% precise
        'recall': 0.70,        # Must be at least 70% recall
        'auc': 0.75           # Must be at least 75% AUC
    }
    
    passed_criteria = []
    failed_criteria = []
    
    for metric, threshold in criteria.items():
        if metric in metrics:
            if metrics[metric] >= threshold:
                passed_criteria.append(f"✅ {metric}: {metrics[metric]:.4f} >= {threshold}")
            else:
                failed_criteria.append(f"❌ {metric}: {metrics[metric]:.4f} < {threshold}")
        else:
            failed_criteria.append(f"❌ {metric}: Not found in metrics")
    
    return len(failed_criteria) == 0, passed_criteria, failed_criteria

def promote_model_to_production(model_name, model_version, run_id):
    """Promote model to production if it meets criteria"""
    client = MlflowClient()
    
    # Get model metrics
    metrics = get_model_metrics(run_id)
    
    # Check promotion criteria
    can_promote, passed, failed = check_promotion_criteria(metrics)
    
    logger.info(f"Model Promotion Assessment for {model_name} v{model_version}:")
    logger.info("Passed Criteria:")
    for item in passed:
        logger.info(f"  {item}")
    
    if failed:
        logger.info("Failed Criteria:")
        for item in failed:
            logger.info(f"  {item}")
    
    if can_promote:
        # Update model tags for production
        client.set_model_version_tag(
            name=model_name,
            version=model_version,
            key="environment",
            value="production"
        )
        
        client.set_model_version_tag(
            name=model_name,
            version=model_version,
            key="promotion_status",
            value="approved"
        )
        
        client.set_model_version_tag(
            name=model_name,
            version=model_version,
            key="deployed_at",
            value=str(pd.Timestamp.now())
        )
        
        client.update_model_version(
            name=model_name,
            version=model_version,
            description="Model promoted to production - meets all quality criteria"
        )
        
        logger.info(f"🎉 Model {model_name} v{model_version} PROMOTED TO PRODUCTION!")
        return True
    else:
        logger.info(f"🚨 Model {model_name} v{model_version} NOT PROMOTED - Failed quality criteria")
        return False

def main():
    """Main promotion workflow"""
    try:
        setup_mlflow()
        
        model_name = "my_model"
        client = MlflowClient()
        
        # Get latest model version
        model_versions = client.search_model_versions(f"name='{model_name}'")
        
        if model_versions:
            latest_version = max(model_versions, key=lambda x: int(x.version))
            model_version = latest_version.version
            run_id = latest_version.run_id
            
            logger.info(f"Evaluating model {model_name} version {model_version} for production promotion...")
            logger.info(f"Using run_id: {run_id}")
            
            success = promote_model_to_production(model_name, model_version, run_id)
            
            if success:
                print("✅ MODEL PROMOTED TO PRODUCTION")
                exit(0)
            else:
                print("❌ MODEL NOT PROMOTED - QUALITY CRITERIA NOT MET")
                exit(1)
        else:
            logger.error(f"No versions found for model {model_name}")
            exit(1)
            
    except Exception as e:
        logger.error(f"Model promotion failed: {e}")
        exit(1)

if __name__ == "__main__":
    import pandas as pd
    main()