#!/usr/bin/env python3
"""
Model Promotion Module
=======================
Automatically evaluates the latest registered model version and promotes it
from staging to production in the MLflow Model Registry if it satisfies all
quality thresholds. This script is executed as the final pipeline stage and
also runs as part of the GitHub Actions CI/CD workflow.

Promotion criteria (ALL must pass):
    - accuracy  >= 0.75
    - precision >= 0.75
    - recall    >= 0.70
    - auc       >= 0.75

On promotion, the model version receives:
    - environment      = "production"
    - promotion_status = "approved"
    - deployed_at      = <current UTC timestamp>

Pipeline stage: model_promotion (seventh and final stage in the DVC pipeline)
Input:  MLflow Model Registry (latest version of 'my_model')
Output: Updated model version tags in MLflow Model Registry
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
    """Configure the MLflow tracking URI and authenticate with DagsHub.

    Reads the DAGSHUB_PAT from the environment (or .env file) and sets
    MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD accordingly.

    Raises:
        EnvironmentError: If DAGSHUB_PAT is not set in the environment.
    """
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


def get_model_metrics(run_id: str) -> dict:
    """Retrieve the logged metrics for a specific MLflow run.

    Args:
        run_id: The MLflow run ID string to retrieve metrics for.

    Returns:
        A dictionary mapping metric names to their float values.
        Returns an empty dict if the run cannot be found.
    """
    client = MlflowClient()
    try:
        run = client.get_run(run_id)
        return run.data.metrics
    except Exception as e:
        logger.error(f"Could not get metrics for run {run_id}: {e}")
        return {}


def check_promotion_criteria(metrics: dict) -> tuple:
    """Evaluate whether a model's metrics satisfy the production promotion thresholds.

    Args:
        metrics: Dictionary of metric name → float value from an MLflow run.

    Returns:
        A tuple of (can_promote, passed_criteria, failed_criteria) where:
            - can_promote (bool): True if all criteria are met.
            - passed_criteria (list[str]): Human-readable strings for passing metrics.
            - failed_criteria (list[str]): Human-readable strings for failing metrics.
    """
    # Minimum thresholds required for production deployment
    criteria = {
        'accuracy': 0.75,      # Must be at least 75% accurate
        'precision': 0.75,     # Must be at least 75% precise
        'recall': 0.70,        # Must be at least 70% recall
        'auc': 0.75            # Must be at least 75% AUC
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


def promote_model_to_production(model_name: str, model_version: str, run_id: str) -> bool:
    """Promote a model version to production if it meets all quality criteria.

    Fetches the run metrics, evaluates them against promotion thresholds, and
    if all criteria are met, updates the model version tags to mark it as
    production-ready.

    Args:
        model_name: Registered model name in MLflow (e.g., 'my_model').
        model_version: Version number string to evaluate for promotion.
        run_id: MLflow run ID associated with this model version.

    Returns:
        True if the model was promoted, False otherwise.
    """
    client = MlflowClient()

    # Retrieve metrics from the MLflow run that produced this model version
    metrics = get_model_metrics(run_id)

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
        # Tag the model version as production-ready in the MLflow Model Registry
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
    """Execute the model promotion pipeline stage.

    Connects to the MLflow Model Registry, retrieves the latest registered
    version of 'my_model', evaluates it against production criteria, and
    promotes it if all thresholds are met. Exits with code 0 on promotion,
    code 1 on failure or if criteria are not met.
    """
    try:
        setup_mlflow()

        model_name = "my_model"
        client = MlflowClient()

        # Retrieve all registered versions and select the most recent one
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