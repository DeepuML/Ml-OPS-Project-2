"""
Model Registration Module
==========================
Registers the trained model artifact from the most recent MLflow evaluation run
into the MLflow Model Registry. The registered version is tagged as 'staging'
and marked ready for promotion review.

Registration tags set on the model version:
    - deployment_status = "ready"
    - environment       = "staging"

Pipeline stage: model_registration (sixth stage in the DVC pipeline)
Input:  reports/experiment_info.json (contains MLflow run_id and model_path)
Output: A new version entry in the MLflow Model Registry under the model name
        'my_model'
"""

# register model

import json
from dotenv import load_dotenv
import mlflow
from mlflow.tracking import MlflowClient
import logging
import os

# Load DAGSHUB_PAT and MLflow credentials from .env / environment
load_dotenv()

dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

dagshub_url = "https://dagshub.com"
repo_owner = "DeepuML"
repo_name = "Ml-OPS-Project-2"

# Authenticate MLflow tracking with DagsHub PAT
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# Point MLflow at the DagsHub-hosted tracking server
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")


# Configure module-level logger with both console (DEBUG) and file (ERROR) handlers
logger = logging.getLogger('model_registration')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_model_info(file_path: str) -> dict:
    """Load the MLflow run metadata saved by the model evaluation stage.

    Args:
        file_path: Path to the JSON file (e.g., 'reports/experiment_info.json').

    Returns:
        A dictionary with 'run_id' and 'model_path' keys.

    Raises:
        FileNotFoundError: If the experiment info file does not exist.
    """
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model info: %s', e)
        raise


def register_model(model_name: str, model_info: dict):
    """Register the MLflow model artifact under the given name in the Model Registry.

    Constructs the model URI from the run_id and model_path stored in model_info,
    registers it, then sets version tags to indicate that this version is in
    the staging environment and ready for promotion review.

    Args:
        model_name: The registered model name in MLflow (e.g., 'my_model').
        model_info: Dictionary with 'run_id' and 'model_path' keys.

    Raises:
        Exception: If model registration or tagging fails.
    """
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"

        # Register the model version in the MLflow Model Registry
        model_version = mlflow.register_model(model_uri, model_name)

        client = MlflowClient()
        client.update_model_version(
            name=model_name,
            version=model_version.version,
            description="Model registered for production deployment"
        )

        # Tag the version as ready for staging review
        client.set_model_version_tag(
            name=model_name,
            version=model_version.version,
            key="deployment_status",
            value="ready"
        )

        # Mark environment as 'staging' — promote_model.py will update this to 'production'
        client.set_model_version_tag(
            name=model_name,
            version=model_version.version,
            key="environment",
            value="staging"
        )

        logger.debug(f'Model {model_name} version {model_version.version} registered with deployment tags.')
    except Exception as e:
        logger.error('Error during model registration: %s', e)
        raise


def main():
    """Execute the model registration pipeline stage.

    Reads experiment metadata from reports/experiment_info.json and registers
    the corresponding MLflow run artifact as a new version of 'my_model'.
    """
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)

        model_name = "my_model"
        register_model(model_name, model_info)
    except Exception as e:
        logger.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()