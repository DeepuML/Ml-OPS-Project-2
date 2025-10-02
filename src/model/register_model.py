# # register model

# import json
# import mlflow
# import logging
# import os
# import dagshub


# # dagshub_token = "38482a552497f9dec23398027b8b85dd86d07772"

# # dagshub_url = "https://dagshub.com"
# # repo_owner = "DeepuML"
# # repo_name = "Ml-OPS-Project-2"

# # os.environ["MLFLOW_TRACKING_USERNAME"] = repo_owner
# # os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# # # Initialize Dagshub MLflow tracking
# # dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
# os.environ["MLFLOW_TRACKING_USERNAME"] = "DeepuML"
# os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# dagshub.init(
#     repo_owner="DeepuML",
#     repo_name="Ml-OPS-Project-2",
#     mlflow=True,
#     token="38482a552497f9dec23398027b8b85dd86d07772" # ✅ explicit
# )

# mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")


# # logging configuration
# logger = logging.getLogger('model_registration')
# logger.setLevel('DEBUG')

# console_handler = logging.StreamHandler()
# console_handler.setLevel('DEBUG')

# file_handler = logging.FileHandler('model_registration_errors.log')
# file_handler.setLevel('ERROR')

# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# console_handler.setFormatter(formatter)
# file_handler.setFormatter(formatter)

# logger.addHandler(console_handler)
# logger.addHandler(file_handler)

# def load_model_info(file_path: str) -> dict:
#     """Load the model info from a JSON file."""
#     try:
#         with open(file_path, 'r') as file:
#             model_info = json.load(file)
#         logger.debug('Model info loaded from %s', file_path)
#         return model_info
#     except FileNotFoundError:
#         logger.error('File not found: %s', file_path)
#         raise
#     except Exception as e:
#         logger.error('Unexpected error occurred while loading the model info: %s', e)
#         raise

# def register_model(model_name: str, model_info: dict):
#     """Register the model to the MLflow Model Registry."""
#     try:
#         model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        
#         # Register the model
#         model_version = mlflow.register_model(model_uri, model_name)
        
#         # Transition the model to "Staging" stage
#         client = mlflow.tracking.MlflowClient()
#         client.transition_model_version_stage(
#             name=model_name,
#             version=model_version.version,
#             stage="Staging"
#         )
        
#         logger.debug(f'Model {model_name} version {model_version.version} registered and transitioned to Staging.')
#     except Exception as e:
#         logger.error('Error during model registration: %s', e)
#         raise

# def main():
#     try:
#         model_info_path = 'reports/experiment_info.json'
#         model_info = load_model_info(model_info_path)
        
#         model_name = "my_model"
#         register_model(model_name, model_info)
#     except Exception as e:
#         logger.error('Failed to complete the model registration process: %s', e)
#         print(f"Error: {e}")

# if __name__ == '__main__':
#     main()

# register_model.py

import json
import mlflow
import logging
import os
import dagshub

# 🔒 Hardcoded credentials (⚠️ not safe for production!)
repo_owner = "DeepuML"
repo_name = "Ml-OPS-Project-2"
dagshub_token = "38482a552497f9dec23398027b8b85dd86d07772"  # personal token
dagshub_url = "https://dagshub.com"

# set env vars explicitly
os.environ["MLFLOW_TRACKING_USERNAME"] = repo_owner
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# initialize dagshub connection
dagshub.init(
    repo_owner=repo_owner,
    repo_name=repo_name,
    mlflow=True,
    token=dagshub_token
)

# set MLflow tracking URI
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

# logging setup
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
    """Load the model info from a JSON file."""
    with open(file_path, 'r') as file:
        model_info = json.load(file)
    logger.debug('Model info loaded from %s', file_path)
    return model_info


def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"

    # Register the model
    model_version = mlflow.register_model(model_uri, model_name)

    # Transition the model to "Staging"
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Staging"
    )

    logger.debug(f'Model {model_name} version {model_version.version} registered and transitioned to Staging.')


def main():
    model_info_path = 'reports/experiment_info.json'
    model_info = load_model_info(model_info_path)
    
    model_name = "my_model"
    register_model(model_name, model_info)


if __name__ == '__main__':
    main()
