"""
Model Evaluation Module
========================
Evaluates the trained Logistic Regression classifier against the held-out test
set and logs all metrics, parameters, and the model artifact to MLflow on
DagsHub. Evaluation results are also saved locally for downstream pipeline
stages.

Metrics logged:
    - accuracy   — fraction of correctly classified samples
    - precision  — true positives / (true positives + false positives)
    - recall     — true positives / (true positives + false negatives)
    - auc        — area under the ROC curve (probability scores used)

Pipeline stage: model_evaluation (fifth stage in the DVC pipeline)
Input:  models/model.pkl, models/vectorizer.pkl, data/processed/test_bow.csv
Output: reports/metrics.json, reports/experiment_info.json
        (MLflow: metrics, params, model artifact, artifact files)
"""

import os
import mlflow.models.signature
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import mlflow
import mlflow.sklearn
import dagshub

from dotenv import load_dotenv
import os

# Load DAGSHUB_PAT and MLflow credentials from .env / environment
load_dotenv()

dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

# Configure MLflow tracking to use DagsHub as the remote tracking server
os.environ["MLFLOW_TRACKING_USERNAME"] = "DeepuML"
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token


dagshub_url = "https://dagshub.com"
repo_owner = "DeepuML"
repo_name = "Ml-OPS-Project-2"

mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

# Configure module-level logger with both console (DEBUG) and file (ERROR) handlers
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_model(file_path: str):
    """Deserialize a pickle model or vectorizer from disk.

    Args:
        file_path: Path to the .pkl file to load.

    Returns:
        The deserialized Python object (e.g., a fitted classifier or vectorizer).

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
    """
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model: %s', e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load a feature matrix CSV produced by the feature engineering stage.

    Args:
        file_path: Path to the CSV file containing BoW features and a label column.

    Returns:
        A pandas DataFrame with feature columns and a 'label' column.

    Raises:
        pd.errors.ParserError: If the CSV file cannot be parsed correctly.
    """
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Compute classification metrics on the held-out test set.

    Args:
        clf: A fitted scikit-learn classifier with `predict` and `predict_proba` methods.
        X_test: 2D numpy array of test features of shape (n_samples, n_features).
        y_test: 1D numpy array of true binary labels.

    Returns:
        A dictionary with keys 'accuracy', 'precision', 'recall', and 'auc'.

    Raises:
        Exception: If prediction or metric computation fails.
    """
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        logger.debug('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise


def save_metrics(metrics: dict, file_path: str) -> None:
    """Persist evaluation metrics to a JSON file for downstream pipeline stages.

    Args:
        metrics: Dictionary of metric name → float value.
        file_path: Destination path for the JSON output (e.g., 'reports/metrics.json').

    Raises:
        Exception: If the file cannot be written.
    """
    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug('Metrics saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the metrics: %s', e)
        raise


def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the MLflow run ID and registered model path to a JSON file.

    This file is consumed by the model_registration stage to locate the correct
    MLflow run when registering the model.

    Args:
        run_id: The MLflow run ID string for the evaluation run.
        model_path: The artifact path under which the model was logged (e.g., 'model').
        file_path: Destination path for the JSON output (e.g., 'reports/experiment_info.json').

    Raises:
        Exception: If the file cannot be written.
    """
    try:
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug('Model info saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model info: %s', e)
        raise


def main():
    """Execute the model evaluation pipeline stage within an MLflow run.

    Loads the test feature matrix, evaluates the trained classifier, logs all
    metrics and parameters to MLflow, saves the model artifact to the registry,
    and persists the experiment metadata for the registration stage.
    """
    mlflow.set_experiment("dvc-pipeline")
    with mlflow.start_run() as run:
        try:
            vectorizer = load_model('./models/vectorizer.pkl')
            clf = load_model('./models/model.pkl')
            test_data = load_data('./data/processed/test_bow.csv')

            # The test_bow.csv is already vectorized; extract feature matrix and labels
            X_test = np.array(test_data.iloc[:, :-1].values)
            y_test = np.array(test_data.iloc[:, -1].values)

            metrics = evaluate_model(clf, X_test, y_test)

            save_metrics(metrics, 'reports/metrics.json')

            # Log all evaluation metrics to the active MLflow run
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Log all model hyperparameters to the active MLflow run
            if hasattr(clf, 'get_params'):
                params = clf.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)

            # Log the sklearn model artifact with inferred signature and an input example
            input_example = pd.DataFrame(X_test, columns=test_data.columns[:-1]).iloc[:5]
            signature = mlflow.models.signature.infer_signature(input_example, clf.predict(X_test[:5]))
            mlflow.sklearn.log_model(clf, "model", signature=signature, input_example=input_example)

            # Save run metadata for the model registration stage
            save_model_info(run.info.run_id, "model", 'reports/experiment_info.json')

            # Upload evaluation artifacts to MLflow artifact store
            mlflow.log_artifact('reports/metrics.json')
            mlflow.log_artifact('reports/experiment_info.json')
            mlflow.log_artifact('model_evaluation_errors.log')
        except Exception as e:
            logger.error('Failed to complete the model evaluation process: %s', e)
            print(f"Error: {e}")


if __name__ == '__main__':
    main()