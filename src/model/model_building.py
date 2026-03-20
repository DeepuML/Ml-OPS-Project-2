"""
Model Building Module
======================
Trains a Logistic Regression classifier on the Bag-of-Words feature matrix
produced by the feature engineering stage. The trained model is serialized to
models/model.pkl for use in evaluation and inference.

Model configuration:
    - Algorithm: Logistic Regression (scikit-learn)
    - Regularization: L2 (Ridge)  |  C=1 (inverse regularization strength)
    - Solver: liblinear (efficient for small/medium datasets)

Pipeline stage: model_building (fourth stage in the DVC pipeline)
Input:  data/processed/train_bow.csv
Output: models/model.pkl
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
import yaml
import logging

# Configure module-level logger with both console (DEBUG) and file (ERROR) handlers
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_building_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


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


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Train a Logistic Regression classifier on the provided feature matrix.

    Uses L2 regularization with the liblinear solver, which is well-suited for
    high-dimensional sparse feature matrices such as Bag-of-Words representations.

    Args:
        X_train: 2D numpy array of shape (n_samples, n_features) — BoW features.
        y_train: 1D numpy array of binary labels (1 = happiness, 0 = sadness).

    Returns:
        A fitted LogisticRegression model instance.

    Raises:
        Exception: If model fitting fails due to convergence or data issues.
    """
    try:
        clf = LogisticRegression(C=1, solver='liblinear', penalty='l2')
        clf.fit(X_train, y_train)
        logger.debug('Model training completed')
        return clf
    except Exception as e:
        logger.error('Error during model training: %s', e)
        raise


def save_model(model, file_path: str) -> None:
    """Serialize and save the trained model to a pickle file.

    Args:
        model: A fitted scikit-learn model object.
        file_path: Destination path for the serialized model file (e.g., 'models/model.pkl').

    Raises:
        Exception: If the file cannot be written to the specified path.
    """
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise


def main():
    """Execute the model building pipeline stage.

    Loads the training BoW feature matrix, trains a Logistic Regression
    classifier, and saves the fitted model to models/model.pkl.
    """
    try:
        train_data = load_data('./data/processed/train_bow.csv')
        # All columns except the last are features; the last column is the label
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        clf = train_model(X_train, y_train)

        save_model(clf, 'models/model.pkl')
    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()