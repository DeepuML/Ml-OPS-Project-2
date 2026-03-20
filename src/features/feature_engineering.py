"""
Feature Engineering Module
===========================
Transforms the normalized text data from the preprocessing stage into numerical
feature vectors using scikit-learn's CountVectorizer (Bag-of-Words model).

The fitted vectorizer is saved to models/vectorizer.pkl so it can be reused
during model evaluation and inference without data leakage.

Pipeline stage: feature_engineering (third stage in the DVC pipeline)
Input:  data/interim/train_processed.csv, data/interim/test_processed.csv
Output: data/processed/train_bow.csv, data/processed/test_bow.csv,
        models/vectorizer.pkl
"""

import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
import yaml
import logging
import pickle

# Configure module-level logger with both console (DEBUG) and file (ERROR) handlers
logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('feature_engineering_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """Load pipeline hyperparameters from a YAML configuration file.

    Args:
        params_path: Path to the YAML file (e.g., 'params.yaml').

    Returns:
        A dictionary containing all pipeline parameters.

    Raises:
        FileNotFoundError: If the params file does not exist at the given path.
        yaml.YAMLError: If the file contains invalid YAML syntax.
    """
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load a preprocessed CSV file and fill any NaN values with empty strings.

    Args:
        file_path: Path to the CSV file to load.

    Returns:
        A pandas DataFrame with NaN values replaced by empty strings.

    Raises:
        pd.errors.ParserError: If the CSV file cannot be parsed correctly.
    """
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def apply_bow(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
    """Vectorize text data using a Bag-of-Words (CountVectorizer) model.

    Fits a CountVectorizer on the training set only (to prevent data leakage),
    then transforms both train and test sets. The fitted vectorizer is serialized
    to models/vectorizer.pkl for reuse during inference.

    Args:
        train_data: Training DataFrame with 'content' and 'sentiment' columns.
        test_data: Test DataFrame with 'content' and 'sentiment' columns.
        max_features: Maximum vocabulary size for the CountVectorizer.

    Returns:
        A tuple (train_df, test_df) where each DataFrame contains BoW feature
        columns plus a 'label' column with the encoded sentiment.

    Raises:
        Exception: If vectorization or serialization fails.
    """
    try:
        vectorizer = CountVectorizer(max_features=max_features)

        X_train = train_data['content'].values
        y_train = train_data['sentiment'].values
        X_test = test_data['content'].values
        y_test = test_data['sentiment'].values

        # Fit on train only, then transform both sets to avoid data leakage
        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)

        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test

        # Persist the fitted vectorizer for use during evaluation and inference
        pickle.dump(vectorizer, open('models/vectorizer.pkl', 'wb'))

        logger.debug('Bag of Words applied and data transformed')
        return train_df, test_df
    except Exception as e:
        logger.error('Error during Bag of Words transformation: %s', e)
        raise


def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save a DataFrame to a CSV file, creating parent directories as needed.

    Args:
        df: DataFrame to persist.
        file_path: Destination file path (directories will be created if missing).

    Raises:
        Exception: If the file cannot be written.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug('Data saved to %s', file_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise


def main():
    """Execute the feature engineering pipeline stage.

    Reads params.yaml for max_features, loads normalized interim data, applies
    Bag-of-Words vectorization, and saves the resulting feature matrices to
    data/processed/.
    """
    try:
        params = load_params('params.yaml')
        max_features = params['feature_engineering']['max_features']

        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')

        train_df, test_df = apply_bow(train_data, test_data, max_features)

        save_data(train_df, os.path.join("./data", "processed", "train_bow.csv"))
        save_data(test_df, os.path.join("./data", "processed", "test_bow.csv"))
    except Exception as e:
        logger.error('Failed to complete the feature engineering process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()