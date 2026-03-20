"""
Data Ingestion Module
=====================
Downloads the raw tweet emotions dataset from a remote URL, filters it to retain
only 'happiness' and 'sadness' labels, encodes them as binary (1/0), splits the
data into train/test sets, and persists the splits to the data/raw directory.

Pipeline stage: data_ingestion (first stage in the DVC pipeline)
Output: data/raw/train.csv, data/raw/test.csv
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
import logging

# Configure module-level logger with both console (DEBUG) and file (ERROR) handlers
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('errors.log')
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


def load_data(data_url: str) -> pd.DataFrame:
    """Load the raw tweet emotions dataset from a CSV file or remote URL.

    Args:
        data_url: Local file path or remote URL pointing to the CSV dataset.

    Returns:
        A pandas DataFrame containing the raw tweet data.

    Raises:
        pd.errors.ParserError: If the CSV file cannot be parsed correctly.
    """
    try:
        df = pd.read_csv(data_url)
        logger.debug('Data loaded from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter and encode the raw tweet dataset for binary sentiment classification.

    Keeps only rows with 'happiness' or 'sadness' sentiment labels, drops the
    non-informative 'tweet_id' column, and encodes labels as 1 (happiness) / 0
    (sadness).

    Args:
        df: Raw DataFrame loaded from the tweet emotions CSV.

    Returns:
        A filtered and label-encoded DataFrame with columns ['content', 'sentiment'].

    Raises:
        KeyError: If required columns ('tweet_id', 'sentiment') are missing.
    """
    try:
        df.drop(columns=['tweet_id'], inplace=True)
        # Keep only binary sentiment classes relevant to the task
        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])]
        final_df['sentiment'].replace({'happiness': 1, 'sadness': 0}, inplace=True)
        logger.debug('Data preprocessing completed')
        return final_df
    except KeyError as e:
        logger.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Persist the train and test splits to the data/raw directory as CSV files.

    Args:
        train_data: Training split DataFrame.
        test_data: Test split DataFrame.
        data_path: Root data directory path (a 'raw' subdirectory will be created).

    Raises:
        Exception: If the directory cannot be created or files cannot be written.
    """
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logger.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise


def main():
    """Execute the data ingestion pipeline stage.

    Loads params.yaml for test_size, downloads the remote tweet emotions dataset,
    filters to binary classes, stratified-splits into train/test, and saves both
    splits to data/raw/.
    """
    try:
        params = load_params(params_path='params.yaml')
        test_size = params['data_ingestion']['test_size']

        df = load_data(data_url='https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        save_data(train_data, test_data, data_path='./data')
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()