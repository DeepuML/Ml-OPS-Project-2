"""
Data Preprocessing Module
==========================
Applies NLP text normalization to the raw tweet data produced by the data
ingestion stage. The cleaned data is saved to data/interim/ for downstream
feature engineering.

Normalization pipeline (applied to the 'content' column):
    1. Lowercase conversion
    2. Stop-word removal (NLTK English corpus)
    3. Number removal
    4. Punctuation removal
    5. URL removal
    6. Lemmatization (NLTK WordNetLemmatizer)

Pipeline stage: data_preprocessing (second stage in the DVC pipeline)
Input:  data/raw/train.csv, data/raw/test.csv
Output: data/interim/train_processed.csv, data/interim/test_processed.csv
"""

import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging

# Configure module-level logger with both console (DEBUG) and file (ERROR) handlers
logger = logging.getLogger('data_transformation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('transformation_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Download required NLTK corpora if not already present
nltk.download('wordnet')
nltk.download('stopwords')


def lemmatization(text: str) -> str:
    """Reduce each word in the text to its base (lemma) form.

    Uses NLTK's WordNetLemmatizer to normalize inflected words (e.g.,
    'running' → 'run', 'better' → 'good').

    Args:
        text: A whitespace-separated string of words.

    Returns:
        The lemmatized text as a single space-joined string.
    """
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)


def remove_stop_words(text: str) -> str:
    """Remove common English stop-words that carry little semantic meaning.

    Args:
        text: Input text string.

    Returns:
        Text with stop-words removed, as a space-joined string.
    """
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)


def removing_numbers(text: str) -> str:
    """Strip all numeric digit characters from the text.

    Args:
        text: Input text string.

    Returns:
        Text with all digit characters removed.
    """
    text = ''.join([char for char in text if not char.isdigit()])
    return text


def lower_case(text: str) -> str:
    """Convert all words in the text to lowercase.

    Args:
        text: Input text string.

    Returns:
        Lowercased text as a space-joined string.
    """
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)


def removing_punctuations(text: str) -> str:
    """Remove standard punctuation characters and normalize whitespace.

    Also removes the Arabic semicolon character '؛' and collapses multiple
    consecutive spaces into a single space.

    Args:
        text: Input text string.

    Returns:
        Text with punctuation removed and whitespace normalized.
    """
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub('\s+', ' ', text).strip()
    return text


def removing_urls(text: str) -> str:
    """Remove HTTP/HTTPS URLs and bare www. addresses from the text.

    Args:
        text: Input text string.

    Returns:
        Text with all URL patterns removed.
    """
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)


def remove_small_sentences(df: pd.DataFrame) -> None:
    """Replace tweet content shorter than 3 words with NaN (in-place).

    Very short texts provide insufficient signal for sentiment classification
    and are nullified so they can be dropped downstream.

    Args:
        df: DataFrame with a 'text' column to be filtered in-place.
    """
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan


def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full text normalization pipeline to the 'content' column.

    Executes the six normalization steps in sequence: lowercase → stop-word
    removal → number removal → punctuation removal → URL removal → lemmatization.

    Args:
        df: DataFrame containing a 'content' column with raw tweet text.

    Returns:
        The same DataFrame with 'content' replaced by normalized text.

    Raises:
        Exception: If any transformation step fails unexpectedly.
    """
    try:
        df['content'] = df['content'].apply(lower_case)
        logger.debug('converted to lower case')
        df['content'] = df['content'].apply(remove_stop_words)
        logger.debug('stop words removed')
        df['content'] = df['content'].apply(removing_numbers)
        logger.debug('numbers removed')
        df['content'] = df['content'].apply(removing_punctuations)
        logger.debug('punctuations removed')
        df['content'] = df['content'].apply(removing_urls)
        logger.debug('URLs removed')
        df['content'] = df['content'].apply(lemmatization)
        logger.debug('lemmatization performed')
        logger.debug('Text normalization completed')
        return df
    except Exception as e:
        logger.error('Error during text normalization: %s', e)
        raise


def main():
    """Execute the data preprocessing pipeline stage.

    Reads train/test CSVs from data/raw/, applies text normalization, and
    saves the processed results to data/interim/.
    """
    try:
        # Load raw train and test splits from the data ingestion stage
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('data loaded properly')

        # Apply the full NLP normalization pipeline to both splits
        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)

        # Persist the normalized data to data/interim for feature engineering
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)

        logger.debug('Processed data saved to %s', data_path)
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()