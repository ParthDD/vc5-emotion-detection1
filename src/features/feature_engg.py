import pandas as pd
import numpy as np
import os
import yaml
import logging
from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer

# Set up logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    filename='logs/feature_engg.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

def load_params(params_path: str = 'params.yaml') -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.info("Loaded parameters from params.yaml successfully.")
        return params
    except Exception as e:
        logging.error(f"Error loading params.yaml: {e}")
        raise

def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed train and test data, dropping rows with missing content."""
    try:
        train_data = pd.read_csv(train_path).dropna(subset=['content'])
        test_data = pd.read_csv(test_path).dropna(subset=['content'])
        logging.info("Loaded and cleaned train and test data.")
        return train_data, test_data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def extract_features_and_labels(
    train_data: pd.DataFrame, test_data: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract features and labels from train and test data."""
    try:
        X_train = train_data['content'].values
        y_train = train_data['sentiment'].values
        X_test = test_data['content'].values
        y_test = test_data['sentiment'].values
        logging.info("Extracted features and labels from data.")
        return X_train, y_train, X_test, y_test
    except Exception as e:
        logging.error(f"Error extracting features and labels: {e}")
        raise

def vectorize_text(
    X_train: np.ndarray, X_test: np.ndarray, max_features: int
) -> Tuple[np.ndarray, np.ndarray, TfidfVectorizer]:
    """Apply Bag of Words (Tfidf) and transform train and test data."""
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        logging.info("Applied Tf IDF to train and test data.")
        return X_train_tfidf, X_test_tfidf, vectorizer
    except Exception as e:
        logging.error(f"Error in vectorization: {e}")
        raise

def save_features(
    X_train_tfidf: np.ndarray, y_train: np.ndarray,
    X_test_tfidf: np.ndarray, y_test: np.ndarray
) -> None:
    """Save the processed feature data to CSV files."""
    try:
        os.makedirs("data/interim", exist_ok=True)
        train_df = pd.DataFrame(X_train_tfidf.toarray())
        train_df['label'] = y_train
        test_df = pd.DataFrame(X_test_tfidf.toarray())
        test_df['label'] = y_test
        train_df.to_csv("data/interim/train_tfidf.csv", index=False)
        test_df.to_csv("data/interim/test_tfidf.csv", index=False)
        logging.info("Saved Bag-of-Words features to CSV files.")
    except Exception as e:
        logging.error(f"Error saving features: {e}")
        raise

def main() -> None:
    """Main function to run feature engineering pipeline."""
    try:
        params = load_params('params.yaml')
        max_features = params['feature_engineering']['max_features']
        train_data, test_data = load_data("data/processed/train.csv", "data/processed/test.csv")
        X_train, y_train, X_test, y_test = extract_features_and_labels(train_data, test_data)
        X_train_tfidf, X_test_tfidf, _ = vectorize_text(X_train, X_test, max_features)
        save_features(X_train_tfidf, y_train, X_test_tfidf, y_test)
        logging.info("Feature engineering completed successfully.")
    except Exception as e:
        logging.critical(f"Feature engineering failed: {e}")
        raise

if __name__ == "__main__":
    main()