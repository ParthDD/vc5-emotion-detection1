import numpy as np
import pandas as pd
import os
import logging
from sklearn.model_selection import train_test_split
import yaml
from typing import Tuple, Dict

# Set up logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    filename='logs/data_ingestion.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

def load_params(params_path: str = 'params.yaml') -> Dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.info("Loaded parameters from params.yaml successfully.")
        return params
    except Exception as e:
        logging.error(f"Error loading params.yaml: {e}")
        raise

def load_dataset(url: str) -> pd.DataFrame:
    """Load dataset from a remote CSV file."""
    try:
        df = pd.read_csv(url)
        logging.info("Dataset loaded successfully from remote CSV.")
        return df
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise

def drop_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Drop a column from the DataFrame if it exists."""
    try:
        df.drop(columns=[column], inplace=True)
        logging.info(f"'{column}' column dropped.")
    except KeyError:
        logging.warning(f"'{column}' column not found in the dataset.")
    except Exception as e:
        logging.error(f"Error dropping '{column}' column: {e}")
        raise
    return df

def filter_and_encode(df: pd.DataFrame) -> pd.DataFrame:
    """Filter for 'happiness' and 'sadness', and encode sentiment as binary."""
    try:
        filtered_df = df[df['sentiment'].isin(['happiness', 'sadness'])].copy()
        logging.info("Filtered dataset for 'happiness' and 'sadness' sentiments.")
        filtered_df['sentiment'].replace({'happiness': 1, 'sadness': 0}, inplace=True)
        logging.info("Converted sentiment labels to binary.")
        return filtered_df
    except Exception as e:
        logging.error(f"Error processing sentiment labels: {e}")
        raise

def split_data(df: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the data into training and testing sets."""
    try:
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
        logging.info(f"Data split into train and test sets with test_size={test_size}.")
        return train_data, test_data
    except Exception as e:
        logging.error(f"Error splitting data: {e}")
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, train_path: str, test_path: str) -> None:
    """Save the train and test datasets as CSV files."""
    try:
        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        logging.info("Train and test datasets saved successfully.")
    except Exception as e:
        logging.error(f"Error saving train/test datasets: {e}")
        raise

def ingest_data() -> None:
    try:
        params = load_params('params.yaml')
        test_size = params['data_ingestion']['test_size']
        df = load_dataset('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
        df = drop_column(df, 'tweet_id')
        final_df = filter_and_encode(df)
        train_data, test_data = split_data(final_df, test_size)
        save_data(train_data, test_data, 'data/raw/train.csv', 'data/raw/test.csv')
        logging.info("Data ingestion pipeline completed successfully.")
    except Exception as e:
        logging.critical(f"Data ingestion failed: {e}")
        raise

if __name__ == "__main__":
    ingest_data()