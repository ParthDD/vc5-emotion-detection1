import os
import re
import numpy as np
import pandas as pd
import nltk
import string
import logging
from typing import Any
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

# Set up logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    filename='logs/data_preprocessing.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# Download required NLTK resources
try:
    nltk.download('wordnet')
    nltk.download('stopwords')
    logging.info("Downloaded NLTK resources successfully.")
except Exception as e:
    logging.error(f"Error downloading NLTK resources: {e}")

def lemmatization(text: str) -> str:
    """Lemmatize each word in the text."""
    try:
        lemmatizer = WordNetLemmatizer()
        text = text.split()
        text = [lemmatizer.lemmatize(y) for y in text]
        return " ".join(text)
    except Exception as e:
        logging.error(f"Error in lemmatization: {e}")
        return text

def remove_stop_words(text: str) -> str:
    """Remove stop words from the text."""
    try:
        stop_words = set(stopwords.words("english"))
        Text = [i for i in str(text).split() if i not in stop_words]
        return " ".join(Text)
    except Exception as e:
        logging.error(f"Error removing stop words: {e}")
        return text

def removing_numbers(text: str) -> str:
    """Remove all digits from the text."""
    try:
        text = ''.join([i for i in text if not i.isdigit()])
        return text
    except Exception as e:
        logging.error(f"Error removing numbers: {e}")
        return text

def lower_case(text: str) -> str:
    """Convert all words in the text to lowercase."""
    try:
        text = text.split()
        text = [y.lower() for y in text]
        return " ".join(text)
    except Exception as e:
        logging.error(f"Error converting to lowercase: {e}")
        return text

def removing_punctuations(text: str) -> str:
    """Remove punctuations and extra whitespace from the text."""
    try:
        text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
        text = text.replace('؛', "", )
        text = re.sub('\s+', ' ', text)
        text = " ".join(text.split())
        return text.strip()
    except Exception as e:
        logging.error(f"Error removing punctuations: {e}")
        return text

def removing_urls(text: str) -> str:
    """Remove URLs from the text."""
    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)
    except Exception as e:
        logging.error(f"Error removing URLs: {e}")
        return text

def remove_small_sentences(df: pd.DataFrame) -> pd.DataFrame:
    """Set text to NaN if sentence has fewer than 3 words."""
    try:
        for i in range(len(df)):
            if len(str(df.text.iloc[i]).split()) < 3:
                df.text.iloc[i] = np.nan
        return df
    except Exception as e:
        logging.error(f"Error removing small sentences: {e}")
        return df

def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all preprocessing steps to the 'content' column of the DataFrame."""
    try:
        df.content = df.content.apply(lambda content: lower_case(content))
        df.content = df.content.apply(lambda content: remove_stop_words(content))
        df.content = df.content.apply(lambda content: removing_numbers(content))
        df.content = df.content.apply(lambda content: removing_punctuations(content))
        df.content = df.content.apply(lambda content: removing_urls(content))
        df.content = df.content.apply(lambda content: lemmatization(content))
        return df
    except Exception as e:
        logging.error(f"Error normalizing text: {e}")
        return df

def normalized_sentence(sentence: str) -> str:
    """Apply all preprocessing steps to a single sentence."""
    try:
        sentence = lower_case(sentence)
        sentence = remove_stop_words(sentence)
        sentence = removing_numbers(sentence)
        sentence = removing_punctuations(sentence)
        sentence = removing_urls(sentence)
        sentence = lemmatization(sentence)
        return sentence
    except Exception as e:
        logging.error(f"Error normalizing sentence: {e}")
        return sentence

def main() -> None:
    """Main function to load, preprocess, and save train/test data."""
    try:
        # Load raw train and test data
        train_data = pd.read_csv("data/raw/train.csv")
        test_data = pd.read_csv("data/raw/test.csv")
        logging.info("Loaded raw train and test data.")

        # Normalize train and test data
        train_data = normalize_text(train_data)
        test_data = normalize_text(test_data)
        logging.info("Normalized train and test data.")

        # Save processed data to CSV files
        os.makedirs("data/processed", exist_ok=True)  # Ensure the directory exists
        train_data.to_csv("data/processed/train.csv", index=False)
        test_data.to_csv("data/processed/test.csv", index=False)
        logging.info("Saved processed train and test data to CSV files.")
    except Exception as e:
        logging.critical(f"Data preprocessing failed: {e}")
        raise

if __name__ == "__main__":
    main()