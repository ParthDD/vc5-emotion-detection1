import pandas as pd
import pickle
import json
import logging
import os
from typing import Dict, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Set up logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    filename='logs/model_evaluation.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

def load_model(model_path: str) -> Any:
    """Load a trained model from disk."""
    try:
        model = pickle.load(open(model_path, "rb"))
        logging.info(f"Loaded model from {model_path}.")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

def load_test_data(test_path: str) -> pd.DataFrame:
    """Load test data from CSV file."""
    try:
        test_data = pd.read_csv(test_path)
        logging.info(f"Loaded test data from {test_path}.")
        return test_data
    except Exception as e:
        logging.error(f"Error loading test data: {e}")
        raise

def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Evaluate the model and return metrics."""
    try:
        y_pred = model.predict(X_test)
        metrics_dict = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred)
        }
        logging.info("Model evaluation completed.")
        return metrics_dict
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        raise

def save_metrics(metrics: Dict[str, float], metrics_path: str) -> None:
    """Save evaluation metrics to a JSON file."""
    try:
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        logging.info(f"Saved metrics to {metrics_path}.")
    except Exception as e:
        logging.error(f"Error saving metrics: {e}")
        raise

def main() -> None:
    """Main function to run model evaluation pipeline."""
    try:
        model = load_model("models/random_forest_model.pkl")
        test_data = load_test_data("data/interim/test_tfidf.csv")
        X_test = test_data.drop(columns=['label']).values
        y_test = test_data['label'].values
        metrics = evaluate_model(model, X_test, y_test)
        save_metrics(metrics, "reports/metrics.json")
        logging.info("Model evaluation pipeline completed successfully.")
    except Exception as e:
        logging.critical(f"Model evaluation pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()