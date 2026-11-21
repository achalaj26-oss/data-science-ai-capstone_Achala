
import os

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS_PATH = os.path.join(PROJECT_PATH, "models")
DATA_PATH = os.path.join(PROJECT_PATH, "data")

MODEL_FILE = os.path.join(MODELS_PATH, "xgboost_model.json")
IMPUTER_FILE = os.path.join(MODELS_PATH, "median_imputer.pkl")
VALIDATION_FILE = os.path.join(DATA_PATH, "validation_modeling_data_15k.csv")

# Model evaluation metrics
MODEL_METRICS = {
    "Accuracy": 0.8872,
    "Precision": 0.7812,
    "Recall": 0.7840,
    "F1 Score": 0.7826,
    "ROC-AUC": 0.9483
}
