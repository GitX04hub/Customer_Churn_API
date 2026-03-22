"""
utils.py - Helper functions for the churn prediction API.
"""
import pickle
import pandas as pd
import os

MODEL_PATH       = os.path.join(os.path.dirname(__file__), "model.pkl")
TRANSFORMER_PATH = os.path.join(os.path.dirname(__file__), "transformer.pkl")


def load_artifacts():
    """Load model and transformer from disk."""
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(TRANSFORMER_PATH, "rb") as f:
        artifact = pickle.load(f)
    return model, artifact


def preprocess(customer_dict: dict, artifact: dict) -> any:
    """
    Convert a raw customer dict into a transformed numpy array
    ready for model inference.

    Parameters
    ----------
    customer_dict : dict  – raw JSON fields from the /predict request
    artifact      : dict  – contains 'preprocessor', 'categorical_cols',
                            'numerical_cols'

    Returns
    -------
    numpy.ndarray of shape (1, n_features)
    """
    preprocessor     = artifact["preprocessor"]
    categorical_cols = artifact["categorical_cols"]
    numerical_cols   = artifact["numerical_cols"]

    # Build a single-row DataFrame with the same column order used at training
    all_cols = numerical_cols + categorical_cols
    row = {col: customer_dict.get(col) for col in all_cols}
    df  = pd.DataFrame([row])

    # Coerce TotalCharges to numeric in case it comes in as a string
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    return preprocessor.transform(df)
