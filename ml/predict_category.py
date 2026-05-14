"""
ml/predict_category.py — Predict expense categories using trained model
"""
import pandas as pd
import joblib
import scipy.sparse as sp
from pathlib import Path
from utils.helpers import get_model_dir, keyword_classify


def load_model():
    """Load saved classifier bundle."""
    path = get_model_dir() / "category_model.pkl"
    if path.exists():
        return joblib.load(path)
    return None


def predict_categories(descriptions: pd.Series) -> pd.Series:
    """
    Predict categories for a series of descriptions.
    Falls back to keyword matching if no trained model exists.
    """
    bundle = load_model()
    if bundle is None:
        return descriptions.apply(keyword_classify)

    model = bundle["model"]
    vectorizer = bundle["vectorizer"]
    encoder = bundle["encoder"]

    X, _ = __import__("ml.preprocess", fromlist=["build_tfidf_features"]).build_tfidf_features(
        descriptions, vectorizer=vectorizer, fit=False
    )

    try:
        if hasattr(model, "predict"):
            if "LightGBM" in str(type(model)):
                X_arr = X.toarray() if sp.issparse(X) else X
                preds = model.predict(X_arr)
            else:
                preds = model.predict(X)
            return pd.Series(encoder.inverse_transform(preds), index=descriptions.index)
    except Exception:
        pass

    return descriptions.apply(keyword_classify)
