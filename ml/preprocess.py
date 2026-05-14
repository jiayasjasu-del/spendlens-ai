"""
ml/preprocess.py — Feature engineering for ML models
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import joblib
from utils.helpers import clean_text, get_model_dir, CATEGORIES


def build_tfidf_features(descriptions: pd.Series, vectorizer=None, fit=True):
    """Build TF-IDF feature matrix from description strings."""
    cleaned = descriptions.apply(clean_text)
    if fit or vectorizer is None:
        vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
        )
        X = vectorizer.fit_transform(cleaned)
    else:
        X = vectorizer.transform(cleaned)
    return X, vectorizer


def encode_labels(categories: pd.Series, encoder=None, fit=True):
    """Encode string categories to integers."""
    if fit or encoder is None:
        encoder = LabelEncoder()
        encoder.fit(CATEGORIES)
    y = encoder.transform(categories)
    return y, encoder


def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append numerical amount-derived features."""
    df = df.copy()
    df["LogAmount"] = np.log1p(df["Amount"])
    df["AmountBin"] = pd.cut(
        df["Amount"],
        bins=[0, 200, 500, 1000, 3000, 10000, np.inf],
        labels=[0, 1, 2, 3, 4, 5]
    ).astype(int)
    return df
