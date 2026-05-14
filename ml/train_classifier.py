"""
ml/train_classifier.py — Train and evaluate expense category classifiers
"""
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
import warnings

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

from ml.preprocess import build_tfidf_features, encode_labels
from utils.helpers import get_model_dir, CATEGORIES, keyword_classify


def train_classifier(df: pd.DataFrame):
    """
    Train multiple classifiers on expense descriptions.
    Returns best model, vectorizer, encoder, and metrics dict.
    """
    model_dir = get_model_dir()

    # Use Category column if available, else use keyword fallback
    if "Category" not in df.columns or df["Category"].isnull().all():
        df = df.copy()
        df["Category"] = df["Description"].apply(keyword_classify)
    else:
        df = df.copy()
        df["Category"] = df["Category"].fillna(
            df["Description"].apply(keyword_classify)
        )

    # Filter to known categories only
    df = df[df["Category"].isin(CATEGORIES)].copy()

    if len(df) < 10:
        return None, None, None, {"error": "Not enough labelled data to train."}

    X_tfidf, vectorizer = build_tfidf_features(df["Description"], fit=True)
    y, encoder = encode_labels(df["Category"], fit=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
    )

    models = {
        "Naive Bayes": MultinomialNB(alpha=0.5),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=5, random_state=42),
        "Linear SVM": LinearSVC(C=1.0, max_iter=2000, random_state=42),
    }
    if LGBM_AVAILABLE:
        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=100, random_state=42, verbosity=-1
        )

    results = {}
    best_name, best_score, best_model = None, -1, None

    for name, clf in models.items():
        try:
            if name == "LightGBM":
                import scipy.sparse as sp
                X_tr_dense = X_train.toarray() if sp.issparse(X_train) else X_train
                X_te_dense = X_test.toarray() if sp.issparse(X_test) else X_test
                clf.fit(X_tr_dense, y_train)
                preds = clf.predict(X_te_dense)
            else:
                clf.fit(X_train, y_train)
                preds = clf.predict(X_test)

            acc = accuracy_score(y_test, preds)
            present_classes = sorted(set(y_test.tolist()) | set(preds.tolist()))
            present_names = [encoder.classes_[i] for i in present_classes]
            results[name] = {
                "accuracy": round(acc * 100, 2),
                "report": classification_report(
                    y_test, preds,
                    labels=present_classes,
                    target_names=present_names,
                    output_dict=True,
                    zero_division=0
                ),
                "confusion_matrix": confusion_matrix(y_test, preds, labels=present_classes).tolist(),
                "confusion_classes": present_names,
                "model": clf,
            }
            if acc > best_score:
                best_score = acc
                best_name = name
                best_model = clf
        except Exception as e:
            results[name] = {"accuracy": 0, "error": str(e)}

    # Persist best model + preprocessing objects
    joblib.dump(
        {"model": best_model, "vectorizer": vectorizer, "encoder": encoder,
         "best_name": best_name},
        model_dir / "category_model.pkl"
    )

    metrics = {
        "best_model": best_name,
        "best_accuracy": round(best_score * 100, 2),
        "all_results": {k: {kk: vv for kk, vv in v.items() if kk != "model"}
                        for k, v in results.items()},
        "classes": encoder.classes_.tolist(),
    }
    return best_model, vectorizer, encoder, metrics
