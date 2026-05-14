"""
dl/lstm_model.py — LSTM-based expense forecasting (TensorFlow-free fallback included)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from utils.helpers import get_model_dir

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False


LOOKBACK = 14  # days of history to use for each prediction


def _build_sequences(values: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(lookback, len(values)):
        X.append(values[i - lookback:i])
        y.append(values[i])
    return np.array(X), np.array(y)


def _simple_moving_avg_forecast(series_values: np.ndarray, days_ahead: int = 30) -> list:
    """Fallback when TensorFlow is not available."""
    window = min(14, len(series_values))
    recent_avg = np.mean(series_values[-window:])
    noise = np.std(series_values[-window:]) * 0.15
    rng = np.random.default_rng(42)
    return [max(0, recent_avg + rng.normal(0, noise)) for _ in range(days_ahead)]


def train_lstm(df: pd.DataFrame) -> dict:
    """
    Train LSTM model on daily expense totals.
    Returns metrics + future predictions.
    """
    # Build daily time-series
    series = df.groupby("Date")["Amount"].sum().sort_index()
    idx = pd.date_range(series.index.min(), series.index.max(), freq="D")
    series = series.reindex(idx, fill_value=0)
    values = series.values.astype(float)

    if len(values) < LOOKBACK + 5:
        return {
            "error": f"Need at least {LOOKBACK + 5} days of data for LSTM.",
            "available": False,
        }

    # Normalise
    vmax = values.max() if values.max() > 0 else 1
    norm = values / vmax

    X, y = _build_sequences(norm, LOOKBACK)

    if not TF_AVAILABLE or len(X) < 10:
        # Fallback to moving-average forecast
        future_raw = _simple_moving_avg_forecast(values, 30)
        last_date = series.index[-1]
        future_dates = [last_date + pd.Timedelta(days=i + 1) for i in range(30)]
        return {
            "model": "Moving Average (LSTM fallback)",
            "available": True,
            "tf_available": False,
            "future_dates": [d.strftime("%Y-%m-%d") for d in future_dates],
            "future_preds": [round(v, 2) for v in future_raw],
            "historical_dates": [d.strftime("%Y-%m-%d") for d in series.index],
            "historical_values": values.tolist(),
            "mae": None,
            "rmse": None,
            "next_week_total": round(sum(future_raw[:7]), 2),
            "next_month_total": round(sum(future_raw[:30]), 2),
        }

    # Reshape for LSTM: [samples, timesteps, features]
    X = X.reshape(X.shape[0], X.shape[1], 1)

    split = max(1, int(len(X) * 0.8))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build model
    tf.random.set_seed(42)
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(LOOKBACK, 1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")

    es = EarlyStopping(patience=5, restore_best_weights=True, verbose=0)
    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=8,
        validation_split=0.1,
        callbacks=[es],
        verbose=0,
    )

    # Metrics
    if len(X_test) > 0:
        preds_norm = model.predict(X_test, verbose=0).flatten()
        preds = preds_norm * vmax
        actual = y_test * vmax
        mae = round(np.mean(np.abs(actual - preds)), 2)
        rmse = round(np.sqrt(np.mean((actual - preds) ** 2)), 2)
    else:
        mae, rmse = None, None

    # Future forecast
    current_seq = norm[-LOOKBACK:].tolist()
    future_preds_norm = []
    for _ in range(30):
        inp = np.array(current_seq[-LOOKBACK:]).reshape(1, LOOKBACK, 1)
        pred_n = float(model.predict(inp, verbose=0)[0][0])
        future_preds_norm.append(max(0, pred_n))
        current_seq.append(pred_n)

    future_raw = [round(v * vmax, 2) for v in future_preds_norm]
    last_date = series.index[-1]
    future_dates = [last_date + pd.Timedelta(days=i + 1) for i in range(30)]

    # Save model
    model_dir = get_model_dir()
    try:
        model.save(str(model_dir / "lstm_expense_model.h5"))
    except Exception:
        pass

    return {
        "model": "LSTM (TensorFlow)",
        "available": True,
        "tf_available": True,
        "mae": mae,
        "rmse": rmse,
        "future_dates": [d.strftime("%Y-%m-%d") for d in future_dates],
        "future_preds": future_raw,
        "historical_dates": [d.strftime("%Y-%m-%d") for d in series.index],
        "historical_values": values.tolist(),
        "next_week_total": round(sum(future_raw[:7]), 2),
        "next_month_total": round(sum(future_raw[:30]), 2),
    }
