"""
ml/forecasting.py — Expense forecasting with Linear Regression & Random Forest
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


def build_daily_series(df: pd.DataFrame) -> pd.Series:
    """Aggregate expenses to daily totals."""
    series = df.groupby("Date")["Amount"].sum().sort_index()
    # Fill missing days with 0
    idx = pd.date_range(series.index.min(), series.index.max(), freq="D")
    series = series.reindex(idx, fill_value=0)
    return series


def sliding_window_features(series: pd.Series, window: int = 7):
    """Create sliding window (X, y) pairs for regression."""
    values = series.values
    X, y = [], []
    for i in range(window, len(values)):
        X.append(values[i - window:i])
        y.append(values[i])
    return np.array(X), np.array(y)


def forecast_ml(df: pd.DataFrame, days_ahead: int = 30) -> dict:
    """
    Train Linear Regression and Random Forest on daily spend history.
    Returns predictions and metrics.
    """
    series = build_daily_series(df)
    if len(series) < 14:
        return {"error": "Need at least 14 days of data for forecasting."}

    window = min(7, len(series) // 2)
    X, y = sliding_window_features(series, window)

    split = max(1, int(len(X) * 0.8))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test) if len(X_test) > 0 else X_test

    results = {}

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_s, y_train)
    if len(X_test) > 0:
        lr_preds = lr.predict(X_test_s)
        results["Linear Regression"] = {
            "mae": round(mean_absolute_error(y_test, lr_preds), 2),
            "rmse": round(np.sqrt(mean_squared_error(y_test, lr_preds)), 2),
        }

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    if len(X_test) > 0:
        rf_preds = rf.predict(X_test)
        results["Random Forest"] = {
            "mae": round(mean_absolute_error(y_test, rf_preds), 2),
            "rmse": round(np.sqrt(mean_squared_error(y_test, rf_preds)), 2),
        }

    # Future forecast using RF (better for non-linear patterns)
    last_window = series.values[-window:]
    future_preds = []
    future_dates = []
    current = list(last_window)

    last_date = series.index[-1]
    for i in range(days_ahead):
        x = np.array(current[-window:]).reshape(1, -1)
        pred = max(0, rf.predict(x)[0])
        future_preds.append(round(pred, 2))
        future_dates.append(last_date + pd.Timedelta(days=i + 1))
        current.append(pred)

    # Historical daily for chart
    hist_dates = series.index.tolist()
    hist_values = series.values.tolist()

    return {
        "model_metrics": results,
        "future_dates": [d.strftime("%Y-%m-%d") for d in future_dates],
        "future_preds": future_preds,
        "historical_dates": [d.strftime("%Y-%m-%d") for d in hist_dates],
        "historical_values": hist_values,
        "next_week_total": round(sum(future_preds[:7]), 2),
        "next_month_total": round(sum(future_preds[:30]), 2),
    }


def forecast_by_category(df: pd.DataFrame, days_ahead: int = 30) -> dict:
    """Forecast spending per category for next month."""
    result = {}
    for cat, grp in df.groupby("Category"):
        monthly = grp.groupby(grp["Date"].dt.to_period("M"))["Amount"].sum()
        if len(monthly) < 2:
            result[cat] = {"predicted": round(monthly.mean(), 2), "trend": "stable"}
            continue
        vals = monthly.values
        # Simple linear trend
        x = np.arange(len(vals)).reshape(-1, 1)
        lr = LinearRegression()
        lr.fit(x, vals)
        next_pred = max(0, lr.predict([[len(vals)]])[0])
        trend = "up" if lr.coef_[0] > 0 else "down"
        result[cat] = {"predicted": round(next_pred, 2), "trend": trend}
    return result
