"""
ml/anomaly_detector.py — Detect unusual expenses using Isolation Forest + statistics
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    """
    Mark anomalous transactions using Isolation Forest on amount features.
    Returns df with 'IsAnomaly' column added.
    """
    df = df.copy()

    features = ["Amount"]
    if "LogAmount" in df.columns:
        features.append("LogAmount")

    X = df[features].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
    preds = iso.fit_predict(X_scaled)
    df["IsAnomaly"] = (preds == -1).astype(int)
    df["AnomalyScore"] = -iso.score_samples(X_scaled)  # higher = more anomalous

    return df


def generate_alerts(df: pd.DataFrame) -> list[dict]:
    """
    Generate human-readable overspending alerts.
    Returns list of alert dicts with type, message, severity.
    """
    alerts = []
    df = df.copy()
    df["YearMonth"] = df["Date"].dt.to_period("M")
    months = df["YearMonth"].unique()

    if len(months) < 2:
        return alerts

    latest = months[-1]
    prev = months[-2]
    latest_df = df[df["YearMonth"] == latest]
    prev_df = df[df["YearMonth"] == prev]

    # 1. Category growth alerts
    for cat in df["Category"].unique():
        cur_amt = latest_df[latest_df["Category"] == cat]["Amount"].sum()
        prv_amt = prev_df[prev_df["Category"] == cat]["Amount"].sum()
        if prv_amt > 0:
            pct = (cur_amt - prv_amt) / prv_amt * 100
            if pct >= 30:
                severity = "high" if pct >= 60 else "medium"
                alerts.append({
                    "type": "category_growth",
                    "category": cat,
                    "message": f"⚠️ {cat} expenses increased by {pct:.0f}% vs last month.",
                    "severity": severity,
                    "pct_change": round(pct, 1),
                })

    # 2. Food delivery addiction (≥10 food txns/month)
    food_count = latest_df[latest_df["Category"] == "Food"].shape[0]
    if food_count >= 10:
        alerts.append({
            "type": "food_addiction",
            "message": f"🍔 {food_count} food delivery orders this month. Consider cooking at home!",
            "severity": "medium",
        })

    # 3. Weekend overspending
    weekend_df = df[df["IsWeekend"] == 1]
    weekday_df = df[df["IsWeekend"] == 0]
    if len(weekend_df) > 0 and len(weekday_df) > 0:
        avg_wknd = weekend_df["Amount"].mean()
        avg_wkdy = weekday_df["Amount"].mean()
        if avg_wknd > avg_wkdy * 1.5:
            alerts.append({
                "type": "weekend_spending",
                "message": f"📅 Weekend spending (avg ₹{avg_wknd:.0f}) is {avg_wknd/avg_wkdy:.1f}× weekday average.",
                "severity": "low",
            })

    # 4. Repeated subscriptions
    sub_keywords = ["netflix", "spotify", "amazon prime", "hotstar", "youtube premium", "disney"]
    subs = df[df["Description"].str.lower().str.contains("|".join(sub_keywords), na=False)]
    sub_total = subs["Amount"].sum()
    if sub_total > 2000:
        alerts.append({
            "type": "subscriptions",
            "message": f"📺 Total subscription spend: ₹{sub_total:.0f}. Review and cancel unused ones.",
            "severity": "low",
        })

    # 5. Anomalous transactions
    anomalies = df[df["IsAnomaly"] == 1]
    if len(anomalies) > 0:
        top = anomalies.nlargest(3, "Amount")
        for _, row in top.iterrows():
            alerts.append({
                "type": "anomaly",
                "message": f"🔍 Unusual expense: {row['Description']} — ₹{row['Amount']:.0f} on {row['Date'].date()}",
                "severity": "high",
            })

    # 6. Rapid total expense growth
    cur_total = latest_df["Amount"].sum()
    prv_total = prev_df["Amount"].sum()
    if prv_total > 0:
        growth = (cur_total - prv_total) / prv_total * 100
        if growth >= 25:
            alerts.append({
                "type": "total_growth",
                "message": f"📈 Total expenses grew by {growth:.0f}% this month (₹{cur_total:,.0f} vs ₹{prv_total:,.0f}).",
                "severity": "high" if growth >= 50 else "medium",
            })

    return alerts
