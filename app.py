"""
app.py — SpendLens AI: Smart Expense Advisor System
Main Streamlit application entry point.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import numpy as np
import io
import json
from datetime import datetime

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="SpendLens AI — Smart Expense Advisor",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Internal imports ───────────────────────────────────────────────────────────
from utils.helpers import load_and_validate, preprocess, format_inr, keyword_classify
from ml.train_classifier import train_classifier
from ml.predict_category import predict_categories
from ml.anomaly_detector import detect_anomalies, generate_alerts
from ml.forecasting import forecast_ml, forecast_by_category
from dl.lstm_model import train_lstm
from ai.advisor import (
    compute_health_score, generate_recommendations,
    get_smart_tips, build_llm_prompt
)
from dashboard.charts import (
    monthly_trend_chart, daily_spending_chart, category_pie_chart,
    top_categories_bar, spending_heatmap, forecast_chart,
    category_forecast_chart, health_score_gauge,
    confusion_matrix_chart, budget_comparison_chart
)
from database.db import init_db, save_upload, save_expenses, save_report

init_db()

# ── CSS styling ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

:root {
    --bg-deep: #020817;
    --bg-card: #0f172a;
    --bg-glass: rgba(15,23,42,0.8);
    --accent: #6C63FF;
    --accent2: #43D9A2;
    --accent3: #FF6584;
    --text: #E2E8F0;
    --text-muted: #94A3B8;
    --border: rgba(108,99,255,0.2);
}

html, body, [data-testid="stApp"] {
    background: var(--bg-deep) !important;
    color: var(--text) !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0f1e 0%, #0f172a 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Hide default header */
header[data-testid="stHeader"] { background: transparent !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), #9C8FFF) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(108,99,255,0.4) !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size: 0.8rem !important; }
[data-testid="stMetricValue"] { color: var(--text) !important; font-size: 1.6rem !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

/* Upload area */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}

/* Tabs */
[data-testid="stTabs"] button {
    color: var(--text-muted) !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}

/* Selectbox / slider */
.stSelectbox > div, .stSlider > div { color: var(--text) !important; }

/* Alerts */
.alert-high {
    background: rgba(255,101,132,0.12);
    border-left: 3px solid #FF6584;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.9rem;
}
.alert-medium {
    background: rgba(255,179,71,0.12);
    border-left: 3px solid #FFB347;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.9rem;
}
.alert-low {
    background: rgba(67,217,162,0.12);
    border-left: 3px solid #43D9A2;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.9rem;
}

/* Cards */
.fin-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin: 0.5rem 0;
}

.kpi-row { display: flex; gap: 1rem; flex-wrap: wrap; margin: 0.8rem 0; }
.kpi-box {
    flex: 1; min-width: 160px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.kpi-label { color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; }
.kpi-value { color: var(--text); font-size: 1.5rem; font-weight: 700; font-family: 'Space Mono', monospace; }
.kpi-sub { color: var(--text-muted); font-size: 0.75rem; margin-top: 2px; }

/* Recommendation cards */
.rec-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: 0.6rem 0;
    display: flex;
    gap: 1rem;
    align-items: flex-start;
}
.rec-icon { font-size: 1.6rem; }
.rec-title { font-weight: 600; font-size: 0.95rem; color: var(--text); }
.rec-detail { color: var(--text-muted); font-size: 0.83rem; margin-top: 2px; }
.rec-savings { color: var(--accent2); font-size: 0.8rem; margin-top: 4px; font-weight: 600; }
.badge-high { background: rgba(255,101,132,0.2); color: #FF6584; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: 700; }
.badge-medium { background: rgba(255,179,71,0.2); color: #FFB347; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: 700; }
.badge-low { background: rgba(67,217,162,0.2); color: #43D9A2; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: 700; }

.logo-text {
    font-family: 'Space Mono', monospace;
    font-size: 1.3rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6C63FF, #43D9A2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.page-title {
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 0.2rem;
}
.page-sub {
    color: var(--text-muted);
    font-size: 0.9rem;
    margin-bottom: 1.5rem;
}
.tip-box {
    background: rgba(108,99,255,0.1);
    border: 1px solid rgba(108,99,255,0.3);
    border-radius: 10px;
    padding: 0.7rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.85rem;
    color: var(--text);
}
</style>
""", unsafe_allow_html=True)


# ── Session state init ─────────────────────────────────────────────────────────
for key, val in {
    "df": None,
    "upload_id": None,
    "filename": "",
    "classifier_metrics": None,
    "alerts": [],
    "forecast_ml": None,
    "forecast_lstm": None,
    "cat_forecast": None,
    "health": None,
    "recs": None,
    "budgets": None,
    "income": 0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="logo-text">💹 SpendLens AI</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#94A3B8;font-size:0.8rem;margin-top:0;'>Smart Expense Advisor</p>", unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "Navigation",
        ["🏠 Home", "📤 Upload Data", "📊 Dashboard", "🤖 ML Classifier",
         "🔮 Forecasting", "🧠 AI Advisor", "⚙️ Settings"],
        label_visibility="collapsed",
    )
    st.divider()

    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown("**📂 Loaded Dataset**")
        st.markdown(f"<p style='color:#94A3B8;font-size:0.8rem;'>📄 {st.session_state.filename}<br>"
                    f"📅 {df['Date'].min().date()} → {df['Date'].max().date()}<br>"
                    f"🔢 {len(df):,} transactions</p>", unsafe_allow_html=True)
        st.divider()

    st.markdown("<p style='color:#475569;font-size:0.72rem;text-align:center;margin-top:auto;'>SpendLens AI v1.0<br>Built with Streamlit + ML</p>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown('<div class="page-title">Welcome to SpendLens AI 💹</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Your intelligent financial analytics & advisory platform</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    features = [
        ("📤", "Upload & Auto-Classify", "Upload CSV/Excel expense data. AI classifies every transaction automatically using NLP + ML."),
        ("📊", "Advanced Analytics", "Interactive dashboards with spending trends, heatmaps, category breakdowns and savings analysis."),
        ("🔮", "Expense Forecasting", "Linear Regression, Random Forest and LSTM Deep Learning predict your future spending."),
        ("⚠️", "Overspending Detection", "Isolation Forest detects anomalies. Smart alerts flag food addiction, subscription bloat, and more."),
        ("🧠", "AI Financial Advisor", "Personalized recommendations, financial health score (0–100), budget goals and savings tips."),
        ("📥", "Export Reports", "Download full PDF reports and CSV predictions for offline review."),
    ]
    cols = [c1, c2, c3, c1, c2, c3]
    for col, (icon, title, desc) in zip(cols, features):
        with col:
            st.markdown(f"""
            <div class="fin-card">
                <div style="font-size:1.8rem;margin-bottom:0.4rem;">{icon}</div>
                <div style="font-weight:600;font-size:0.95rem;color:#E2E8F0;">{title}</div>
                <div style="color:#94A3B8;font-size:0.82rem;margin-top:0.3rem;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🚀 Quick Start")
    st.markdown("""
    <div class="fin-card">
    <ol style="color:#94A3B8;font-size:0.9rem;line-height:2;">
    <li>Go to <b style="color:#6C63FF;">📤 Upload Data</b> and upload your expense CSV or Excel file.</li>
    <li>The system automatically preprocesses, classifies, and detects anomalies.</li>
    <li>Visit <b style="color:#6C63FF;">📊 Dashboard</b> for full analytics and spending insights.</li>
    <li>Check <b style="color:#6C63FF;">🔮 Forecasting</b> to see predicted future expenses.</li>
    <li>Get personalized advice on <b style="color:#6C63FF;">🧠 AI Advisor</b>.</li>
    </ol>
    <p style="color:#94A3B8;font-size:0.85rem;">
    👉 <b style="color:#43D9A2;">No data?</b> A sample dataset is included — just click Upload Data and use the sample file button.
    </p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: UPLOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📤 Upload Data":
    st.markdown('<div class="page-title">📤 Upload Expense Data</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Upload a CSV or Excel file with your transaction history.</div>', unsafe_allow_html=True)

    col_upload, col_info = st.columns([3, 2])

    with col_upload:
        st.markdown("**Required columns:** `Date`, `Description`, `Amount` &nbsp;|&nbsp; Optional: `Category`")
        uploaded = st.file_uploader(
            "Drop your expense file here",
            type=["csv", "xlsx", "xls"],
            label_visibility="collapsed",
        )

        st.markdown("**Or use the built-in sample dataset:**")
        if st.button("📂 Load Sample Dataset"):
            sample_path = Path(__file__).parent / "data" / "sample_expenses.csv"
            if sample_path.exists():
                with open(sample_path, "rb") as f:
                    uploaded = io.BytesIO(f.read())
                    uploaded.name = "sample_expenses.csv"
                    st.success("✅ Sample dataset loaded!")
            else:
                st.error("Sample file not found.")

    with col_info:
        st.markdown("""
        <div class="fin-card">
        <b>📌 Supported Formats</b><br><br>
        <span style="color:#94A3B8;font-size:0.85rem;">
        • CSV (.csv)<br>
        • Excel (.xlsx, .xls)<br><br>
        <b style="color:#E2E8F0;">Expected Columns:</b><br>
        • <code>Date</code> — any date format<br>
        • <code>Description</code> — transaction note<br>
        • <code>Amount</code> — expense value (₹)<br>
        • <code>Category</code> — optional, auto-predicted if missing
        </span>
        </div>
        """, unsafe_allow_html=True)

    if uploaded is not None:
        try:
            with st.spinner("⏳ Processing your data..."):
                raw = load_and_validate(uploaded)
                df = preprocess(raw)

                has_category = (
                    "Category" in df.columns and
                    df["Category"].notna().sum() > len(df) * 0.5
                )

                if not has_category:
                    st.info("🔍 Category column missing or incomplete — predicting categories with ML + NLP...")
                    df["Category"] = predict_categories(df["Description"])
                else:
                    missing_cats = df["Category"].isna() | (df["Category"].str.strip() == "")
                    if missing_cats.any():
                        df.loc[missing_cats, "Category"] = predict_categories(
                            df.loc[missing_cats, "Description"]
                        )

                df = detect_anomalies(df)
                alerts = generate_alerts(df)

                filename = getattr(uploaded, "name", "dataset.csv")
                upload_id = save_upload(filename, len(df))
                save_expenses(df, upload_id)

                st.session_state.df = df
                st.session_state.upload_id = upload_id
                st.session_state.filename = filename
                st.session_state.alerts = alerts
                st.session_state.classifier_metrics = None
                st.session_state.forecast_ml = None
                st.session_state.forecast_lstm = None

            st.success(f"✅ Dataset processed: **{len(df):,} transactions** | **{df['Date'].min().date()}** to **{df['Date'].max().date()}**")

            # Preview
            st.markdown("### 👀 Data Preview")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Total Transactions", f"{len(df):,}")
            with c2:
                st.metric("Total Spent", format_inr(df["Amount"].sum()))
            with c3:
                st.metric("Categories Found", df["Category"].nunique())
            with c4:
                st.metric("Anomalies Detected", int(df["IsAnomaly"].sum()))

            st.dataframe(
                df[["Date", "Description", "Amount", "Category", "IsAnomaly"]].head(20),
                use_container_width=True,
                hide_index=True,
            )

            if alerts:
                st.markdown("### ⚠️ Initial Alerts")
                for a in alerts[:5]:
                    cls = f"alert-{a.get('severity','low')}"
                    st.markdown(f'<div class="{cls}">{a["message"]}</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
            import traceback
            st.code(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Dashboard":
    if st.session_state.df is None:
        st.warning("⚠️ Please upload a dataset first (📤 Upload Data).")
        st.stop()

    df = st.session_state.df.copy()

    st.markdown('<div class="page-title">📊 Expense Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Comprehensive analytics of your spending patterns</div>', unsafe_allow_html=True)

    # ── Filters ────────────────────────────────────────────────────────────────
    with st.expander("🔧 Filters", expanded=False):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            all_cats = ["All"] + sorted(df["Category"].unique().tolist())
            sel_cat = st.selectbox("Category", all_cats)
        with fc2:
            min_d = df["Date"].min().date()
            max_d = df["Date"].max().date()
            date_range = st.date_input("Date Range", value=(min_d, max_d),
                                       min_value=min_d, max_value=max_d)
        with fc3:
            amt_max = int(df["Amount"].max()) + 1
            amt_range = st.slider("Amount Range (₹)", 0, amt_max, (0, amt_max))

    # Apply filters
    fdf = df.copy()
    if sel_cat != "All":
        fdf = fdf[fdf["Category"] == sel_cat]
    if len(date_range) == 2:
        fdf = fdf[(fdf["Date"].dt.date >= date_range[0]) & (fdf["Date"].dt.date <= date_range[1])]
    fdf = fdf[(fdf["Amount"] >= amt_range[0]) & (fdf["Amount"] <= amt_range[1])]

    if fdf.empty:
        st.warning("No data matches your filters.")
        st.stop()

    # ── KPI Row ────────────────────────────────────────────────────────────────
    total = fdf["Amount"].sum()
    avg_monthly = fdf.groupby(fdf["Date"].dt.to_period("M"))["Amount"].sum().mean()
    avg_daily = fdf.groupby(fdf["Date"].dt.date)["Amount"].sum().mean()
    top_cat = fdf.groupby("Category")["Amount"].sum().idxmax()
    anomaly_count = int(fdf["IsAnomaly"].sum())

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: st.metric("💸 Total Spent", format_inr(total))
    with k2: st.metric("📅 Avg Monthly", format_inr(avg_monthly))
    with k3: st.metric("📆 Avg Daily", format_inr(avg_daily))
    with k4: st.metric("🏷️ Top Category", top_cat)
    with k5: st.metric("🔍 Anomalies", anomaly_count)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Monthly Trend + Pie ─────────────────────────────────────────────
    col1, col2 = st.columns([3, 2])
    with col1:
        st.plotly_chart(monthly_trend_chart(fdf), use_container_width=True, config={"displayModeBar": False})
    with col2:
        st.plotly_chart(category_pie_chart(fdf), use_container_width=True, config={"displayModeBar": False})

    # ── Row 2: Daily + Top Categories ─────────────────────────────────────────
    col3, col4 = st.columns([3, 2])
    with col3:
        st.plotly_chart(daily_spending_chart(fdf), use_container_width=True, config={"displayModeBar": False})
    with col4:
        st.plotly_chart(top_categories_bar(fdf), use_container_width=True, config={"displayModeBar": False})

    # ── Heatmap ────────────────────────────────────────────────────────────────
    st.plotly_chart(spending_heatmap(fdf), use_container_width=True, config={"displayModeBar": False})

    # ── Anomaly Table ──────────────────────────────────────────────────────────
    st.markdown("### 🔍 Anomalous Transactions")
    anom_df = fdf[fdf["IsAnomaly"] == 1][["Date", "Description", "Amount", "Category", "AnomalyScore"]]
    if not anom_df.empty:
        anom_df_display = anom_df.sort_values("AnomalyScore", ascending=False).head(10)
        st.dataframe(anom_df_display, use_container_width=True, hide_index=True)
    else:
        st.success("✅ No anomalous transactions detected.")

    # ── Alerts ─────────────────────────────────────────────────────────────────
    if st.session_state.alerts:
        st.markdown("### ⚠️ Smart Alerts")
        for a in st.session_state.alerts:
            cls = f"alert-{a.get('severity','low')}"
            st.markdown(f'<div class="{cls}">{a["message"]}</div>', unsafe_allow_html=True)

    # ── Export ─────────────────────────────────────────────────────────────────
    st.markdown("### 📥 Export")
    csv_data = fdf.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Filtered Data (CSV)", csv_data,
                       file_name="spendlens_filtered.csv", mime="text/csv")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ML CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🤖 ML Classifier":
    if st.session_state.df is None:
        st.warning("⚠️ Please upload a dataset first.")
        st.stop()

    df = st.session_state.df

    st.markdown('<div class="page-title">🤖 ML Expense Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Train and compare NLP-based classification models</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="fin-card">
    <b>How it works:</b><br>
    <span style="color:#94A3B8;font-size:0.85rem;">
    Transaction descriptions are vectorized with <b>TF-IDF</b> and classified using 
    <b>Naive Bayes</b>, <b>Logistic Regression</b>, <b>Linear SVM</b>, and <b>LightGBM</b>.
    The best model is saved and used for all future predictions.
    </span>
    </div>
    """, unsafe_allow_html=True)

    col_btn, _ = st.columns([2, 5])
    with col_btn:
        train_btn = st.button("🚀 Train Classifier", use_container_width=True)

    if train_btn:
        with st.spinner("🏋️ Training models…"):
            _, _, _, metrics = train_classifier(df)
            st.session_state.classifier_metrics = metrics

    metrics = st.session_state.classifier_metrics
    if metrics:
        if "error" in metrics:
            st.error(metrics["error"])
        else:
            st.success(f"✅ Best model: **{metrics['best_model']}** — Accuracy: **{metrics['best_accuracy']:.1f}%**")

            # Model comparison
            st.markdown("### 📈 Model Comparison")
            model_names = [k for k in metrics["all_results"] if "accuracy" in metrics["all_results"][k]]
            accs = [metrics["all_results"][k]["accuracy"] for k in model_names]

            import plotly.graph_objects as go
            palette = ["#6C63FF", "#FF6584", "#43D9A2", "#FFB347"]
            fig_acc = go.Figure(go.Bar(
                x=model_names, y=accs,
                marker_color=palette[:len(model_names)],
                text=[f"{a:.1f}%" for a in accs],
                textposition="outside",
                textfont=dict(color="#E2E8F0"),
            ))
            fig_acc.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.6)",
                font=dict(family="IBM Plex Sans", color="#E2E8F0"),
                title=dict(text="Model Accuracy Comparison", font=dict(size=14, color="#E2E8F0")),
                yaxis=dict(range=[0, 110], gridcolor="rgba(255,255,255,0.05)"),
                margin=dict(l=20, r=20, t=50, b=30),
            )
            st.plotly_chart(fig_acc, use_container_width=True, config={"displayModeBar": False})

            # Best model details
            best = metrics["best_model"]
            best_res = metrics["all_results"][best]
            if "report" in best_res:
                st.markdown(f"### 📋 Classification Report — {best}")
                report_data = []
                for cls, vals in best_res["report"].items():
                    if isinstance(vals, dict) and "precision" in vals:
                        report_data.append({
                            "Class": cls,
                            "Precision": f"{vals['precision']:.2f}",
                            "Recall": f"{vals['recall']:.2f}",
                            "F1-Score": f"{vals['f1-score']:.2f}",
                            "Support": int(vals.get("support", 0)),
                        })
                if report_data:
                    st.dataframe(pd.DataFrame(report_data), use_container_width=True, hide_index=True)

            if "confusion_matrix" in best_res and best_res["confusion_matrix"]:
                st.markdown("### 🔢 Confusion Matrix")
                cm_classes = best_res.get("confusion_classes", metrics["classes"])
                st.plotly_chart(
                    confusion_matrix_chart(best_res["confusion_matrix"], cm_classes),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

            # Classified sample
            st.markdown("### 🏷️ Sample Classified Transactions (Top 20)")
            sample = df[["Date", "Description", "Amount", "Category"]].head(20)
            st.dataframe(sample, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: FORECASTING
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔮 Forecasting":
    if st.session_state.df is None:
        st.warning("⚠️ Please upload a dataset first.")
        st.stop()

    df = st.session_state.df

    st.markdown('<div class="page-title">🔮 Expense Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Predict future spending using ML and Deep Learning models</div>', unsafe_allow_html=True)

    days_ahead = st.slider("Forecast horizon (days)", 7, 60, 30, step=7)

    c_ml, c_lstm = st.columns(2)
    with c_ml:
        if st.button("📈 Run ML Forecast (LR + RF)", use_container_width=True):
            with st.spinner("Training Linear Regression + Random Forest…"):
                result = forecast_ml(df, days_ahead=days_ahead)
                cat_result = forecast_by_category(df)
                st.session_state.forecast_ml = result
                st.session_state.cat_forecast = cat_result
    with c_lstm:
        if st.button("🧠 Run LSTM Forecast (Deep Learning)", use_container_width=True):
            with st.spinner("Training LSTM neural network…"):
                lstm_result = train_lstm(df)
                st.session_state.forecast_lstm = lstm_result

    # ── ML Results ─────────────────────────────────────────────────────────────
    ml_res = st.session_state.forecast_ml
    if ml_res:
        if "error" in ml_res:
            st.error(ml_res["error"])
        else:
            st.markdown("### 📈 ML Forecast Results")
            k1, k2 = st.columns(2)
            with k1:
                st.metric("📅 Next 7 Days (predicted)", format_inr(ml_res["next_week_total"]))
            with k2:
                st.metric("🗓️ Next Month (predicted)", format_inr(ml_res["next_month_total"]))

            st.plotly_chart(
                forecast_chart(
                    ml_res["historical_dates"], ml_res["historical_values"],
                    ml_res["future_dates"], ml_res["future_preds"],
                    "Random Forest"
                ),
                use_container_width=True, config={"displayModeBar": False}
            )

            if "model_metrics" in ml_res and ml_res["model_metrics"]:
                st.markdown("#### 🎯 Model Accuracy")
                met_rows = []
                for name, m in ml_res["model_metrics"].items():
                    met_rows.append({"Model": name, "MAE (₹)": f"{m['mae']:,.0f}", "RMSE (₹)": f"{m['rmse']:,.0f}"})
                st.dataframe(pd.DataFrame(met_rows), use_container_width=True, hide_index=True)

            # Category forecast
            if st.session_state.cat_forecast:
                st.markdown("### 🏷️ Category-wise Forecast (Next Month)")
                st.plotly_chart(
                    category_forecast_chart(st.session_state.cat_forecast),
                    use_container_width=True, config={"displayModeBar": False}
                )
                cat_rows = []
                for cat, info in st.session_state.cat_forecast.items():
                    trend_icon = "📈" if info["trend"] == "up" else "📉"
                    cat_rows.append({
                        "Category": cat,
                        "Predicted (₹)": f"₹{info['predicted']:,.0f}",
                        "Trend": f"{trend_icon} {info['trend'].capitalize()}",
                    })
                st.dataframe(pd.DataFrame(cat_rows), use_container_width=True, hide_index=True)

            # Download
            pred_df = pd.DataFrame({
                "Date": ml_res["future_dates"],
                "Predicted_Amount": ml_res["future_preds"],
            })
            st.download_button(
                "⬇️ Download Predictions (CSV)",
                pred_df.to_csv(index=False).encode("utf-8"),
                "spendlens_predictions.csv", "text/csv"
            )

    # ── LSTM Results ───────────────────────────────────────────────────────────
    lstm_res = st.session_state.forecast_lstm
    if lstm_res:
        if "error" in lstm_res:
            st.error(lstm_res["error"])
        else:
            st.markdown(f"### 🧠 LSTM Forecast — {lstm_res.get('model','LSTM')}")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Next 7 Days", format_inr(lstm_res["next_week_total"]))
            with col2:
                st.metric("Next 30 Days", format_inr(lstm_res["next_month_total"]))
            with col3:
                if lstm_res.get("mae"):
                    st.metric("MAE", f"₹{lstm_res['mae']:,.0f}")
                st.metric("TF Available", "✅ Yes" if lstm_res.get("tf_available") else "⚡ Fallback Mode")

            st.plotly_chart(
                forecast_chart(
                    lstm_res["historical_dates"], lstm_res["historical_values"],
                    lstm_res["future_dates"], lstm_res["future_preds"],
                    lstm_res.get("model", "LSTM")
                ),
                use_container_width=True, config={"displayModeBar": False}
            )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: AI ADVISOR
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🧠 AI Advisor":
    if st.session_state.df is None:
        st.warning("⚠️ Please upload a dataset first.")
        st.stop()

    df = st.session_state.df

    st.markdown('<div class="page-title">🧠 AI Financial Advisor</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Personalized financial advice powered by rule-based AI</div>', unsafe_allow_html=True)

    col_input, _ = st.columns([2, 3])
    with col_input:
        income = st.number_input(
            "💰 Monthly Income (₹) — Optional",
            min_value=0, max_value=10_000_000, value=st.session_state.income, step=5000,
            help="Enter your monthly net income for savings analysis. Leave 0 to skip."
        )
        st.session_state.income = income

    if st.button("🧠 Generate AI Analysis", use_container_width=False):
        with st.spinner("Analyzing your finances…"):
            health = compute_health_score(df, income)
            recs, budgets = generate_recommendations(df, income)
            tips = get_smart_tips(df)
            st.session_state.health = health
            st.session_state.recs = recs
            st.session_state.budgets = budgets

            report = {"health": health, "recommendations": recs, "budgets": budgets}
            save_report(st.session_state.upload_id, health["score"], report)

    health = st.session_state.health
    recs = st.session_state.recs
    budgets = st.session_state.budgets

    if health:
        # ── Health Score ────────────────────────────────────────────────────────
        col_gauge, col_details = st.columns([2, 3])
        with col_gauge:
            st.plotly_chart(
                health_score_gauge(health["score"], health["grade"]),
                use_container_width=True, config={"displayModeBar": False}
            )
        with col_details:
            st.markdown(f"""
            <div class="fin-card" style="margin-top:1rem;">
            <div style="font-size:1.1rem;font-weight:700;color:#E2E8F0;">Financial Health: {health['label']}</div>
            <div style="color:#94A3B8;font-size:0.85rem;margin:0.5rem 0;">Score breakdown:</div>
            """, unsafe_allow_html=True)
            if health["deductions"]:
                for k, v in health["deductions"].items():
                    label = k.replace("_", " ").title()
                    st.markdown(f'<div style="color:#FF6584;font-size:0.82rem;">▼ -{v} pts: {label}</div>', unsafe_allow_html=True)
            if health["bonuses"]:
                for k, v in health["bonuses"].items():
                    label = k.replace("_", " ").title()
                    st.markdown(f'<div style="color:#43D9A2;font-size:0.82rem;">▲ +{v} pts: {label}</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Recommendations ─────────────────────────────────────────────────────
        if recs:
            st.markdown("### 💡 Personalized Recommendations")
            for rec in recs:
                priority_badge = f'<span class="badge-{rec["priority"]}">{rec["priority"].upper()}</span>'
                savings_line = (f'<div class="rec-savings">💚 Potential savings: {format_inr(rec["savings_estimate"])}/month</div>'
                                if rec.get("savings_estimate", 0) > 0 else "")
                st.markdown(f"""
                <div class="rec-card">
                    <div class="rec-icon">{rec["icon"]}</div>
                    <div>
                        <div class="rec-title">{rec["title"]} {priority_badge}</div>
                        <div class="rec-detail">{rec["detail"]}</div>
                        {savings_line}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ── Budget Comparison ───────────────────────────────────────────────────
        if budgets:
            st.markdown("### 📊 Budget Analysis")
            st.plotly_chart(
                budget_comparison_chart(budgets),
                use_container_width=True, config={"displayModeBar": False}
            )
            budget_rows = []
            for cat, info in budgets.items():
                over = info["actual"] > info["suggested"]
                status = "🔴 Over" if over else "🟢 OK"
                budget_rows.append({
                    "Category": cat,
                    "Actual (₹)": f"₹{info['actual']:,.0f}",
                    "Suggested (₹)": f"₹{info['suggested']:,.0f}",
                    "Status": status,
                })
            st.dataframe(pd.DataFrame(budget_rows), use_container_width=True, hide_index=True)

        # ── Smart Tips ──────────────────────────────────────────────────────────
        st.markdown("### 🌟 Smart Financial Tips")
        tips = get_smart_tips(df)
        for tip in tips:
            st.markdown(f'<div class="tip-box">{tip}</div>', unsafe_allow_html=True)

        # ── LLM Prompt ─────────────────────────────────────────────────────────
        with st.expander("🤖 LLM-Ready Advisor Prompt (Advanced)"):
            prompt = build_llm_prompt(df, health, income)
            st.markdown("""
            <div style="color:#94A3B8;font-size:0.85rem;margin-bottom:0.8rem;">
            Copy this prompt and paste it into any LLM (Claude, ChatGPT, Gemini) 
            to get AI-generated personalised financial advice:
            </div>
            """, unsafe_allow_html=True)
            st.code(prompt, language="text")

        # ── Export ─────────────────────────────────────────────────────────────
        st.markdown("### 📥 Export Report")
        report_json = json.dumps({
            "health_score": health,
            "recommendations": recs,
            "budgets": budgets,
            "generated_at": datetime.now().isoformat(),
        }, indent=2)
        st.download_button(
            "⬇️ Download Report (JSON)",
            report_json.encode("utf-8"),
            "spendlens_report.json",
            "application/json",
        )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: SETTINGS / ABOUT
# ─────────────────────────────────────────────────────────────────────────────
elif page == "⚙️ Settings":
    st.markdown('<div class="page-title">⚙️ Settings & About</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="fin-card">
        <b>📋 About SpendLens AI</b><br><br>
        <span style="color:#94A3B8;font-size:0.85rem;">
        SpendLens AI is a production-grade Smart Expense Advisor System built with:
        <br><br>
        • <b style="color:#E2E8F0;">Streamlit</b> — Modern web UI<br>
        • <b style="color:#E2E8F0;">Scikit-learn</b> — ML classification & forecasting<br>
        • <b style="color:#E2E8F0;">LightGBM</b> — Gradient boosting classifier<br>
        • <b style="color:#E2E8F0;">TensorFlow/Keras</b> — LSTM forecasting<br>
        • <b style="color:#E2E8F0;">Plotly</b> — Interactive charts<br>
        • <b style="color:#E2E8F0;">SQLite</b> — Local database<br>
        • <b style="color:#E2E8F0;">Pandas + NumPy</b> — Data processing<br>
        </span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="fin-card">
        <b>🔧 Data Management</b><br><br>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🗑️ Clear Session Data", use_container_width=True):
            for key in ["df", "upload_id", "filename", "classifier_metrics",
                        "alerts", "forecast_ml", "forecast_lstm", "cat_forecast",
                        "health", "recs", "budgets"]:
                st.session_state[key] = None if key not in ["alerts"] else []
            st.success("Session cleared!")

    st.markdown("""
    <div class="fin-card" style="margin-top:1rem;">
    <b>📦 Model Storage</b><br>
    <span style="color:#94A3B8;font-size:0.85rem;">
    • Category classifier: <code>models/category_model.pkl</code><br>
    • LSTM model: <code>models/lstm_expense_model.h5</code><br>
    • Database: <code>database/spendlens.db</code>
    </span>
    </div>

    <div class="fin-card" style="margin-top:1rem;">
    <b>🚀 Deployment</b><br>
    <span style="color:#94A3B8;font-size:0.85rem;">
    This app is ready for deployment on:<br>
    • <b style="color:#E2E8F0;">Streamlit Cloud</b> — streamlit.io/cloud<br>
    • <b style="color:#E2E8F0;">Render</b> — render.com<br>
    • <b style="color:#E2E8F0;">Railway</b> — railway.app<br><br>
    Run locally: <code>streamlit run app.py</code>
    </span>
    </div>
    """, unsafe_allow_html=True)
