# 💹 SpendLens AI — Smart Expense Advisor System

> A production-grade AI-powered expense analytics and financial advisory platform built with Python, Streamlit, Machine Learning, Deep Learning, and NLP.

---

## 🌟 Features

| Feature | Technology |
|---|---|
| Auto expense classification | TF-IDF + Naive Bayes / Logistic Regression / SVM / LightGBM |
| Anomaly detection | Isolation Forest + Statistical methods |
| Expense forecasting | Linear Regression + Random Forest + LSTM |
| Interactive dashboards | Plotly (12+ chart types) |
| AI financial advisor | Rule-based engine + LLM prompt builder |
| Financial health score | Weighted scoring (0–100) |
| Data persistence | SQLite database |
| Export | CSV predictions + JSON reports |

---

## 🚀 Quick Start

### 1. Clone / Download
```bash
cd spendlens_ai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`

---

## 📁 Project Structure

```
spendlens_ai/
├── app.py                      # Main Streamlit app (7 pages)
├── requirements.txt
├── README.md
│
├── data/
│   └── sample_expenses.csv     # 120-row sample dataset
│
├── database/
│   ├── db.py                   # SQLite utilities
│   └── spendlens.db             # Auto-created on first run
│
├── models/
│   ├── category_model.pkl      # Trained classifier (auto-saved)
│   └── lstm_expense_model.h5   # LSTM model (auto-saved)
│
├── ml/
│   ├── preprocess.py           # TF-IDF + Label encoding
│   ├── train_classifier.py     # Multi-model training + evaluation
│   ├── predict_category.py     # Inference pipeline
│   ├── anomaly_detector.py     # Isolation Forest + alert engine
│   └── forecasting.py          # LR + RF forecasting
│
├── dl/
│   └── lstm_model.py           # LSTM deep learning forecaster
│
├── ai/
│   └── advisor.py              # AI advisor engine
│
├── dashboard/
│   └── charts.py               # 12+ Plotly chart generators
│
└── utils/
    └── helpers.py              # Shared utilities
```

---

## 📊 Pages

1. **🏠 Home** — Feature overview and quick-start guide
2. **📤 Upload Data** — CSV/Excel upload with auto-preprocessing
3. **📊 Dashboard** — Full analytics with filters and charts
4. **🤖 ML Classifier** — Train & compare 4 NLP classifiers
5. **🔮 Forecasting** — ML + LSTM future expense prediction
6. **🧠 AI Advisor** — Health score, recommendations, budgets
7. **⚙️ Settings** — About, data management, deployment info

---

## 📄 Data Format

Your CSV/Excel should have these columns:

| Column | Required | Notes |
|---|---|---|
| `Date` | ✅ | Any date format |
| `Description` | ✅ | Transaction description |
| `Amount` | ✅ | Positive numbers (₹) |
| `Category` | ❌ | Auto-predicted if missing |

---

## 🤖 ML Models

### Classifier
- **Input:** Transaction description (text)
- **Vectorization:** TF-IDF (500 features, bigrams)
- **Models:** Naive Bayes, Logistic Regression, Linear SVM, LightGBM
- **Auto-selects** the highest-accuracy model

### Forecaster
- **Linear Regression** — Baseline trend model
- **Random Forest** — Ensemble with sliding-window features
- **LSTM** — Sequential deep learning (requires TensorFlow)

---

## 🧠 AI Advisor Engine

The advisor generates:
- **Financial Health Score** (0–100) based on spending ratios, EMI burden, savings rate, anomalies
- **Personalized recommendations** with priority levels (high/medium/low)
- **Budget suggestions** per category (50/30/20 framework)
- **Smart tips** contextually selected from spending patterns
- **LLM-ready prompt** — paste into Claude, ChatGPT, or Gemini for generative advice

---

## 🚀 Deployment

### Streamlit Cloud
1. Push to GitHub
2. Go to share.streamlit.io → New app → select repo
3. Set main file: `app.py`

### Render / Railway
```bash
# Start command
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

---

## 📦 Requirements

```
streamlit==1.32.0
pandas==2.1.4
numpy==1.26.4
scikit-learn==1.4.0
lightgbm==4.3.0
plotly==5.18.0
joblib==1.3.2
openpyxl==3.1.2
fpdf2==2.7.8
scipy==1.12.0
```

TensorFlow is **optional** — the LSTM module gracefully falls back to a moving-average forecaster if TF is not installed.

---

## 📝 License

MIT License — free to use, modify, and deploy.

---

*Built with ❤️ using Python + Streamlit + ML/DL*
