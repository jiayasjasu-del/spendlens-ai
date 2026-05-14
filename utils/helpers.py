"""
utils/helpers.py — Shared utility functions for SpendLens AI
"""
import pandas as pd
import numpy as np
from pathlib import Path
import re

CATEGORIES = [
    "Food", "Shopping", "Travel", "Bills",
    "Entertainment", "Medical", "Education", "EMI", "Other"
]

CATEGORY_KEYWORDS = {
    "Food": [
        "swiggy", "zomato", "food", "restaurant", "cafe", "lunch", "dinner",
        "breakfast", "pizza", "burger", "biryani", "snack", "dessert", "coffee",
        "tea", "grocery", "supermarket", "kitchen", "meal", "eat", "sushi",
        "chinese", "thai", "cake", "bread", "hotel", "dine",
    ],
    "Shopping": [
        "amazon", "flipkart", "myntra", "ajio", "nykaa", "shopping", "mall",
        "purchase", "order", "buy", "store", "clothes", "fashion", "bag",
        "shoes", "cosmetics", "gadget", "electronics", "watch", "jewellery",
    ],
    "Travel": [
        "uber", "ola", "rapido", "metro", "bus", "train", "auto", "cab",
        "petrol", "fuel", "flight", "ticket", "travel", "transport", "ride",
        "commute", "taxi", "airfare", "booking", "trip",
    ],
    "Bills": [
        "electricity", "water", "gas", "internet", "broadband", "mobile",
        "bill", "recharge", "utility", "wifi", "dth", "cable", "phone",
        "telephone", "postpaid", "prepaid",
    ],
    "Entertainment": [
        "netflix", "amazon prime", "spotify", "hotstar", "youtube", "disney",
        "movie", "pvr", "inox", "cinema", "concert", "game", "gaming",
        "subscription", "streaming", "music", "show", "theatre",
    ],
    "Medical": [
        "pharmacy", "medicine", "doctor", "hospital", "clinic", "medical",
        "lab", "test", "health", "gym", "fitness", "chemist", "drug",
        "consultation", "checkup", "diagnostic",
    ],
    "Education": [
        "udemy", "coursera", "book", "course", "class", "school", "college",
        "tuition", "fee", "library", "study", "learning", "training",
        "workshop", "seminar", "exam",
    ],
    "EMI": [
        "emi", "loan", "mortgage", "installment", "credit", "finance",
        "repayment", "home loan", "car loan", "personal loan",
    ],
}


def load_and_validate(file) -> pd.DataFrame:
    """
    Load CSV or Excel file into DataFrame.
    Validates and normalises required columns.
    """
    name = getattr(file, "name", str(file))
    if name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    df.columns = df.columns.str.strip()

    # Flexible column matching
    col_map = {}
    for col in df.columns:
        lc = col.lower()
        if "date" in lc:
            col_map[col] = "Date"
        elif "desc" in lc or "narr" in lc or "note" in lc or "particular" in lc:
            col_map[col] = "Description"
        elif "amount" in lc or "debit" in lc or "spend" in lc or "value" in lc:
            col_map[col] = "Amount"
        elif "category" in lc or "type" in lc or "tag" in lc:
            col_map[col] = "Category"
    df.rename(columns=col_map, inplace=True)

    required = ["Date", "Description", "Amount"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and enrich expense DataFrame."""
    df = df.copy()

    # Parse dates
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.dropna(subset=["Date"], inplace=True)

    # Amount: coerce, drop negatives/zero
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df.dropna(subset=["Amount"], inplace=True)
    df = df[df["Amount"] > 0]

    # Description: clean text
    df["Description"] = df["Description"].astype(str).str.strip()

    # Remove duplicates
    df.drop_duplicates(subset=["Date", "Description", "Amount"], inplace=True)

    # Derived time columns
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["MonthName"] = df["Date"].dt.strftime("%b %Y")
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["DayOfWeek"] = df["Date"].dt.day_name()
    df["DayOfWeekNum"] = df["Date"].dt.dayofweek
    df["Hour"] = 0  # hour not in dataset; placeholder
    df["IsWeekend"] = df["DayOfWeekNum"].isin([5, 6]).astype(int)

    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def keyword_classify(description: str) -> str:
    """Rule-based fallback classification using keyword matching."""
    desc_lower = description.lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in desc_lower:
                return cat
    return "Other"


def clean_text(text: str) -> str:
    """Normalise description text for NLP."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def month_label(year: int, month: int) -> str:
    return pd.Timestamp(year=year, month=month, day=1).strftime("%b %Y")


def format_inr(amount: float) -> str:
    """Format number as Indian Rupee string."""
    return f"₹{amount:,.0f}"


def get_model_dir() -> Path:
    p = Path(__file__).parent.parent / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p
