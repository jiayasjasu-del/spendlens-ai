"""
ai/advisor.py — Rule-based + LLM-ready AI Financial Advisor for SpendLens AI
"""
import pandas as pd
import numpy as np
from utils.helpers import format_inr, CATEGORIES


# ─── Financial Health Scoring ────────────────────────────────────────────────

def compute_health_score(df: pd.DataFrame, income: float = 0) -> dict:
    """
    Compute a 0–100 Financial Health Score from spending patterns.
    """
    score = 100
    deductions = {}
    bonuses = {}

    total_spend = df["Amount"].sum()
    avg_monthly = df.groupby(df["Date"].dt.to_period("M"))["Amount"].sum().mean()

    # 1. Savings ratio (if income provided)
    if income > 0:
        savings_ratio = max(0, (income - avg_monthly) / income)
        if savings_ratio >= 0.3:
            bonuses["savings_ratio"] = 10
        elif savings_ratio < 0.1:
            deductions["low_savings"] = 15
            score -= 15
        else:
            score -= 5
            deductions["moderate_savings"] = 5
    else:
        # No income: use spending trend
        monthly = df.groupby(df["Date"].dt.to_period("M"))["Amount"].sum()
        if len(monthly) >= 2:
            growth = (monthly.iloc[-1] - monthly.iloc[-2]) / monthly.iloc[-2] * 100
            if growth > 30:
                score -= 10
                deductions["expense_growth"] = 10

    # 2. Category breakdown penalties
    cat_totals = df.groupby("Category")["Amount"].sum()
    if total_spend > 0:
        food_pct = cat_totals.get("Food", 0) / total_spend * 100
        ent_pct = cat_totals.get("Entertainment", 0) / total_spend * 100
        shop_pct = cat_totals.get("Shopping", 0) / total_spend * 100

        if food_pct > 35:
            score -= 8
            deductions["high_food_pct"] = 8
        if ent_pct > 15:
            score -= 5
            deductions["high_entertainment"] = 5
        if shop_pct > 30:
            score -= 7
            deductions["high_shopping"] = 7

    # 3. Anomaly penalty
    if "IsAnomaly" in df.columns:
        anomaly_count = df["IsAnomaly"].sum()
        penalty = min(15, int(anomaly_count) * 3)
        if penalty > 0:
            score -= penalty
            deductions["anomalies"] = penalty

    # 4. EMI burden
    emi_total = cat_totals.get("EMI", 0)
    if income > 0 and emi_total > 0:
        emi_pct = emi_total / income * 100
        if emi_pct > 40:
            score -= 10
            deductions["high_emi"] = 10
        elif emi_pct > 25:
            score -= 5
            deductions["moderate_emi"] = 5

    # 5. Education spend bonus
    edu_pct = cat_totals.get("Education", 0) / max(total_spend, 1) * 100
    if edu_pct > 3:
        bonuses["education_investment"] = 5
        score += 5

    final_score = max(0, min(100, score))
    grade = (
        "A" if final_score >= 85 else
        "B" if final_score >= 70 else
        "C" if final_score >= 55 else
        "D" if final_score >= 40 else "F"
    )

    return {
        "score": round(final_score, 1),
        "grade": grade,
        "deductions": deductions,
        "bonuses": bonuses,
        "label": _score_label(final_score),
    }


def _score_label(score: float) -> str:
    if score >= 85:
        return "Excellent 🏆"
    elif score >= 70:
        return "Good 👍"
    elif score >= 55:
        return "Fair ⚡"
    elif score >= 40:
        return "Needs Work ⚠️"
    else:
        return "Critical 🚨"


# ─── Recommendations Engine ───────────────────────────────────────────────────

def generate_recommendations(df: pd.DataFrame, income: float = 0) -> list[dict]:
    """
    Analyse spending patterns and generate personalised financial advice.
    Returns list of recommendation dicts.
    """
    recs = []
    total_spend = df["Amount"].sum()
    cat_totals = df.groupby("Category")["Amount"].sum().to_dict()
    monthly_avg = df.groupby(df["Date"].dt.to_period("M"))["Amount"].sum().mean()

    # Food delivery
    food = cat_totals.get("Food", 0)
    if food / max(total_spend, 1) > 0.3:
        recs.append({
            "category": "Food",
            "icon": "🍔",
            "title": "Cut Down on Food Delivery",
            "detail": f"Food spend is {food/total_spend*100:.0f}% of total (₹{food:,.0f}). "
                      "Cook at home 3× per week to save ₹1,500–3,000/month.",
            "priority": "high",
            "savings_estimate": round(food * 0.35),
        })

    # Shopping
    shopping = cat_totals.get("Shopping", 0)
    if shopping / max(total_spend, 1) > 0.25:
        recs.append({
            "category": "Shopping",
            "icon": "🛍️",
            "title": "Review Discretionary Shopping",
            "detail": f"Shopping at ₹{shopping:,.0f} is {shopping/total_spend*100:.0f}% of spend. "
                      "Set a monthly shopping budget of ₹2,000 and use wishlists.",
            "priority": "high",
            "savings_estimate": round(shopping * 0.4),
        })

    # Entertainment / subscriptions
    ent = cat_totals.get("Entertainment", 0)
    sub_keywords = ["netflix", "spotify", "amazon prime", "hotstar", "youtube", "disney"]
    subs = df[df["Description"].str.lower().str.contains("|".join(sub_keywords), na=False)]
    sub_count = subs["Description"].str.lower().str.split().apply(
        lambda x: next((kw for kw in sub_keywords if any(kw in w for w in x)), None)
    ).nunique()
    if sub_count >= 3:
        recs.append({
            "category": "Entertainment",
            "icon": "📺",
            "title": f"Too Many Subscriptions ({sub_count})",
            "detail": "You have multiple streaming/music subscriptions. Cancel the least-used one.",
            "priority": "medium",
            "savings_estimate": round(subs["Amount"].min()) if len(subs) > 0 else 0,
        })

    # Travel
    travel = cat_totals.get("Travel", 0)
    if travel / max(total_spend, 1) > 0.2:
        recs.append({
            "category": "Travel",
            "icon": "🚗",
            "title": "Optimise Commute Costs",
            "detail": f"Transport at ₹{travel:,.0f} ({travel/total_spend*100:.0f}%). "
                      "Consider a monthly pass or carpooling.",
            "priority": "medium",
            "savings_estimate": round(travel * 0.2),
        })

    # EMI burden
    emi = cat_totals.get("EMI", 0)
    if income > 0 and emi / income > 0.4:
        recs.append({
            "category": "EMI",
            "icon": "🏦",
            "title": "High EMI Burden",
            "detail": f"EMIs consume {emi/income*100:.0f}% of income. "
                      "Ideal is <30%. Consider prepaying one loan.",
            "priority": "high",
            "savings_estimate": 0,
        })

    # Medical wellness
    medical = cat_totals.get("Medical", 0)
    if medical == 0:
        recs.append({
            "category": "Medical",
            "icon": "💊",
            "title": "Invest in Preventive Health",
            "detail": "No medical spend detected. Schedule an annual health checkup.",
            "priority": "low",
            "savings_estimate": 0,
        })

    # Education
    edu = cat_totals.get("Education", 0)
    if edu / max(total_spend, 1) < 0.02:
        recs.append({
            "category": "Education",
            "icon": "📚",
            "title": "Invest in Skills",
            "detail": "Education spend is low. Allocate 2–5% of income toward learning.",
            "priority": "low",
            "savings_estimate": 0,
        })

    # Savings target
    if income > 0:
        savings = income - monthly_avg
        target_savings = income * 0.2
        if savings < target_savings:
            recs.append({
                "category": "Savings",
                "icon": "💰",
                "title": "Boost Your Savings Rate",
                "detail": f"You're saving ₹{savings:,.0f}/month. Target is ₹{target_savings:,.0f} (20% of income). "
                          f"Try automating ₹{target_savings-savings:,.0f} more via SIP.",
                "priority": "high",
                "savings_estimate": round(target_savings - max(savings, 0)),
            })

    # Budget recommendations per category
    budgets = _suggest_budgets(cat_totals, monthly_avg, income)

    return recs, budgets


def _suggest_budgets(cat_totals: dict, monthly_avg: float, income: float) -> dict:
    """Return suggested monthly budget per category."""
    ideal_pct = {
        "Food": 0.20,
        "Shopping": 0.15,
        "Travel": 0.12,
        "Bills": 0.15,
        "Entertainment": 0.08,
        "Medical": 0.05,
        "Education": 0.05,
        "EMI": 0.25,
        "Other": 0.05,
    }
    base = income if income > 0 else monthly_avg
    budgets = {}
    for cat, pct in ideal_pct.items():
        actual = cat_totals.get(cat, 0)
        suggested = round(base * pct)
        budgets[cat] = {
            "suggested": suggested,
            "actual": round(actual),
            "status": "ok" if actual <= suggested else "over",
        }
    return budgets


# ─── Smart Tips ───────────────────────────────────────────────────────────────

SMART_TIPS = [
    "💡 Set up a 50/30/20 budget: 50% needs, 30% wants, 20% savings.",
    "💡 Automate your savings by setting up a SIP on the 1st of every month.",
    "💡 Before any purchase >₹2,000, wait 48 hours to avoid impulse buys.",
    "💡 Use cashback credit cards for recurring bills to earn while you spend.",
    "💡 Track every expense daily — even ₹10 coffee adds up to ₹3,650/year.",
    "💡 Increase investments by 10% each year as your salary grows.",
    "💡 Build an emergency fund of 6 months' expenses before investing.",
    "💡 Compare prices across platforms before shopping online.",
    "💡 Cook at home at least 3 meals per week to cut food expenses by 30%.",
    "💡 Cancel subscriptions you haven't used in 30 days.",
]


def get_smart_tips(df: pd.DataFrame) -> list[str]:
    """Return contextually relevant smart tips."""
    selected = []
    cat_totals = df.groupby("Category")["Amount"].sum()

    if cat_totals.get("Food", 0) > 3000:
        selected.append(SMART_TIPS[8])
    if cat_totals.get("Entertainment", 0) > 1500:
        selected.append(SMART_TIPS[9])
    if cat_totals.get("Shopping", 0) > 4000:
        selected.append(SMART_TIPS[2])

    # Fill remaining with general tips
    for tip in SMART_TIPS:
        if tip not in selected:
            selected.append(tip)
        if len(selected) >= 5:
            break

    return selected[:5]


# ─── LLM-Ready Advisor Prompt Builder ────────────────────────────────────────

def build_llm_prompt(df: pd.DataFrame, health_score: dict, income: float = 0) -> str:
    """
    Build a prompt string ready to send to any LLM API (GPT, Claude, Gemini, etc.)
    for generating personalised financial advice.
    """
    cat_totals = df.groupby("Category")["Amount"].sum().to_dict()
    monthly = df.groupby(df["Date"].dt.to_period("M"))["Amount"].sum()

    summary = "\n".join(
        f"  - {cat}: ₹{amt:,.0f}" for cat, amt in sorted(cat_totals.items(), key=lambda x: -x[1])
    )

    months_summary = "\n".join(
        f"  - {str(m)}: ₹{v:,.0f}" for m, v in monthly.items()
    )

    prompt = f"""You are SpendLens AI, an expert personal finance advisor for Indian consumers.

Analyse the following expense data and provide 5 specific, actionable financial recommendations.
Be conversational, empathetic, and specific to Indian financial context (use ₹, mention SIP, FD, etc.)

EXPENSE SUMMARY:
{summary}

MONTHLY TREND:
{months_summary}

FINANCIAL HEALTH SCORE: {health_score['score']}/100 ({health_score['label']})
MONTHLY INCOME (if provided): ₹{income:,.0f}

Provide:
1. Key insight about their biggest spending problem
2. Three specific actions to reduce expenses this month
3. A savings goal recommendation
4. A long-term wealth building tip specific to their situation

Keep your response friendly, motivating, and under 300 words.
"""
    return prompt
