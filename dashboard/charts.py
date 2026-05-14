"""
dashboard/charts.py — Plotly chart generators for SpendLens AI dashboard
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Consistent colour palette
PALETTE = [
    "#6C63FF", "#FF6584", "#43D9A2", "#FFB347", "#4FC3F7",
    "#FF8A65", "#A5D6A7", "#CE93D8", "#80DEEA", "#FFCC02",
]
BG = "rgba(0,0,0,0)"
FONT = dict(family="IBM Plex Sans, sans-serif", color="#E2E8F0")


def _layout(title="", **kwargs):
    return dict(
        title=dict(text=title, font=dict(size=16, color="#E2E8F0")),
        paper_bgcolor=BG,
        plot_bgcolor="rgba(15,23,42,0.6)",
        font=FONT,
        margin=dict(l=40, r=20, t=50, b=40),
        **kwargs,
    )


def monthly_trend_chart(df: pd.DataFrame) -> go.Figure:
    monthly = df.groupby(df["Date"].dt.to_period("M"))["Amount"].sum().reset_index()
    monthly["Date"] = monthly["Date"].astype(str)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monthly["Date"], y=monthly["Amount"],
        marker_color=PALETTE[0], opacity=0.85,
        name="Monthly Spend",
    ))
    fig.add_trace(go.Scatter(
        x=monthly["Date"], y=monthly["Amount"],
        mode="lines+markers",
        line=dict(color=PALETTE[2], width=2.5),
        marker=dict(size=8, color=PALETTE[2]),
        name="Trend",
    ))
    fig.update_layout(**_layout("Monthly Expense Trend"))
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
    return fig


def daily_spending_chart(df: pd.DataFrame) -> go.Figure:
    daily = df.groupby("Date")["Amount"].sum().reset_index()
    avg = daily["Amount"].mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["Date"], y=daily["Amount"],
        fill="tozeroy",
        fillcolor="rgba(108,99,255,0.2)",
        line=dict(color=PALETTE[0], width=2),
        name="Daily Spend",
    ))
    fig.add_hline(
        y=avg, line_dash="dash",
        line_color=PALETTE[1],
        annotation_text=f"Avg ₹{avg:,.0f}",
        annotation_font_color=PALETTE[1],
    )
    fig.update_layout(**_layout("Daily Spending"))
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
    return fig


def category_pie_chart(df: pd.DataFrame) -> go.Figure:
    cat = df.groupby("Category")["Amount"].sum().reset_index()
    fig = go.Figure(go.Pie(
        labels=cat["Category"],
        values=cat["Amount"],
        hole=0.55,
        marker=dict(colors=PALETTE),
        textinfo="label+percent",
        textfont=dict(size=12),
        hovertemplate="<b>%{label}</b><br>₹%{value:,.0f}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(**_layout("Spending by Category"))
    return fig


def top_categories_bar(df: pd.DataFrame) -> go.Figure:
    cat = df.groupby("Category")["Amount"].sum().sort_values(ascending=True).reset_index()
    fig = go.Figure(go.Bar(
        x=cat["Amount"],
        y=cat["Category"],
        orientation="h",
        marker=dict(
            color=PALETTE[:len(cat)],
            line=dict(width=0),
        ),
        text=[f"₹{v:,.0f}" for v in cat["Amount"]],
        textposition="outside",
        textfont=dict(color="#E2E8F0"),
    ))
    fig.update_layout(**_layout("Top Spending Categories"))
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    return fig


def spending_heatmap(df: pd.DataFrame) -> go.Figure:
    df2 = df.copy()
    df2["DayOfWeek"] = df2["Date"].dt.day_name()
    df2["MonthName"] = df2["Date"].dt.strftime("%b %Y")
    pivot = df2.pivot_table(
        values="Amount", index="DayOfWeek", columns="MonthName", aggfunc="sum", fill_value=0
    )
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = pivot.reindex([d for d in day_order if d in pivot.index])

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale="Viridis",
        hovertemplate="Day: %{y}<br>Month: %{x}<br>Amount: ₹%{z:,.0f}<extra></extra>",
    ))
    fig.update_layout(**_layout("Spending Heatmap (Day × Month)"))
    return fig


def forecast_chart(historical_dates, historical_values,
                   future_dates, future_preds,
                   model_name: str = "Forecast") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=historical_dates, y=historical_values,
        mode="lines",
        line=dict(color=PALETTE[2], width=2),
        name="Historical",
    ))
    fig.add_trace(go.Scatter(
        x=future_dates, y=future_preds,
        mode="lines+markers",
        line=dict(color=PALETTE[1], width=2.5, dash="dot"),
        marker=dict(size=5, color=PALETTE[1]),
        name=f"Forecast ({model_name})",
    ))
    # Confidence band (±15%)
    upper = [v * 1.15 for v in future_preds]
    lower = [max(0, v * 0.85) for v in future_preds]
    fig.add_trace(go.Scatter(
        x=future_dates + future_dates[::-1],
        y=upper + lower[::-1],
        fill="toself",
        fillcolor="rgba(255,101,132,0.1)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Confidence Band",
        showlegend=True,
    ))
    fig.update_layout(**_layout(f"Expense Forecast — {model_name}"))
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
    return fig


def category_forecast_chart(cat_forecast: dict) -> go.Figure:
    cats = list(cat_forecast.keys())
    preds = [cat_forecast[c]["predicted"] for c in cats]
    trends = [cat_forecast[c]["trend"] for c in cats]
    colors = [PALETTE[2] if t == "down" else PALETTE[1] for t in trends]

    fig = go.Figure(go.Bar(
        x=cats, y=preds,
        marker_color=colors,
        text=[f"₹{v:,.0f}" for v in preds],
        textposition="outside",
        textfont=dict(color="#E2E8F0"),
    ))
    fig.update_layout(**_layout("Predicted Next Month — Category Breakdown"))
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
    return fig


def health_score_gauge(score: float, grade: str) -> go.Figure:
    color = (
        "#43D9A2" if score >= 75 else
        "#FFB347" if score >= 50 else
        "#FF6584"
    )
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "/100", "font": {"size": 36, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#E2E8F0"},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "rgba(15,23,42,0.8)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "rgba(255,101,132,0.15)"},
                {"range": [40, 70], "color": "rgba(255,179,71,0.15)"},
                {"range": [70, 100], "color": "rgba(67,217,162,0.15)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.75,
                "value": score,
            },
        },
        title={"text": f"Financial Health Score — Grade {grade}", "font": {"color": "#E2E8F0", "size": 14}},
    ))
    fig.update_layout(paper_bgcolor=BG, height=280, margin=dict(l=20, r=20, t=40, b=10))
    return fig


def confusion_matrix_chart(cm: list, classes: list) -> go.Figure:
    cm_arr = np.array(cm)
    fig = go.Figure(go.Heatmap(
        z=cm_arr,
        x=classes,
        y=classes,
        colorscale="Blues",
        text=cm_arr,
        texttemplate="%{text}",
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
    ))
    fig.update_layout(**_layout("Confusion Matrix"))
    return fig


def budget_comparison_chart(budgets: dict) -> go.Figure:
    cats = list(budgets.keys())
    actual = [budgets[c]["actual"] for c in cats]
    suggested = [budgets[c]["suggested"] for c in cats]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Actual", x=cats, y=actual,
        marker_color=PALETTE[1], opacity=0.9,
        text=[f"₹{v:,.0f}" for v in actual],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Suggested Budget", x=cats, y=suggested,
        marker_color=PALETTE[0], opacity=0.6,
        text=[f"₹{v:,.0f}" for v in suggested],
        textposition="outside",
    ))
    fig.update_layout(
        **_layout("Actual vs Suggested Budget"),
        barmode="group",
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
    return fig
