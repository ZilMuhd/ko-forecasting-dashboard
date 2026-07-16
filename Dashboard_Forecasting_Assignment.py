import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Coca-Cola | Demand Forecast Dashboard",
    page_icon="🥤",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "KO_CocaCola_Stock_Prices_1980_2026.csv"

# ----------------------------------------------------------------------------
# BRAND THEME / CUSTOM CSS
# ----------------------------------------------------------------------------
COKE_RED = "#E4110C"
COKE_RED_DARK = "#B40000"
COKE_RED_TINT = "#FDEBEA"
INK = "#1A1A1A"
MUTED = "#6B6B6B"
SURFACE = "#FFFFFF"
BORDER = "#EAE3E1"
CREAM = "#FFFBF6"
GREEN = "#1B7A4C"
GREEN_BG = "#E7F5EC"
RED_BG = "#FBEAE9"
GOLD = "#B8860B"
GOLD_BG = "#FBF3E3"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Pacifico&family=Montserrat:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Montserrat', -apple-system, sans-serif;
    }}

    /* Force a light background regardless of the visitor's OS/browser dark-mode
       setting -- Streamlit's own dark theme sits above plain html/body CSS,
       so these container-level selectors are needed too. */
    [data-testid="stAppViewContainer"], [data-testid="stHeader"], .stApp {{
        background-color: {CREAM} !important;
    }}
    [data-testid="stSidebar"] {{
        background-color: {SURFACE} !important;
        border-right: 1px solid {BORDER};
    }}
    [data-testid="stSidebar"] * {{
        color: {INK} !important;
    }}
    .stMarkdown, .stMarkdown p, label, .stSlider label, .stSelectbox label {{
        color: {INK};
    }}

    /* Native widgets that otherwise stay dark even with the theme forced */
    .stButton button, .stDownloadButton button {{
        background-color: {SURFACE} !important;
        color: {INK} !important;
        border: 1px solid {BORDER} !important;
    }}
    div[data-baseweb="select"] > div {{
        background-color: {SURFACE} !important;
        color: {INK} !important;
    }}
    div[data-baseweb="popover"] {{
        background-color: {SURFACE} !important;
    }}
    li[role="option"] {{
        background-color: {SURFACE} !important;
        color: {INK} !important;
    }}
    [data-testid="stExpander"] {{
        background-color: {SURFACE} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 10px;
    }}
    [data-testid="stExpander"] summary {{
        color: {INK} !important;
    }}
    [data-testid="stDataFrame"] {{
        background-color: {SURFACE} !important;
    }}
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"], [data-testid="stMetricDelta"] {{
        color: {INK} !important;
    }}
    [data-testid="stFileUploaderDropzone"] {{
        background-color: {SURFACE} !important;
        color: {INK} !important;
    }}

    .block-container {{
        padding-top: 1.2rem;
        padding-bottom: 3rem;
        max-width: 1300px;
    }}

    /* ---------- Brand header ---------- */
    .coke-header {{
        background: linear-gradient(120deg, {COKE_RED} 0%, {COKE_RED_DARK} 100%);
        border-radius: 18px;
        padding: 1.6rem 2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 6px 18px rgba(180,0,0,0.18);
        margin-bottom: 1.8rem;
    }}
    .coke-script {{
        font-family: 'Pacifico', cursive;
        font-size: 2.5rem;
        color: #FFFFFF;
        margin: 0;
        line-height: 1;
    }}
    .coke-subtitle {{
        color: #FFE9E7;
        font-size: 0.95rem;
        margin-top: 0.4rem;
        font-weight: 500;
    }}
    .coke-pill {{
        background: rgba(255,255,255,0.16);
        border: 1px solid rgba(255,255,255,0.35);
        color: #FFFFFF;
        font-weight: 700;
        font-size: 0.78rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        padding: 0.45rem 1rem;
        border-radius: 999px;
        white-space: nowrap;
    }}

    /* ---------- Section headers ---------- */
    .section-title {{
        font-size: 1.15rem;
        font-weight: 700;
        color: {INK};
        margin: 1.8rem 0 0.4rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    .section-caption {{
        color: {MUTED};
        font-size: 0.9rem;
        margin-bottom: 0.9rem;
    }}

    /* ---------- KPI cards ---------- */
    .kpi-card {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-top: 3px solid {COKE_RED};
        border-radius: 14px;
        padding: 1.1rem 1.2rem;
        box-shadow: 0 1px 3px rgba(16,24,40,0.05);
        height: 100%;
    }}
    .kpi-label {{
        color: {MUTED};
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-bottom: 0.35rem;
    }}
    .kpi-value {{
        color: {INK};
        font-size: 1.65rem;
        font-weight: 800;
        letter-spacing: -0.01em;
    }}
    .kpi-help {{
        color: {MUTED};
        font-size: 0.78rem;
        margin-top: 0.3rem;
    }}

    /* ---------- Badges ---------- */
    .badge {{
        display: inline-block;
        padding: 0.3rem 0.75rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.85rem;
    }}
    .badge-up {{ background: {GREEN_BG}; color: {GREEN}; }}
    .badge-down {{ background: {RED_BG}; color: {COKE_RED_DARK}; }}
    .badge-stable {{ background: {GOLD_BG}; color: {GOLD}; }}
    .badge-best {{ background: {GREEN_BG}; color: {GREEN}; }}

    /* ---------- Panels ---------- */
    .summary-panel {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-left: 4px solid {COKE_RED};
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-top: 0.8rem;
        color: {INK};
        font-size: 0.95rem;
        line-height: 1.55;
    }}
    .risk-panel {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-left: 4px solid {GOLD};
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0 1rem 0;
        color: {INK};
        font-size: 0.95rem;
        line-height: 1.55;
    }}
    .accuracy-panel {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-left: 4px solid {GREEN};
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0 1rem 0;
        color: {INK};
        font-size: 0.95rem;
        line-height: 1.55;
    }}
    .action-item {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 0.6rem 0.9rem;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        color: {INK};
    }}

    [data-testid="stMetricValue"] {{
        font-size: 1.4rem;
    }}

    footer {{visibility: hidden;}}
    </style>
    """,
    unsafe_allow_html=True,
)


def kpi_card(label: str, value: str, help_text: str = "") -> str:
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-help">{help_text}</div>
    </div>
    """


# ----------------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Loads the bundled Coca-Cola volume CSV."""
    if not DEFAULT_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find '{DEFAULT_DATA_PATH.name}'. Make sure it sits in the "
            "same folder as this script."
        )
    df = pd.read_csv(DEFAULT_DATA_PATH, parse_dates=["Date"])

    df = df.sort_values("Date").copy()
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    df = df.set_index("Date")

    full_index = pd.date_range(df.index.min(), df.index.max(), freq="B")
    df = df.reindex(full_index)
    df["Volume"] = df["Volume"].interpolate(method="time")
    df = df.loc[df.index >= "2023-01-01"].copy()
    df["log_Volume"] = np.log(df["Volume"])
    return df


# ----------------------------------------------------------------------------
# MODEL FUNCTIONS
# ----------------------------------------------------------------------------
def _fit_arima(train: pd.Series):
    try:
        return ARIMA(train, order=(1, 1, 2), trend=None).fit()
    except Exception:
        return ARIMA(train, order=(1, 0, 1), trend=None).fit()


@st.cache_data(show_spinner=False)
def build_forecast(horizon: int, confidence_level: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = load_data()
    if len(data) < horizon + 20:
        raise ValueError("Not enough historical data for the selected horizon.")

    train = data["log_Volume"].dropna().iloc[:-horizon]
    if len(train) < 20:
        raise ValueError("Not enough training data for ARIMA fitting.")

    fitted = _fit_arima(train)
    forecast_obj = fitted.get_forecast(steps=horizon)
    pred_mean = forecast_obj.predicted_mean
    conf_int = forecast_obj.conf_int(alpha=1 - confidence_level / 100)

    forecast_df = pd.DataFrame(
        {
            "Date": pred_mean.index,
            "Forecast": np.exp(pred_mean).astype(float),
            "Lower": np.maximum(np.exp(conf_int.iloc[:, 0]).astype(float), 0),
            "Upper": np.maximum(np.exp(conf_int.iloc[:, 1]).astype(float), 0),
        }
    ).set_index("Date")

    history = data[["Volume"]].copy()
    history = history[history["Volume"].notna()].copy()
    history.index.name = "Date"

    return forecast_df, history


@st.cache_data(show_spinner=False)
def get_recommendation(forecast_df: pd.DataFrame, history: pd.DataFrame, horizon: int) -> dict:
    recent_avg = history["Volume"].tail(20).mean()
    next_avg = forecast_df["Forecast"].mean()
    band_width = (forecast_df["Upper"] - forecast_df["Lower"]).mean()
    end_period = forecast_df["Forecast"].iloc[-1]
    total_forecast = forecast_df["Forecast"].sum()
    growth_pct = (next_avg / recent_avg - 1) * 100 if recent_avg else 0

    if next_avg > recent_avg * 1.08:
        headline = "Upside demand signal"
        badge = "badge-up"
        rec = (
            "The forecast points to stronger-than-usual trading volume, used here as a proxy "
            "for market demand intensity, over the next planning window. Management should "
            "prepare for higher demand by increasing promotional support, aligning supply chain "
            "and distribution capacity, and ensuring inventory coverage across key markets."
        )
        actions = [
            "Increase promotional support where demand is likely to be strongest.",
            "Confirm supply chain and distribution readiness for a busier period.",
            "Protect inventory and service levels with a small buffer.",
        ]
    elif next_avg < recent_avg * 0.92:
        headline = "Downside demand signal"
        badge = "badge-down"
        rec = (
            "The forecast suggests softer volume than recent levels. Management should keep "
            "spend disciplined, preserve cash flexibility, and avoid overcommitting inventory "
            "or production capacity."
        )
        actions = [
            "Hold discretionary spend and promotional intensity steady.",
            "Avoid overcommitting production and inventory beyond the near-term plan.",
            "Monitor execution closely and keep contingency funds available.",
        ]
    else:
        headline = "Stable demand outlook"
        badge = "badge-stable"
        rec = (
            "The outlook is broadly stable. Maintain the current operating plan, watch the "
            "next few weeks closely, and only adjust if the confidence band widens."
        )
        actions = [
            "Keep the current operating plan in place.",
            "Monitor the next 2-3 weeks for early signs of change.",
            "Use the confidence band to trigger action only if volatility rises.",
        ]

    if band_width > recent_avg * 0.25:
        rec += " The uncertainty band is wide, so a conservative buffer plan is advisable."
    if end_period > recent_avg * 1.05:
        rec += " The tail-end of the forecast also remains above the recent average, supporting a proactive planning stance."
    if total_forecast > recent_avg * horizon * 1.02:
        rec += " Overall cumulative volume is expected to be above the recent run-rate."

    return {
        "headline": headline,
        "badge": badge,
        "summary": rec,
        "actions": actions,
        "growth_pct": growth_pct,
    }


@st.cache_data(show_spinner=False)
def build_monthly_summary(forecast_df: pd.DataFrame) -> pd.DataFrame:
    monthly = forecast_df.reset_index().copy()
    monthly["Month"] = monthly["Date"].dt.to_period("M").astype(str)
    monthly_summary = monthly.groupby("Month", as_index=False)["Forecast"].sum()
    monthly_summary = monthly_summary.rename(columns={"Forecast": "Forecasted Volume"})
    return monthly_summary


@st.cache_data(show_spinner=False)
def run_simulation(horizon: int, n_sims: int = 1000, threshold_pct: float = 90) -> dict:
    data = load_data()
    train = data["log_Volume"].dropna()
    fitted = _fit_arima(train)

    sim_results = fitted.simulate(nsimulations=horizon, repetitions=n_sims, anchor="end")
    sim_paths = np.exp(sim_results.values.reshape(horizon, n_sims))

    sim_median = np.median(sim_paths, axis=1)
    sim_p5 = np.percentile(sim_paths, 5, axis=1)
    sim_p95 = np.percentile(sim_paths, 95, axis=1)

    threshold = np.percentile(data["Volume"].dropna(), threshold_pct)
    prob_exceed = (sim_paths > threshold).mean() * 100

    return {
        "sim_paths": sim_paths,
        "median": sim_median,
        "p5": sim_p5,
        "p95": sim_p95,
        "threshold": threshold,
        "prob_exceed": prob_exceed,
    }


@st.cache_data(show_spinner=False)
def backtest_accuracy(test_horizon: int) -> dict:
    """Holds out the last `test_horizon` days and compares ARIMA against two
    simple benchmark methods, reporting MAPE and RMSE for each — the
    standard forecast accuracy evaluation expected in a forecasting
    management exercise."""
    data = load_data()
    series = data["log_Volume"].dropna()

    if len(series) < test_horizon + 30:
        raise ValueError("Not enough history to run a reliable backtest.")

    train = series.iloc[:-test_horizon]
    test = series.iloc[-test_horizon:]
    actual = np.exp(test)

    # 1. ARIMA
    fitted = _fit_arima(train)
    arima_pred = pd.Series(np.exp(fitted.forecast(steps=test_horizon)).values, index=test.index)

    # 2. Naive (random walk: repeat last observed value)
    last_val = np.exp(train.iloc[-1])
    naive_pred = pd.Series(last_val, index=test.index)

    # 3. Simple moving average (10-day trailing average, held flat)
    ma_val = np.exp(train.tail(10)).mean()
    ma_pred = pd.Series(ma_val, index=test.index)

    def mape(a, p):
        return float(np.mean(np.abs((a - p) / a)) * 100)

    def rmse(a, p):
        return float(np.sqrt(np.mean((a - p) ** 2)))

    results = pd.DataFrame(
        {
            "Model": ["ARIMA(1,1,2)", "Naive (random walk)", "Moving average (10-day)"],
            "MAPE (%)": [mape(actual, arima_pred), mape(actual, naive_pred), mape(actual, ma_pred)],
            "RMSE": [rmse(actual, arima_pred), rmse(actual, naive_pred), rmse(actual, ma_pred)],
        }
    )

    return {
        "results": results,
        "actual": actual,
        "arima_pred": arima_pred,
        "naive_pred": naive_pred,
        "ma_pred": ma_pred,
    }


# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🥤 Coca-Cola Forecast")
    st.caption(f"Data source: `{DEFAULT_DATA_PATH.name}`")
    st.markdown("---")
    st.markdown("### ⚙️ Forecast controls")
    horizon = st.slider("Forecast horizon (trading days)", min_value=5, max_value=90, value=60)
    confidence_level = st.selectbox("Confidence level", [80, 90, 95], index=2)
    threshold_pct = st.slider("Risk threshold (historical percentile)", 75, 99, 90)
    backtest_horizon = st.slider("Accuracy backtest window (days)", 10, 60, 20)

    st.markdown("---")
    st.caption(
        "Adjust the planning horizon, confidence band, backtest window, and the volume "
        "percentile used to flag elevated demand risk."
    )
    st.markdown("---")
    st.caption("Model: ARIMA(1,1,2) on log-transformed daily trading volume, used as a demand proxy for Coca-Cola (KO).")

# ----------------------------------------------------------------------------
# DATA / MODEL RUN
# ----------------------------------------------------------------------------
try:
    forecast_df, history = build_forecast(horizon, confidence_level)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except ValueError as e:
    st.error(str(e))
    st.stop()

forecast_df["Cumulative"] = forecast_df["Forecast"].cumsum()
summary = get_recommendation(forecast_df, history, horizon)
monthly_summary = build_monthly_summary(forecast_df)
sim = run_simulation(horizon, threshold_pct=threshold_pct)

try:
    backtest = backtest_accuracy(backtest_horizon)
    backtest_ok = True
except ValueError as e:
    backtest_ok = False
    backtest_error = str(e)

# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="coke-header">
        <div>
            <p class="coke-script">Coca-Cola</p>
            <p class="coke-subtitle">Demand Forecast &amp; Executive Management Dashboard — Trading volume as a demand proxy</p>
        </div>
        <div class="coke-pill">{horizon}-Day Outlook</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# KPI ROW
# ----------------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(
        kpi_card(
            "Next-day forecast",
            f"{forecast_df.iloc[0]['Forecast']:,.0f}",
            "Expected volume, next trading day",
        ),
        unsafe_allow_html=True,
    )
with k2:
    st.markdown(
        kpi_card(
            "Avg. next 7 days",
            f"{forecast_df.head(7)['Forecast'].mean():,.0f}",
            "Average expected weekly volume",
        ),
        unsafe_allow_html=True,
    )
with k3:
    st.markdown(
        kpi_card(
            f"Total, {horizon}-day horizon",
            f"{forecast_df['Forecast'].sum():,.0f}",
            "Cumulative expected volume",
        ),
        unsafe_allow_html=True,
    )
with k4:
    rel_unc = (forecast_df["Upper"] - forecast_df["Lower"]).mean() / max(forecast_df["Forecast"].mean(), 1)
    st.markdown(
        kpi_card(
            "Relative uncertainty",
            f"{rel_unc:.1%}",
            f"Avg. {confidence_level}% CI width vs forecast",
        ),
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------------
# EXECUTIVE SUMMARY
# ----------------------------------------------------------------------------
st.markdown('<div class="section-title">📋 Executive summary</div>', unsafe_allow_html=True)
st.markdown(f"<span class='badge {summary['badge']}'>{summary['headline']}</span>", unsafe_allow_html=True)
st.markdown(f"<div class='summary-panel'>{summary['summary']}</div>", unsafe_allow_html=True)

c1, c2 = st.columns([2, 1])
with c1:
    st.markdown("**Recommended management actions**")
    for action in summary["actions"]:
        st.markdown(f"<div class='action-item'>✅ {action}</div>", unsafe_allow_html=True)
with c2:
    st.metric("Expected change vs recent average", f"{summary['growth_pct']:+.1f}%")
    st.caption("Positive values suggest a stronger near-term outlook than recent levels.")

# ----------------------------------------------------------------------------
# FORECAST CHART
# ----------------------------------------------------------------------------
st.markdown('<div class="section-title">📈 Forecast vs. historical volume</div>', unsafe_allow_html=True)
st.markdown(
    f"<div class='section-caption'>ARIMA(1,1,2) point forecast with {confidence_level}% confidence interval</div>",
    unsafe_allow_html=True,
)

fig1 = go.Figure()
recent_history = history.tail(120)
fig1.add_trace(go.Scatter(
    x=recent_history.index, y=recent_history["Volume"],
    mode="lines", name="Historical volume",
    line=dict(color=INK, width=1.6),
))
fig1.add_trace(go.Scatter(
    x=forecast_df.index, y=forecast_df["Upper"],
    mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
))
fig1.add_trace(go.Scatter(
    x=forecast_df.index, y=forecast_df["Lower"],
    mode="lines", line=dict(width=0), fill="tonexty",
    fillcolor="rgba(228,17,12,0.12)", name=f"{confidence_level}% confidence interval",
))
fig1.add_trace(go.Scatter(
    x=forecast_df.index, y=forecast_df["Forecast"],
    mode="lines", name="Forecast",
    line=dict(color=COKE_RED, width=2.6),
))
fig1.add_vline(x=history.index[-1], line_dash="dash", line_color=MUTED, line_width=1)
fig1.update_layout(
    template="plotly_white",
    height=430,
    margin=dict(l=10, r=10, t=10, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    xaxis_title=None,
    yaxis_title="Volume",
    hovermode="x unified",
)
st.plotly_chart(fig1, use_container_width=True, theme=None)

# ----------------------------------------------------------------------------
# MODEL ACCURACY EVALUATION (NEW)
# ----------------------------------------------------------------------------
st.markdown('<div class="section-title">🎓 Model accuracy evaluation</div>', unsafe_allow_html=True)
st.markdown(
    f"<div class='section-caption'>Backtest over the last {backtest_horizon} trading days — ARIMA compared "
    "against a naive (random walk) and a moving-average benchmark</div>",
    unsafe_allow_html=True,
)

if not backtest_ok:
    st.warning(backtest_error)
else:
    results = backtest["results"].copy()
    best_model = results.loc[results["MAPE (%)"].idxmin(), "Model"]

    a1, a2, a3 = st.columns(3)
    for col, (_, row) in zip([a1, a2, a3], results.iterrows()):
        is_best = row["Model"] == best_model
        badge_html = "<span class='badge badge-best'>Best fit</span>" if is_best else ""
        with col:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">{row['Model']}</div>
                    <div class="kpi-value">{row['MAPE (%)']:.1f}% <span style="font-size:0.9rem; color:{MUTED};">MAPE</span></div>
                    <div class="kpi-help">RMSE: {row['RMSE']:,.0f} &nbsp; {badge_html}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        f"""
        <div class="accuracy-panel">
        Across the {backtest_horizon}-day holdout window, <b>{best_model}</b> produced the lowest MAPE,
        making it the preferred forecasting method for this dataset. Lower MAPE and RMSE indicate the
        model's predictions stayed closer to actual observed volume. Benchmarking against naive and
        moving-average methods confirms whether the added complexity of ARIMA is actually justified —
        a core check in any forecasting management framework.
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(
        x=backtest["actual"].index, y=backtest["actual"].values,
        mode="lines+markers", name="Actual volume", line=dict(color=INK, width=2),
    ))
    fig_acc.add_trace(go.Scatter(
        x=backtest["arima_pred"].index, y=backtest["arima_pred"].values,
        mode="lines", name="ARIMA(1,1,2)", line=dict(color=COKE_RED, width=2.2),
    ))
    fig_acc.add_trace(go.Scatter(
        x=backtest["naive_pred"].index, y=backtest["naive_pred"].values,
        mode="lines", name="Naive", line=dict(color=GOLD, width=1.6, dash="dash"),
    ))
    fig_acc.add_trace(go.Scatter(
        x=backtest["ma_pred"].index, y=backtest["ma_pred"].values,
        mode="lines", name="Moving average (10-day)", line=dict(color=GREEN, width=1.6, dash="dot"),
    ))
    fig_acc.update_layout(
        template="plotly_white",
        height=380,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        yaxis_title="Volume",
        hovermode="x unified",
    )
    st.plotly_chart(fig_acc, use_container_width=True, theme=None)

    with st.expander("View full accuracy table"):
        st.dataframe(results.round(2), use_container_width=True, hide_index=True)

# ----------------------------------------------------------------------------
# RISK SCENARIO ANALYSIS (Monte Carlo)
# ----------------------------------------------------------------------------
st.markdown('<div class="section-title">🎯 Risk scenario analysis</div>', unsafe_allow_html=True)
st.markdown(
    "<div class='section-caption'>Monte Carlo simulation — 1,000 ARIMA-based paths, "
    "capturing realistic volatility beyond a single point forecast</div>",
    unsafe_allow_html=True,
)

rk1, rk2, rk3 = st.columns(3)
with rk1:
    st.markdown(
        kpi_card("Downside risk (5th %ile)", f"{sim['p5'][-1]:,.0f}", f"End of {horizon}-day horizon"),
        unsafe_allow_html=True,
    )
with rk2:
    st.markdown(
        kpi_card("Upside scenario (95th %ile)", f"{sim['p95'][-1]:,.0f}", f"End of {horizon}-day horizon"),
        unsafe_allow_html=True,
    )
with rk3:
    st.markdown(
        kpi_card(
            f"P(volume > {sim['threshold']:,.0f})",
            f"{sim['prob_exceed']:.1f}%",
            f"On any day, based on {threshold_pct}th historical percentile",
        ),
        unsafe_allow_html=True,
    )

st.markdown(
    f"""
    <div class="risk-panel">
    Based on 1,000 simulated paths, there is a <b>{sim['prob_exceed']:.1f}% chance</b> that trading
    volume exceeds <b>{sim['threshold']:,.0f}</b> shares (the {threshold_pct}th historical percentile)
    on any given day within the next {horizon} trading days. Forecast uncertainty widens with horizon
    length — near-term estimates (day 1: {sim['p5'][0]:,.0f} to {sim['p95'][0]:,.0f}) are notably
    tighter than longer-horizon estimates (day {horizon}: {sim['p5'][-1]:,.0f} to {sim['p95'][-1]:,.0f}).
    </div>
    """,
    unsafe_allow_html=True,
)

future_dates = pd.bdate_range(start=history.index[-1] + pd.Timedelta(days=1), periods=horizon)
n_sample = min(120, sim["sim_paths"].shape[1])
sample_paths = sim["sim_paths"][:, np.random.choice(sim["sim_paths"].shape[1], size=n_sample, replace=False)]

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=history.tail(60).index, y=history.tail(60)["Volume"],
    mode="lines", name="Historical volume", line=dict(color=INK, width=1.6),
))
for i in range(sample_paths.shape[1]):
    fig2.add_trace(go.Scatter(
        x=future_dates, y=sample_paths[:, i],
        mode="lines", line=dict(color="rgba(228,17,12,0.05)", width=1),
        showlegend=False, hoverinfo="skip",
    ))
fig2.add_trace(go.Scatter(
    x=future_dates, y=sim["p95"],
    mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
))
fig2.add_trace(go.Scatter(
    x=future_dates, y=sim["p5"],
    mode="lines", line=dict(width=0), fill="tonexty",
    fillcolor="rgba(228,17,12,0.15)", name="5th-95th percentile band",
))
fig2.add_trace(go.Scatter(
    x=future_dates, y=sim["median"],
    mode="lines", name="Median simulated path", line=dict(color=COKE_RED_DARK, width=2.6),
))
fig2.add_hline(
    y=sim["threshold"], line_dash="dash", line_color=GOLD, line_width=1.4,
    annotation_text=f"{threshold_pct}th percentile threshold", annotation_position="top left",
)
fig2.update_layout(
    template="plotly_white",
    height=440,
    margin=dict(l=10, r=10, t=10, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    yaxis_title="Volume",
    hovermode="x unified",
)
st.plotly_chart(fig2, use_container_width=True, theme=None)

# ----------------------------------------------------------------------------
# DAILY / CUMULATIVE CHARTS
# ----------------------------------------------------------------------------
st.markdown('<div class="section-title">📊 Forecast breakdown</div>', unsafe_allow_html=True)

col5, col6 = st.columns(2)
with col5:
    fig3 = go.Figure(go.Bar(
        x=forecast_df.index, y=forecast_df["Forecast"],
        marker_color=COKE_RED, opacity=0.9,
    ))
    fig3.update_layout(
        template="plotly_white", height=340,
        margin=dict(l=10, r=10, t=30, b=10),
        title=dict(text="Daily forecast", font=dict(size=14)),
        yaxis_title="Volume",
    )
    st.plotly_chart(fig3, use_container_width=True, theme=None)
with col6:
    fig4 = go.Figure(go.Scatter(
        x=forecast_df.index, y=forecast_df["Cumulative"],
        mode="lines", line=dict(color=GREEN, width=2.6), fill="tozeroy",
        fillcolor="rgba(27,122,76,0.10)",
    ))
    fig4.update_layout(
        template="plotly_white", height=340,
        margin=dict(l=10, r=10, t=30, b=10),
        title=dict(text="Cumulative outlook", font=dict(size=14)),
        yaxis_title="Cumulative volume",
    )
    st.plotly_chart(fig4, use_container_width=True, theme=None)

# ----------------------------------------------------------------------------
# MONTHLY SUMMARY + DATA TABLE
# ----------------------------------------------------------------------------
st.markdown('<div class="section-title">🗓️ Monthly planning summary</div>', unsafe_allow_html=True)
st.dataframe(monthly_summary.round(2), use_container_width=True, hide_index=True)

st.markdown('<div class="section-title">📄 Forecast data</div>', unsafe_allow_html=True)
with st.expander("View full forecast table & download"):
    st.download_button(
        label="⬇️ Download forecast table (CSV)",
        data=forecast_df.reset_index().to_csv(index=False).encode("utf-8"),
        file_name="ko_volume_forecast.csv",
        mime="text/csv",
    )
    st.dataframe(forecast_df.reset_index().round(2), use_container_width=True, hide_index=True)

st.markdown(
    f"<p style='color:{MUTED}; font-size:0.8rem; margin-top:2rem;'>"
    "Model: ARIMA(1,1,2) on log-transformed daily volume, benchmarked against naive and moving-average "
    "methods above. Risk bands derived from a 1,000-path Monte Carlo simulation. Prepared for academic "
    "coursework — for internal planning illustration only, not investment advice."
    "</p>",
    unsafe_allow_html=True,
)
