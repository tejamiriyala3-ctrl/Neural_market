"""
NEURAL//MARKET — Streamlit front-end for a 60-day-lookback LSTM price model.

Expected files next to this script:
    stockpredict.keras   (preferred; save with model.save("stockpredict.keras"))
    stockpredict.pkl     (fallback; joblib-pickled Keras model, fragile)
    scaler.pkl           (joblib-pickled MinMaxScaler fitted on df[['Close']])

Run:  streamlit run app.py
"""

import os
from datetime import date, timedelta

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

LOOKBACK = 60
MODEL_KERAS = "stockpredict.keras"
MODEL_PKL = "stockpredict.pkl"
SCALER_PKL = "scaler.pkl"

st.set_page_config(page_title="NEURAL//MARKET", page_icon="📈", layout="wide")

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=Inter:wght@400;600&display=swap');

:root {
  --bg0: #05060f; --bg1: #0b1026; --cyan: #00f0ff; --mag: #ff2bd6;
  --lime: #9dff00; --red: #ff4d6d; --text: #e6f1ff; --muted: #7d8bb3;
  --glass: rgba(255,255,255,0.04); --border: rgba(0,240,255,0.25);
}
.stApp {
  background: radial-gradient(1200px 600px at 10% -10%, #12204d 0%, transparent 60%),
              radial-gradient(900px 500px at 110% 10%, #3a0f4d 0%, transparent 55%),
              linear-gradient(180deg, var(--bg0), var(--bg1));
  color: var(--text); font-family: 'Inter', sans-serif;
}
h1, h2, h3, .orb { font-family: 'Orbitron', sans-serif !important; letter-spacing: 0.06em; }
section[data-testid="stSidebar"] {
  background: rgba(5,6,15,0.85); border-right: 1px solid var(--border);
}
.hero {
  padding: 22px 28px; border: 1px solid var(--border); border-radius: 18px;
  background: linear-gradient(135deg, rgba(0,240,255,0.08), rgba(255,43,214,0.06));
  box-shadow: 0 0 40px rgba(0,240,255,0.08) inset; margin-bottom: 18px;
}
.hero h1 { margin: 0; font-size: 2.1rem;
  background: linear-gradient(90deg, var(--cyan), var(--mag));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.hero p { margin: 6px 0 0; color: var(--muted); }
.card {
  background: var(--glass); border: 1px solid var(--border); border-radius: 14px;
  padding: 14px 16px; backdrop-filter: blur(8px);
  box-shadow: 0 0 18px rgba(0,240,255,0.06);
}
.card .label { color: var(--muted); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.12em; }
.card .value { font-family: 'Orbitron', sans-serif; font-size: 1.55rem; margin-top: 4px; }
.card .sub { font-size: 0.85rem; margin-top: 2px; }
.up { color: var(--lime); } .down { color: var(--red); } .neutral { color: var(--cyan); }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
  background: var(--glass); border: 1px solid var(--border); border-radius: 10px;
  padding: 8px 16px; color: var(--text);
}
.stTabs [aria-selected="true"] { background: rgba(0,240,255,0.12) !important; }
.stButton>button, .stDownloadButton>button {
  background: linear-gradient(90deg, rgba(0,240,255,0.15), rgba(255,43,214,0.15));
  border: 1px solid var(--border); color: var(--text); border-radius: 10px;
}
</style>
""",
    unsafe_allow_html=True,
)


def card(label: str, value: str, sub: str = "", tone: str = "neutral") -> str:
    return (
        f'<div class="card"><div class="label">{label}</div>'
        f'<div class="value {tone}">{value}</div>'
        f'<div class="sub {tone}">{sub}</div></div>'
    )


def style_fig(fig: go.Figure, height: int = 460) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#e6f1ff"),
        legend=dict(orientation="h", y=1.08, x=0),
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified",
        xaxis=dict(gridcolor="rgba(125,139,179,0.15)"),
        yaxis=dict(gridcolor="rgba(125,139,179,0.15)"),
    )
    return fig


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Booting neural core…")
def load_artifacts():
    model, source = None, None
    if os.path.exists(MODEL_KERAS):
        from tensorflow import keras

        model = keras.models.load_model(MODEL_KERAS)
        source = "native .keras"
    elif os.path.exists(MODEL_PKL):
        model = joblib.load(MODEL_PKL)
        source = "joblib .pkl (fragile)"
    scaler = joblib.load(SCALER_PKL) if os.path.exists(SCALER_PKL) else None
    return model, scaler, source


@st.cache_data(ttl=3600, show_spinner="Pulling market data…")
def fetch_prices(ticker: str, start: date, end: date) -> pd.Series:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.Series(dtype=float, name="Close")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    s = df["Close"].dropna().astype(float)
    s.name = "Close"
    return s


def parse_pasted(text: str, last_date: date) -> pd.Series:
    tokens = text.replace(",", " ").split()
    try:
        vals = [float(t) for t in tokens]
    except ValueError as e:
        raise ValueError(f"non-numeric value in pasted prices ({e})")
    # Dates are synthetic business days ending at last_date; only the order matters to the model.
    idx = pd.bdate_range(end=pd.Timestamp(last_date), periods=len(vals))
    return pd.Series(vals, index=idx, name="Close")


def parse_csv(file) -> pd.Series:
    df = pd.read_csv(file)
    cols = {c.lower().strip(): c for c in df.columns}
    if "close" not in cols:
        raise ValueError("CSV needs a 'Close' column (a 'Date' column is optional)")
    vals = pd.to_numeric(df[cols["close"]], errors="coerce").values
    if "date" in cols:
        idx = pd.DatetimeIndex(pd.to_datetime(df[cols["date"]], errors="coerce"))
    else:
        idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=len(vals))
    s = pd.Series(vals, index=idx, name="Close")
    s = s[~s.index.isna()].dropna().sort_index()
    return s.astype(float)


# ----------------------------------------------------------------------------
# Core maths
# ----------------------------------------------------------------------------
def to_scaled(scaler, values) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    return scaler.transform(pd.DataFrame({"Close": arr})).ravel()


def from_scaled(scaler, values) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1, 1)
    return scaler.inverse_transform(arr).ravel()


def make_windows(scaled: np.ndarray) -> np.ndarray:
    # windows[i] = scaled[i : i+LOOKBACK]  -> predicts scaled[i+LOOKBACK]
    w = np.lib.stride_tricks.sliding_window_view(scaled, LOOKBACK)[:-1]
    return w[..., None]


def backtest(model, scaler, prices: pd.Series, n_days: int) -> pd.DataFrame:
    n_days = min(n_days, len(prices) - LOOKBACK)
    vals = prices.values[-(n_days + LOOKBACK):]
    X = make_windows(to_scaled(scaler, vals))
    pred = from_scaled(scaler, model.predict(X, verbose=0))
    return pd.DataFrame(
        {
            "Actual": vals[LOOKBACK:],
            "LSTM": pred,
            "Naive (yesterday)": vals[LOOKBACK - 1:-1],
        },
        index=prices.index[-n_days:],
    )


def metrics(df: pd.DataFrame, col: str) -> dict:
    err = df[col] - df["Actual"]
    prev = df["Naive (yesterday)"]
    dir_hit = np.sign(df[col] - prev) == np.sign(df["Actual"] - prev)
    return {
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAE": float(np.mean(np.abs(err))),
        "MAPE": float(np.mean(np.abs(err / df["Actual"])) * 100),
        "Bias": float(np.mean(err)),
        "DirAcc": float(dir_hit.mean() * 100),
    }


def recursive_forecast(model, scaler, prices: pd.Series, horizon: int) -> np.ndarray:
    window = list(to_scaled(scaler, prices.values[-LOOKBACK:]))
    out = []
    for _ in range(horizon):
        x = np.array(window[-LOOKBACK:]).reshape(1, LOOKBACK, 1)
        p = float(model.predict(x, verbose=0)[0, 0])
        out.append(p)
        window.append(p)  # feeds its own output back in -> errors compound
    return from_scaled(scaler, out)


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
model, train_scaler, model_source = load_artifacts()

with st.sidebar:
    st.markdown('<h3 class="orb">CONTROL DECK</h3>', unsafe_allow_html=True)
    source = st.radio("Data source", ["Yahoo Finance", "Paste prices", "Upload CSV"])
    ticker = st.text_input("Ticker (symbol / label)", "AAPL").strip().upper()
    pasted, uploaded, last_date = "", None, date.today()
    if source == "Yahoo Finance":
        c1, c2 = st.columns(2)
        start = c1.date_input("Start", date(2018, 1, 1))
        end = c2.date_input("End", date.today() + timedelta(days=1))
    elif source == "Paste prices":
        pasted = st.text_area(
            f"Closing prices, oldest → newest (≥ {LOOKBACK}; ≥ {LOOKBACK + 30} enables backtest)",
            height=200,
            placeholder="336.91, 340.08, 338.19, ...",
        )
        last_date = st.date_input("Date of the last price", date.today())
    else:
        uploaded = st.file_uploader("CSV with Close (and optional Date) column", type="csv")
    scaler_mode = st.radio(
        "Scaling mode",
        ["Training scaler", "Refit on loaded history"],
        help=(
            "Training scaler = exactly what the model saw (AAPL 2018–2025 range). "
            "Prices above that range produce scaled values > 1, which the model "
            "never trained on. Refit = re-maps the loaded history to 0–1; a hack "
            "that avoids out-of-range inputs but changes what the numbers mean."
        ),
    )
    horizon = st.slider("Forecast horizon (trading days)", 1, 30, 5)
    bt_days = st.slider("Backtest window (days)", 30, 500, 250, step=10)
    st.markdown("---")
    st.caption(f"Model: {model_source or 'NOT FOUND'}")
    st.caption(f"Scaler: {'loaded' if train_scaler is not None else 'NOT FOUND'}")

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    f"""<div class="hero"><h1>NEURAL//MARKET</h1>
    <p>Stacked LSTM · {LOOKBACK}-day lookback · ticker <b>{ticker}</b> ·
    educational model, not trading advice</p></div>""",
    unsafe_allow_html=True,
)

if model is None or train_scaler is None:
    st.error(
        f"Missing files. Put `{MODEL_KERAS}` (or `{MODEL_PKL}`) and `{SCALER_PKL}` "
        "in the same folder as app.py."
    )
    st.stop()

try:
    if source == "Yahoo Finance":
        prices = fetch_prices(ticker, start, end)
    elif source == "Paste prices":
        if not pasted.strip():
            st.info("Paste closing prices in the sidebar to run a prediction.")
            st.stop()
        prices = parse_pasted(pasted, last_date)
    else:
        if uploaded is None:
            st.info("Upload a CSV in the sidebar to run a prediction.")
            st.stop()
        prices = parse_csv(uploaded)
except ValueError as e:
    st.error(f"Could not read prices: {e}")
    st.stop()

if len(prices) < LOOKBACK:
    st.error(f"Got {len(prices)} prices; the model needs at least {LOOKBACK}.")
    st.stop()
has_bt = len(prices) >= LOOKBACK + 30
if not has_bt:
    st.info(f"{len(prices)} prices: forecast only. Backtest needs ≥ {LOOKBACK + 30}.")

if scaler_mode == "Training scaler":
    scaler = train_scaler
else:
    scaler = MinMaxScaler((0, 1)).fit(pd.DataFrame({"Close": prices.values}))

if ticker != "AAPL":
    st.warning(
        "The model was trained only on AAPL. Applying it to another ticker is "
        "out-of-distribution; treat results as a stress test, not a forecast."
    )

# ----------------------------------------------------------------------------
# Compute
# ----------------------------------------------------------------------------
with st.spinner("Running inference…"):
    bt = backtest(model, scaler, prices, bt_days) if has_bt else None
    m_lstm = metrics(bt, "LSTM") if has_bt else None
    m_naive = metrics(bt, "Naive (yesterday)") if has_bt else None
    fc = recursive_forecast(model, scaler, prices, horizon)

last_price = float(prices.iloc[-1])
next_price = float(fc[0])
delta = next_price - last_price
future_idx = pd.bdate_range(prices.index[-1] + pd.Timedelta(days=1), periods=horizon)
resid_sd = float((bt["LSTM"] - bt["Actual"]).std()) if has_bt else 0.0
band = resid_sd * np.sqrt(np.arange(1, horizon + 1))  # heuristic widening

last_window_scaled = to_scaled(scaler, prices.values[-LOOKBACK:])
out_of_range = (last_window_scaled.max() > 1.0) or (last_window_scaled.min() < 0.0)

# ----------------------------------------------------------------------------
# KPI strip
# ----------------------------------------------------------------------------
k = st.columns(5)
k[0].markdown(card("Last close", f"${last_price:,.2f}", str(prices.index[-1].date())), unsafe_allow_html=True)
k[1].markdown(
    card("Next-day prediction", f"${next_price:,.2f}", f"{delta:+.2f} ({delta / last_price * 100:+.2f}%)",
         "up" if delta >= 0 else "down"),
    unsafe_allow_html=True,
)
if has_bt:
    beats = m_lstm["RMSE"] < m_naive["RMSE"]
    k[2].markdown(
        card("LSTM RMSE", f"${m_lstm['RMSE']:.2f}", f"naive ${m_naive['RMSE']:.2f}", "up" if beats else "down"),
        unsafe_allow_html=True,
    )
    k[3].markdown(card("Direction hit-rate", f"{m_lstm['DirAcc']:.1f}%", "coin flip = 50%",
                       "up" if m_lstm["DirAcc"] > 52 else "down"), unsafe_allow_html=True)
else:
    beats = True
    k[2].markdown(card("LSTM RMSE", "—", "needs backtest"), unsafe_allow_html=True)
    k[3].markdown(card("Direction hit-rate", "—", "needs backtest"), unsafe_allow_html=True)
k[4].markdown(card("Input range", "OUT" if out_of_range else "OK",
                   f"scaled max {last_window_scaled.max():.2f}", "down" if out_of_range else "up"),
              unsafe_allow_html=True)

if out_of_range:
    st.error(
        "Recent prices fall outside the range the scaler was fitted on, so the model "
        "is extrapolating. Predictions in this regime are systematically unreliable."
    )
if not beats:
    st.warning(
        "On this backtest window the LSTM is worse than simply predicting yesterday's "
        "close. That is the benchmark it has to beat to be useful."
    )

st.write("")
tab_fc, tab_bt, tab_diag, tab_data = st.tabs(["⚡ Forecast", "🧪 Backtest", "📡 Diagnostics", "🗂 Data"])

# ----------------------------------------------------------------------------
# Forecast tab
# ----------------------------------------------------------------------------
with tab_fc:
    hist = prices.iloc[-180:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="History",
                             line=dict(color="#00f0ff", width=2)))
    x_band = list(future_idx) + list(future_idx[::-1])
    y_band = list(fc + band) + list((fc - band)[::-1])
    fig.add_trace(go.Scatter(x=x_band, y=y_band, fill="toself", fillcolor="rgba(255,43,214,0.12)",
                             line=dict(width=0), name="±1σ (heuristic)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[hist.index[-1], *future_idx], y=[last_price, *fc], name="Forecast",
                             line=dict(color="#ff2bd6", width=2.5, dash="dot"),
                             mode="lines+markers", marker=dict(size=6)))
    st.plotly_chart(style_fig(fig), use_container_width=True)
    st.caption(
        "Multi-day values are recursive: each prediction is fed back as input, so errors "
        "compound. The band is backtest residual σ × √h — a rough guide, not a calibrated interval."
    )
    fc_df = pd.DataFrame({"Date": future_idx.date, "Forecast": fc.round(2),
                          "Low (−1σ)": (fc - band).round(2), "High (+1σ)": (fc + band).round(2)})
    st.dataframe(fc_df, hide_index=True, use_container_width=True)
    st.download_button("⬇ Download forecast CSV", fc_df.to_csv(index=False), f"{ticker}_forecast.csv")

# ----------------------------------------------------------------------------
# Backtest tab
# ----------------------------------------------------------------------------
with tab_bt:
    if not has_bt:
        st.info(f"Provide at least {LOOKBACK + 30} prices to run a backtest.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=bt.index, y=bt["Actual"], name="Actual", line=dict(color="#00f0ff", width=2)))
        fig.add_trace(go.Scatter(x=bt.index, y=bt["LSTM"], name="LSTM", line=dict(color="#ff2bd6", width=2)))
        fig.add_trace(go.Scatter(x=bt.index, y=bt["Naive (yesterday)"], name="Naive",
                                 line=dict(color="#9dff00", width=1, dash="dot")))
        st.plotly_chart(style_fig(fig), use_container_width=True)

        comp = pd.DataFrame([m_lstm, m_naive], index=["LSTM", "Naive (yesterday)"]).round(3)
        st.dataframe(comp, use_container_width=True)
        st.caption(
            "Bias = mean(pred − actual). A large negative bias means the model sits below "
            "the price, the classic symptom of lag/out-of-range inputs."
        )

        err = bt["LSTM"] - bt["Actual"]
        fig_e = go.Figure(go.Histogram(x=err, nbinsx=40, marker_color="#ff2bd6", opacity=0.8))
        fig_e.update_layout(title="Error distribution (pred − actual)")
        st.plotly_chart(style_fig(fig_e, 320), use_container_width=True)

# ----------------------------------------------------------------------------
# Diagnostics tab
# ----------------------------------------------------------------------------
with tab_diag:
    c1, c2 = st.columns([2, 1])
    with c1:
        fig = go.Figure(go.Scatter(y=last_window_scaled, mode="lines+markers",
                                   line=dict(color="#00f0ff"), name="Scaled input"))
        fig.add_hline(y=1.0, line_dash="dash", line_color="#ff4d6d", annotation_text="train max")
        fig.add_hline(y=0.0, line_dash="dash", line_color="#ff4d6d", annotation_text="train min")
        fig.update_layout(title=f"Last {LOOKBACK}-day input as the model sees it")
        st.plotly_chart(style_fig(fig, 380), use_container_width=True)
    with c2:
        st.markdown(card("Scaler min", f"${float(scaler.data_min_[0]):,.2f}"), unsafe_allow_html=True)
        st.write("")
        st.markdown(card("Scaler max", f"${float(scaler.data_max_[0]):,.2f}"), unsafe_allow_html=True)
        st.write("")
        st.markdown(card("Lookback", f"{LOOKBACK} days"), unsafe_allow_html=True)
        st.write("")
        st.markdown(card("Model params", f"{model.count_params():,}"), unsafe_allow_html=True)

    if has_bt:
        lag_corr = {
            lag: float(np.corrcoef(bt["LSTM"].values[lag:], bt["Actual"].values[: len(bt) - lag])[0, 1])
            for lag in range(0, 6)
        }
        best_lag = max(lag_corr, key=lag_corr.get)
        st.markdown(
            f"**Lag check:** predictions correlate best with the actual price shifted by "
            f"**{best_lag} day(s)** (r = {lag_corr[best_lag]:.4f}). If this is ≥ 1, the model "
            "is mostly echoing past prices rather than anticipating moves."
        )

# ----------------------------------------------------------------------------
# Data tab
# ----------------------------------------------------------------------------
with tab_data:
    st.dataframe(prices.to_frame().iloc[::-1], use_container_width=True, height=420)
    st.download_button("⬇ Download price history", prices.to_csv(), f"{ticker}_prices.csv")
    if has_bt:
        st.download_button("⬇ Download backtest", bt.to_csv(), f"{ticker}_backtest.csv")