import time
import warnings
warnings.filterwarnings("ignore")

import ccxt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

st.set_page_config(
    page_title="JOGI AI TRADING PRO",
    page_icon="📈",
    layout="wide"
)

st.title("📈 JOGI AI TRADING PRO")
st.caption("ICT + SMC + LSTM + Machine Learning Ensemble | Educational signal dashboard")

st.warning(
    "Signals are estimates, not guaranteed outcomes. Binance prices may differ "
    "from Binarium quotes. This app does not place trades."
)

SYMBOLS = {
    "BTCUSDT": "BTC/USDT",
    "ETHUSDT": "ETH/USDT",
    "SOLUSDT": "SOL/USDT"
}
LOOKBACK = 30
HISTORY_DAYS = 10

with st.sidebar:
    st.header("Bot settings")
    selected_symbol = st.selectbox("Coin", list(SYMBOLS.keys()))
    call_threshold = st.slider("CALL score threshold", 0.50, 0.80, 0.60, 0.01)
    put_threshold = st.slider("PUT score threshold", 0.20, 0.50, 0.40, 0.01)
    refresh = st.checkbox("Auto refresh every 60 seconds", value=False)
    run_button = st.button("🔄 Refresh / Analyze", type="primary")

@st.cache_data(ttl=60, show_spinner=False)
def fetch_ohlcv(symbol, days=10):
    exchange = ccxt.binance({"enableRateLimit": True, "timeout": 20000})
    since = exchange.milliseconds() - days * 24 * 60 * 60 * 1000
    rows = []
    while since < exchange.milliseconds():
        batch = exchange.fetch_ohlcv(symbol, timeframe="1m", since=since, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        next_since = batch[-1][0] + 60_000
        if next_since <= since:
            break
        since = next_since
        time.sleep(exchange.rateLimit / 1000)
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    if df.empty:
        raise RuntimeError("Binance ne candle data return nahi kiya.")
    df = df.drop_duplicates("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    now = pd.Timestamp.now(tz="UTC")
    df = df[df.index + pd.Timedelta(minutes=1) <= now]
    return df.astype(float)

def add_indicators(df):
    d = df.copy()
    delta = d.close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    d["rsi"] = 100 - 100 / (1 + rs)

    for span in [9, 21, 50, 200]:
        d[f"ema{span}"] = d.close.ewm(span=span, adjust=False).mean()

    e12 = d.close.ewm(span=12, adjust=False).mean()
    e26 = d.close.ewm(span=26, adjust=False).mean()
    d["macd"] = e12 - e26
    d["macd_signal"] = d.macd.ewm(span=9, adjust=False).mean()

    tr = pd.concat([
        d.high - d.low,
        (d.high - d.close.shift()).abs(),
        (d.low - d.close.shift()).abs()
    ], axis=1).max(axis=1)
    d["atr_pct"] = tr.rolling(14).mean() / d.close

    d["vol_ma"] = d.volume.rolling(20).mean()
    d["volume_ratio"] = d.volume / d.vol_ma.replace(0, np.nan)
    mid = d.close.rolling(20).mean()
    sd = d.close.rolling(20).std()
    d["bb_width"] = 4 * sd / mid.replace(0, np.nan)
    d["body_pct"] = (d.close - d.open).abs() / d.close
    d["return_1"] = d.close.pct_change(1)
    d["return_3"] = d.close.pct_change(3)
    d["return_5"] = d.close.pct_change(5)
    return d

def add_smc_ict(df):
    d = df.copy()
    # Swing pivots are confirmed two candles after the pivot.
    rh = d.high.rolling(5).max()
    rl = d.low.rolling(5).min()
    ph = d.high.shift(2).eq(rh)
    pl = d.low.shift(2).eq(rl)
    d["swing_high"] = d.high.shift(2).where(ph).ffill()
    d["swing_low"] = d.low.shift(2).where(pl).ffill()

    prev_high = d.swing_high.shift(1)
    prev_low = d.swing_low.shift(1)
    d["bos_up"] = ((d.close > prev_high) & (d.close.shift() <= prev_high)).astype(int)
    d["bos_down"] = ((d.close < prev_low) & (d.close.shift() >= prev_low)).astype(int)

    structure = np.where(d.bos_up == 1, 1, np.where(d.bos_down == 1, -1, np.nan))
    d["structure"] = pd.Series(structure, index=d.index).ffill().fillna(0)
    d["choch_up"] = ((d.structure.shift() == -1) & (d.bos_up == 1)).astype(int)
    d["choch_down"] = ((d.structure.shift() == 1) & (d.bos_down == 1)).astype(int)

    d["bull_fvg"] = (d.low > d.high.shift(2)).astype(int)
    d["bear_fvg"] = (d.high < d.low.shift(2)).astype(int)
    bull_lower = d.high.shift(2).where(d.bull_fvg == 1).ffill()
    bear_upper = d.low.shift(2).where(d.bear_fvg == 1).ffill()
    d["ifvg_bear"] = (bull_lower.notna() & (d.close < bull_lower)).astype(int)
    d["ifvg_bull"] = (bear_upper.notna() & (d.close > bear_upper)).astype(int)

    old_high = d.high.rolling(20).max().shift(1)
    old_low = d.low.rolling(20).min().shift(1)
    d["sweep_high"] = ((d.high > old_high) & (d.close < old_high)).astype(int)
    d["sweep_low"] = ((d.low < old_low) & (d.close > old_low)).astype(int)

    tol = d.close * 0.0005
    d["equal_highs"] = ((d.high - d.high.shift()).abs() < tol).astype(int)
    d["equal_lows"] = ((d.low - d.low.shift()).abs() < tol).astype(int)

    d["support"] = d.low.rolling(50).min().shift(1)
    d["resistance"] = d.high.rolling(50).max().shift(1)
    d["dist_support"] = (d.close - d.support) / d.close
    d["dist_resistance"] = (d.resistance - d.close) / d.close

    hi = d.high.rolling(50).max().shift(1)
    lo = d.low.rolling(50).min().shift(1)
    d["premium_discount"] = (d.close - (hi + lo) / 2) / d.close

    # Simplified OB / breaker proxies, not full discretionary ICT zones.
    d["bull_ob"] = ((d.close.shift() < d.open.shift()) & (d.bos_up == 1)).astype(int)
    d["bear_ob"] = ((d.close.shift() > d.open.shift()) & (d.bos_down == 1)).astype(int)
    d["bull_breaker"] = ((d.bear_ob.shift() == 1) & (d.close > d.high.shift())).astype(int)
    d["bear_breaker"] = ((d.bull_ob.shift() == 1) & (d.close < d.low.shift())).astype(int)
    return d

def make_htf(raw, rule, prefix):
    x = raw.resample(rule, label="right", closed="left").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna()
    x = add_indicators(x)
    cols = ["rsi", "ema9", "ema21", "ema50", "ema200", "macd", "atr_pct", "bb_width", "volume_ratio"]
    x = x[cols].copy()
    x.columns = [prefix + c for c in cols]
    return x

def build_features(raw):
    d = add_smc_ict(add_indicators(raw))
    f5 = make_htf(raw, "5min", "m5_")
    f15 = make_htf(raw, "15min", "m15_")
    d = pd.merge_asof(d.sort_index(), f5.sort_index(), left_index=True, right_index=True, direction="backward")
    d = pd.merge_asof(d.sort_index(), f15.sort_index(), left_index=True, right_index=True, direction="backward")
    return d.replace([np.inf, -np.inf], np.nan)

def make_sequences(X, y, lookback):
    xs, ys, positions = [], [], []
    for i in range(lookback - 1, len(X)):
        xs.append(X[i-lookback+1:i+1])
        ys.append(y[i])
        positions.append(i)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), np.asarray(positions)

def make_lstm(n_features):
    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=(LOOKBACK, n_features)),
        Dropout(0.20),
        LSTM(16),
        Dropout(0.15),
        Dense(8, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

@st.cache_resource(show_spinner="Training AI models (first run can take several minutes)...", ttl=21600)
def train_models(symbol, data_signature, feature_data):
    # feature_data is passed in so the cache refreshes when source data changes.
    data = feature_data.copy()
    excluded = {"open", "high", "low", "close", "volume", "target"}
    feature_cols = [c for c in data.columns if c not in excluded]
    X = data[feature_cols].to_numpy(dtype=np.float32)
    y = data["target"].to_numpy(dtype=int)

    split = int(len(X) * 0.80)
    if split <= LOOKBACK + 100 or len(X) - split < LOOKBACK + 20:
        raise RuntimeError("Training/test data kam hai. History ko barhayein.")

    positives = max(int(y[:split].sum()), 1)
    negatives = max(int(len(y[:split]) - positives), 1)

    rf = RandomForestClassifier(
        n_estimators=120, max_depth=7, min_samples_leaf=8,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    xgb = XGBClassifier(
        n_estimators=120, max_depth=3, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=negatives / positives,
        eval_metric="logloss", random_state=42, n_jobs=2
    )
    lgb = LGBMClassifier(
        n_estimators=120, max_depth=5, learning_rate=0.04,
        class_weight="balanced", verbosity=-1, random_state=42,
        n_jobs=2
    )

    rf.fit(X[:split], y[:split])
    xgb.fit(X[:split], y[:split])
    lgb.fit(X[:split], y[:split])

    rf_test = rf.predict_proba(X[split:])[:, 1]
    xgb_test = xgb.predict_proba(X[split:])[:, 1]
    lgb_test = lgb.predict_proba(X[split:])[:, 1]

    scaler_test = StandardScaler().fit(X[:split])
    X_scaled_test = np.vstack([
        scaler_test.transform(X[:split]),
        scaler_test.transform(X[split:])
    ])
    Xseq, yseq, positions = make_sequences(X_scaled_test, y, LOOKBACK)
    train_mask = positions < split
    test_mask = positions >= split

    tf.keras.backend.clear_session()
    lstm_test = make_lstm(X.shape[1])
    lstm_test.fit(
        Xseq[train_mask], yseq[train_mask],
        epochs=5, batch_size=128, shuffle=False,
        validation_split=0.10,
        callbacks=[EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)],
        verbose=0
    )
    lstm_probs = lstm_test.predict(Xseq[test_mask], verbose=0).ravel()
    test_positions = positions[test_mask]
    offset = test_positions - split

    ensemble = (
        rf_test[offset] + xgb_test[offset] +
        lgb_test[offset] + lstm_probs
    ) / 4
    actual = y[test_positions]
    predicted = (ensemble >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "test_rows": int(len(actual))
    }

    # Refit final models on all available rows for current signal.
    rf.fit(X, y)
    xgb.fit(X, y)
    lgb.fit(X, y)

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    Xfull, yfull, _ = make_sequences(X_scaled, y, LOOKBACK)

    tf.keras.backend.clear_session()
    lstm = make_lstm(X.shape[1])
    lstm.fit(Xfull, yfull, epochs=5, batch_size=128, shuffle=False, verbose=0)

    return feature_cols, rf, xgb, lgb, lstm, scaler, metrics

def get_signal(data, feature_cols, rf, xgb, lgb, lstm, scaler, call_cut, put_cut):
    X = data[feature_cols].to_numpy(dtype=np.float32)
    last = data.iloc[-1]

    rf_p = float(rf.predict_proba(X[-1:])[0, 1])
    xgb_p = float(xgb.predict_proba(X[-1:])[0, 1])
    lgb_p = float(lgb.predict_proba(X[-1:])[0, 1])

    seq = scaler.transform(X[-LOOKBACK:]).reshape(1, LOOKBACK, len(feature_cols))
    lstm_p = float(lstm.predict(seq, verbose=0)[0][0])
    ensemble = float(np.mean([rf_p, xgb_p, lgb_p, lstm_p]))

    bullish_trend = (
        last.ema9 > last.ema21 and
        last.m5_ema9 > last.m5_ema21 and
        last.m15_ema9 > last.m15_ema21
    )
    bearish_trend = (
        last.ema9 < last.ema21 and
        last.m5_ema9 < last.m5_ema21 and
        last.m15_ema9 < last.m15_ema21
    )

    bullish_smc = any([
        last.bos_up == 1, last.choch_up == 1,
        last.sweep_low == 1, last.bull_fvg == 1,
        last.ifvg_bull == 1, last.bull_ob == 1,
        last.bull_breaker == 1
    ])
    bearish_smc = any([
        last.bos_down == 1, last.choch_down == 1,
        last.sweep_high == 1, last.bear_fvg == 1,
        last.ifvg_bear == 1, last.bear_ob == 1,
        last.bear_breaker == 1
    ])

    signal = "WAIT"
    if ensemble >= call_cut and bullish_trend and bullish_smc and 35 < last.rsi < 75:
        signal = "CALL"
    elif ensemble <= put_cut and bearish_trend and bearish_smc and 25 < last.rsi < 65:
        signal = "PUT"

    return {
        "signal": signal, "ensemble": ensemble,
        "rf": rf_p, "xgb": xgb_p, "lgb": lgb_p, "lstm": lstm_p,
        "price": float(last.close), "rsi": float(last.rsi),
        "support": float(last.support), "resistance": float(last.resistance),
        "bullish_trend": bool(bullish_trend),
        "bearish_trend": bool(bearish_trend),
        "bos_up": int(last.bos_up), "bos_down": int(last.bos_down),
        "choch_up": int(last.choch_up), "choch_down": int(last.choch_down),
        "bull_fvg": int(last.bull_fvg), "bear_fvg": int(last.bear_fvg),
        "ifvg_bull": int(last.ifvg_bull), "ifvg_bear": int(last.ifvg_bear),
        "sweep_high": int(last.sweep_high), "sweep_low": int(last.sweep_low)
    }

try:
    symbol = SYMBOLS[selected_symbol]
    with st.spinner(f"Downloading {selected_symbol} candles from Binance..."):
        raw = fetch_ohlcv(symbol, HISTORY_DAYS)

    if len(raw) < 5000:
        st.error("Binance se kafi candles nahi milin. Thori dair baad dobara try karein.")
        st.stop()

    with st.spinner("Calculating ICT / SMC / multi-timeframe features..."):
        features = build_features(raw)
        features["target"] = (features.close.shift(-1) > features.close).astype(float)
        features.loc[features.close.shift(-1).isna(), "target"] = np.nan
        data = features.dropna().copy()

    if len(data) < 3000:
        st.error("Clean training data kam hai. Kuch dair baad dobara try karein.")
        st.stop()

    # Use the last timestamp as a cache key; model retrains when fetched data changes.
    signature = str(data.index[-1]) + str(len(data))

    feature_cols, rf, xgb, lgb, lstm, scaler, metrics = train_models(
        selected_symbol, signature, data
    )

    result = get_signal(
        data, feature_cols, rf, xgb, lgb, lstm,
        scaler, call_threshold, put_threshold
    )

    st.subheader(f"{selected_symbol} — Latest Signal")
    c1, c2, c3 = st.columns(3)
    c1.metric("SIGNAL", result["signal"])
    c2.metric("BTC/coin price", f'{result["price"]:,.6f}')
    c3.metric("Ensemble score (not win probability)", f'{result["ensemble"]*100:.1f}%')

    st.caption("Signal horizon: next 1-minute candle estimate. Check the broker's live quote before any decision.")

    st.subheader("AI model scores")
    score_cols = st.columns(4)
    for col, name in zip(score_cols, ["Random Forest", "XGBoost", "LightGBM", "LSTM"]):
        key = {"Random Forest": "rf", "XGBoost": "xgb", "LightGBM": "lgb", "LSTM": "lstm"}[name]
        col.metric(name, f'{result[key]*100:.1f}%')

    st.subheader("Market structure / ICT / SMC")
    a, b, c, d = st.columns(4)
    a.metric("RSI", f'{result["rsi"]:.1f}')
    b.metric("Support", f'{result["support"]:,.6f}')
    c.metric("Resistance", f'{result["resistance"]:,.6f}')
    d.metric("Out-of-sample balanced accuracy", f'{metrics["balanced_accuracy"]*100:.1f}%')

    status = pd.DataFrame({
        "Feature": [
            "BOS Up", "BOS Down", "CHoCH Up", "CHoCH Down",
            "Bullish FVG", "Bearish FVG", "Bullish IFVG",
            "Bearish IFVG", "Liquidity sweep high", "Liquidity sweep low",
            "5m/15m bullish trend", "5m/15m bearish trend"
        ],
        "Current": [
            result["bos_up"], result["bos_down"],
            result["choch_up"], result["choch_down"],
            result["bull_fvg"], result["bear_fvg"],
            result["ifvg_bull"], result["ifvg_bear"],
            result["sweep_high"], result["sweep_low"],
            result["bullish_trend"], result["bearish_trend"]
        ]
    })
    st.dataframe(status, use_container_width=True, hide_index=True)

    st.subheader("Recent BTC / coin candles")
    chart_data = raw.tail(180)
    fig = go.Figure(data=[go.Candlestick(
        x=chart_data.index,
        open=chart_data.open, high=chart_data.high,
        low=chart_data.low, close=chart_data.close,
        name=selected_symbol
    )])
    fig.update_layout(
        height=480, xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=25, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Model validation")
    st.write(
        f'Chronological holdout rows: {metrics["test_rows"]} | '
        f'Accuracy: {metrics["accuracy"]*100:.2f}% | '
        f'Balanced accuracy: {metrics["balanced_accuracy"]*100:.2f}%'
    )
    st.caption(
        "Validation is a simple historical holdout, not a full trading backtest. "
        "It does not include Binarium payout, spread, execution delay, or broker quote differences."
    )

    if refresh:
        time.sleep(60)
        st.rerun()

except Exception as exc:
    st.error(f"App error: {exc}")
    st.info(
        "Agar pehli deployment par error aaye, Manage app > Logs check karein. "
        "Free hosting resources limited ho sakte hain; pehli AI training mein waqt lagta hai."
    )

st.divider()
st.caption(
    "JOGI AI TRADING PRO • Educational tool only • No guaranteed profit • "
    "No broker login, API key, or automatic order execution."
)
