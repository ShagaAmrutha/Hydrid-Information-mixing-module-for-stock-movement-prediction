import os
import sys
import pytz
import warnings

# --- 1. SILENCE ENVIRONMENT ---
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings('ignore', category=FutureWarning)

import tensorflow as tf
if not hasattr(tf, 'version'):
    class V: VERSION = "2.15.0"
    tf.version = V
    tf.__version__ = "2.15.0"
# ------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from lightweight_charts_v5 import lightweight_charts_v5_component
from transformers import BertTokenizer, TFBertModel
import feedparser
import plotly.graph_objects as go

# --- 2. INTELLIGENT TICKER ENGINE ---
@st.cache_data
def get_himm_data(ticker, interval="15m"):
    try:
        raw_symbol = ticker.strip().upper()
        
        # SMART TICKER FORMATTING
        if raw_symbol in ["NSEI", "NSEBANK", "BSESN"]: 
            symbol = f"^{raw_symbol}"
        elif any(x in raw_symbol for x in ["BTC", "ETH", "SOL", "DOGE", "XRP"]):
            # Crypto must have a hyphen and NO =X
            symbol = raw_symbol if "-" in raw_symbol else f"{raw_symbol}-USD"
        elif len(raw_symbol) == 6 and raw_symbol.isalpha():
            # Likely Forex (e.g., EURUSD)
            symbol = f"{raw_symbol}=X"
        else:
            symbol = raw_symbol

        # Timeframe History Logic (TradingView Style Long History)
        fetch_period = "60d" if interval in ["5m", "15m"] else "2y"
        if interval == "1m": fetch_period = "7d"
        
        df = yf.download(symbol, period=fetch_period, interval=interval)
        if df.empty: return None
        
        # IST (Indian Standard Time) Standard
        ist = pytz.timezone('Asia/Kolkata')
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC').tz_convert(ist)
        else:
            df.index = df.index.tz_convert(ist)
            
        df = df.reset_index()
        df.columns = [str(c[0]) if isinstance(c, tuple) else str(c) for c in df.columns]
        if 'Date' in df.columns: df = df.rename(columns={'Date': 'Datetime'})
        
        # Indicators
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain/(loss + 1e-7))))
        
        return df.dropna().reset_index(drop=True)
    except: return None

@st.cache_resource
def load_himm_assets():
    model = tf.keras.models.load_model("model.h5", custom_objects={"TFBertModel": TFBertModel}, compile=False)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer

# --- 3. UI DASHBOARD ---
st.set_page_config(page_title="HIMM IST Pro", layout="wide")

if 'markers' not in st.session_state: st.session_state.markers = []
if 'prediction' not in st.session_state: st.session_state.prediction = None

st.sidebar.title("💠 HIMM Control")
ticker_input = st.sidebar.text_input("Ticker (BTC-USD, RELIANCE.NS, EURUSD)", "BTC-USD")
interval_input = st.sidebar.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "1d"], index=2)
sensitivity = st.sidebar.slider("Signal Sensitivity", 0.0, 0.1, 0.005, step=0.001)

model, tokenizer = load_himm_assets()
data = get_himm_data(ticker_input, interval_input)

if data is not None:
    st.header(f"📊 {ticker_input} Professional Terminal (IST)")
    
    # Precise Timestamping
    ts = pd.to_datetime(data['Datetime']).astype(np.int64) // 10**9
    
    ohlc = [{"time": int(t), "open": float(o), "high": float(h), "low": float(l), "close": float(c)} for t, o, h, l, c in zip(ts, data['Open'], data['High'], data['Low'], data['Close'])]
    ema = [{"time": int(t), "value": float(v)} for t, v in zip(ts, data['EMA_50'])]
    rsi = [{"time": int(t), "value": float(v)} for t, v in zip(ts, data['RSI'])]
    vol = [{"time": int(t), "value": float(v), "color": "#26a69a" if c >= o else "#ef5350"} for t, v, o, c in zip(ts, data['Volume'], data['Open'], data['Close'])]

    # --- THE TRADINGVIEW 3-PANE GRID ---
    lightweight_charts_v5_component(
        name="himm_terminal_final",
        charts=[
            {   # PRICE PANE
                "chart": {"layout": {"background": {"color": "#0c0d14"}, "textColor": "#d1d4dc"}, "timeScale": {"timeVisible": True, "secondsVisible": False}}, 
                "series": [
                    {"type": "Candlestick", "data": ohlc, "markers": st.session_state.markers, "options": {"upColor": "#089981", "downColor": "#f23645"}}, 
                    {"type": "Line", "data": ema, "options": {"color": "#f2ad06", "lineWidth": 1, "title": "EMA 50"}}
                ], "height": 450
            },
            {   # RSI PANE
                "chart": {"layout": {"background": {"color": "#0c0d14"}, "textColor": "#d1d4dc"}}, 
                "series": [{"type": "Line", "data": rsi, "options": {"color": "#7e57c2", "lineWidth": 1, "title": "RSI"}}], "height": 130
            },
            {   # VOLUME PANE
                "chart": {"layout": {"background": {"color": "#0c0d14"}, "textColor": "#d1d4dc"}}, 
                "series": [{"type": "Histogram", "data": vol, "options": {"title": "Volume"}}], "height": 130
            }
        ], height=750
    )

    if st.button("RUN HIMM ANALYSIS 🚀", width='stretch'):
        window = 60
        if len(data) >= window:
            with st.spinner("Processing Hybrid Inputs..."):
                # Normalize Price
                raw_p = data['Close'].tail(window).values
                scaled_p = (raw_p - np.mean(raw_p)) / (np.std(raw_p) + 1e-7)
                
                # Semantic News
                search_term = ticker_input.split('.')[0].replace('^', '').replace('-USD', '')
                feed = feedparser.parse(f"https://news.google.com/rss/search?q={search_term}")
                news_txt = feed.entries[0].title if feed.entries else "Neutral."
                tokens = tokenizer(news_txt, padding='max_length', truncation=True, max_length=32, return_tensors="tf")
                
                # Predict
                pred = float(model.predict([scaled_p.reshape(1, window, 1), tokens['input_ids'], tokens['attention_mask']], verbose=0)[0][0])
                st.session_state.prediction = pred
                
                # IST Signal Marker
                last_ts = int(pd.to_datetime(data['Datetime'].iloc[-1]).timestamp())
                if pred > (0.5 + sensitivity):
                    st.session_state.markers = [{"time": last_ts, "position": "belowBar", "color": "#089981", "shape": "arrowUp", "text": "BUY"}]
                elif pred < (0.5 - sensitivity):
                    st.session_state.markers = [{"time": last_ts, "position": "aboveBar", "color": "#f23645", "shape": "arrowDown", "text": "SELL"}]
                else:
                    st.session_state.markers = []
                st.rerun()

    if st.session_state.prediction is not None:
        p = st.session_state.prediction
        sig = "🔥 BUY" if p > (0.5 + sensitivity) else "📉 SELL" if p < (0.5 - sensitivity) else "⚠️ NEUTRAL"
        st.subheader(f"AI Decision: {sig}")
        fig = go.Figure(go.Indicator(mode="gauge+number", value=p*100, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#2962ff"}}))
        st.plotly_chart(fig, width='stretch')