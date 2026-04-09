# 💠 Hybrid Information Mixing Module (HIMM) for Stock Market Prediction

A Multimodal Deep Learning System that fuses **BERT-based Sentiment Analysis** and **GRU-based Temporal Modeling** to provide high-accuracy market signals. Built with Python & Streamlit, it delivers real-time IST-synchronized charts and interaction-mixing predictions for data-driven trading decisions.

---

## 📈 Hybrid AI-Driven Financial Forecasting System

A Real-Time Financial Dashboard that uses a **Hybrid Information Mixing Module (HIMM)** to analyze news sentiment and price action simultaneously to predict trend direction.

---

## 📖 Table of Contents
* [About the Project](#-about-the-project)
* [Key Features](#-key-features)
* [Tech Stack](#-tech-stack)
* [System Architecture](#-system-architecture)
* [Installation & Setup](#-installation--setup)
* [How It Works](#-how-it-works)
* [Future Scope](#-future-scope)
* [Contact](#-contact)

---

## 📌 About The Project
Retail traders often struggle with emotional decision-making and reliance on lagging indicators that only reflect past market behavior. Existing tools fail to account for the impact of global news sentiment on market volatility.

This project solves that gap. It is a **Hybrid Analytical System** that combines:
* **Semantic Analysis:** Uses **BERT** to translate unstructured financial headlines into mathematical sentiment weights.
* **Temporal Modeling:** Employs **GRU (Gated Recurrent Units)** to identify patterns in 60-period historical price sequences.
* **Interaction Mixing:** Features a custom **MLP Fusion Layer** that identifies non-linear correlations between news and momentum to filter out market noise.

---

## 🚀 Key Features

🔮 **Multimodal AI Fusion:** Fuses BERT (NLP) and GRU (Time-Series) for a holistic market view.
📊 **Interactive Pro Charts:** Professional-grade, zoomable 3-Pane candlestick charts (Price, RSI, Volume) powered by `lightweight-charts`.
🇮🇳 **IST Standard Synchronization:** Automatically aligns global market data (UTC) to **Indian Standard Time (IST)**.
🎯 **Dynamic Signal Sensitivity:** User-adjustable confidence thresholds to calibrate model risk and signal frequency.
🌍 **Universal Asset Support:** Analyzes **Indian Stocks** (.NS), **Crypto** (BTC-USD), and **Forex** via Yahoo Finance.

---

## 🛠 Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Language** | Python 3.10 | Core Logic & ML Pipeline |
| **Frontend** | Streamlit | Web Dashboard & UI |
| **NLP Core** | BERT (HuggingFace) | News Sentiment Extraction |
| **Temporal Core** | GRU (TensorFlow) | Sequence Pattern Recognition |
| **Visuals** | Lightweight Charts v5 | TradingView-Standard Charts |
| **API** | yfinance | Live OHLC Market Data |

---

## 🏗 System Architecture

The system follows a **4-Layer Multimodal Architecture**:
1. **Data Layer:** Fetches live OHLCV data and real-time news via Yahoo Finance and RSS feeds.
2. **Processing Layer:** Standardizes timezones to IST and calculates Technical Indicators (EMA, RSI).
3. **Intelligence Layer:** Fuses BERT embeddings and GRU hidden states through the **HIMM Interaction-Mixing MLP**.
4. **Presentation Layer:** Renders the Gauge confidence meter and dynamic Buy/Sell markers on the dashboard.

---

## ⚡ Installation & Setup

Follow these steps to run the project locally.

### Prerequisites
* Python 3.10 or higher.
* Trained `model.h5` file placed in the project root directory.

### Steps
1. **Clone the Repository**
   ```bash
   git clone [https://github.com/shaga-amrutha/Hybrid-HIMM-Trading.git](https://github.com/shaga-amrutha/Hybrid-HIMM-Trading.git)
   cd Hybrid-HIMM-Trading
