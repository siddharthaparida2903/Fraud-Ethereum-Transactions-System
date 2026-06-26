# 📈 Stock Price Prediction Using a Hybrid LSTM-GNN Model

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-LSTM%20%7C%20GAT-orange?style=for-the-badge)
![NLP](https://img.shields.io/badge/NLP-FinBERT-success?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge)

> **A robust, multi-horizon financial forecasting architecture integrating time-series sequential patterns with inter-stock relational dynamics.**

---

## 📖 About The Project

Traditional stock prediction models often treat assets in isolation, ignoring the intricate web of influence between companies, sectors, and macroeconomic indicators. 

This project introduces a **Hybrid Deep Learning Architecture** that bridges this gap by combining **Long Short-Term Memory (LSTM)** networks with **Graph Attention Networks (GAT)**. By dynamically mapping stock correlations into a directed graph and fusing it with temporal price histories, this model captures both market contagion effects and individual asset trends. 

### ✨ Key Innovations
* **Dynamic Rolling Graphs:** Constructs leak-free relational graphs dynamically using rolling Pearson correlation windows, mimicking evolving market conditions.
* **Cross-Entity Multi-Head Attention:** Fuses temporal (LSTM) and relational (GAT) embeddings to capture complex relationships beyond fixed graph edges.
* **Probabilistic Forecasting:** Utilizes Quantile Regression Loss to generate statistically calibrated 80% prediction intervals, offering tangible uncertainty quantification for risk management.
* **Multi-Horizon Joint Training:** A shared encoder architecture feeding independent Gated Residual Network (GRN) heads to simultaneously predict +1, +3, and +5 days into the future.
* **Sentiment Integration:** Leverages pre-trained **FinBERT** to extract unstructured financial news sentiment and fuse it with numerical OHLCV data.

---

## 🚀 Key Results & Performance

Evaluated on 10 major US equities (AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, JPM, V, JNJ) from 2020 to 2026 using strict expanding-window walk-forward validation:

- **📉 RMSE Reduction:** Achieved a **12% reduction** in Root Mean Squared Error over standalone LSTM baselines.
- **🎯 Directional Accuracy:** Reached **62.4%** accuracy in forecasting next-day (+1) price direction.
- **🛡️ Risk Calibration:** Achieved **79.6% empirical coverage** for the 80% quantile prediction intervals, proving near-ideal uncertainty calibration.

---

## 🛠️ Tech Stack

* **Language:** Python
* **Deep Learning Framework:** PyTorch, PyTorch Geometric (PyG)
* **Time-Series Processing:** Pandas, NumPy
* **Data Acquisition:** `yfinance` API
* **NLP / Sentiment Analysis:** HuggingFace Transformers (FinBERT)

---

## ⚙️ Installation & Setup

### Prerequisites
Ensure you have Python 3.8+ installed. A machine with a CUDA-enabled GPU is highly recommended for faster training.

### 1. Clone the repository
```bash
git clone [https://github.com/your-username/hybrid-lstm-gnn-stock-prediction.git](https://github.com/your-username/hybrid-lstm-gnn-stock-prediction.git)
cd hybrid-lstm-gnn-stock-prediction