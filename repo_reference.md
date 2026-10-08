# 📘 Repo Reference: Fraud-Ethereum-Transactions-System

> **Last Updated:** 2026-10-08  
> **Path:** `d:\Projects\Projects-MachineLearning\Fraud-Ethereum-Transactions-System`  
> **Purpose:** Agent reference file — summarizes all files, architecture, models, and key implementation details.

---

## 📁 Repository Structure

```
Fraud-Ethereum-Transactions-System/
├── README.md                          # Project overview (XGBoost + SHAP pipeline)
├── DSML.pptx                          # Presentation slides (5.4 MB)
├── FraudDetectionEthereum.ipynb       # Main research notebook (4 experiment cells)
├── FraudDetectionEthereum_(XAI).ipynb # Focused XAI notebook (Colab-ready, 2 cells)
├── dataset/
│   └── etheruem_transaction_dataset.csv   # ~2.8 MB, 9841+ addresses, 47 raw features
├── Outputs/
│   ├── confusion_matrix_heatmap.png       # Model evaluation heatmap
│   ├── decision_tree_structure.png        # Top-3-level tree diagram (large, 805 KB)
│   ├── feature_importance_plot.png        # Top-10 feature bar chart
│   └── partial_dependence_plots.png       # PDP for top-3 features
└── streamlit-app/
    ├── app.py                             # Main Streamlit web app (634 lines)
    ├── requirements.txt                   # Pinned Python dependencies
    ├── streamlit_config.toml              # Streamlit server/theme config
    └── README.md                          # Deployment guide (497 lines)
```

---

## 🧠 System Architecture

```
Ethereum Account Data  →  ML Pipeline (XGBoost / Decision Tree)  →  SHAP Explainability
 (Volumes, Freq, ERC20)     (Imbalance-Optimized)                  (Force/Waterfall Plots)
                                                                            │
                                                                            ▼
                                                                  Interactive Dashboard
                                                                   (Streamlit Live Web)
```

---

## 📊 Dataset

| Property | Details |
|---|---|
| **File** | `dataset/etheruem_transaction_dataset.csv` |
| **Size** | ~2.8 MB |
| **Addresses** | 9,841 (training) + 1,969 (test) |
| **Total Features** | 47 raw → 43 predictive (after dropping non-predictive cols) |
| **Target Column** | `FLAG` (0 = Legitimate, 1 = Fraud) |
| **Class Distribution** | ~78.9% Legitimate, ~21.1% Fraudulent |
| **Source** | Elliptic++ Ethereum Transaction Dataset |

### Dropped Columns (Non-Predictive)
- `Unnamed: 0`, `Index`, `Address`
- `ERC20 most sent token type`, `ERC20_most_rec_token_type`

### Feature Categories (43 features used)

| Category | Count | Examples |
|---|---|---|
| Temporal | 3 | `Avg min between sent tnx`, `Time Diff first/last (Mins)` |
| Transaction Counts | 3 | `Sent tnx`, `Received Tnx`, `Number of Created Contracts` |
| Network | 2 | `Unique Received From Addresses`, `Unique Sent To Addresses` |
| Value – Received | 3 | `min/max/avg value received` |
| Value – Sent | 3 | `min/max/avg val sent` |
| Contract Txns | 3 | `min/max/avg value sent to contract` |
| Totals | 5 | `total Ether sent/received/balance`, `total transactions` |
| ERC20 Tokens | 21 | Timing, value, unique address features for ERC20 transfers |

---

## 🧪 Research Notebook: `FraudDetectionEthereum.ipynb`

**7 cells total — 4 standalone experiment blocks separated by markdown headers.**

### Cell 0 — RandomForest Baseline (White-box labels experiment)
- **Model:** `RandomForestClassifier`
- **Task:** Basic classification with feature importance
- **Outputs:** `feature_importance_plot.png`

### Cell 1 (Markdown) — Section Label
> *"WITH WHITE-BOX MODEL AND XAI (FEATURE IMPORTANCE)"*

### Cell 2 — Logistic Regression (White-box + Feature Importance)
- **Model:** `LogisticRegression`
- **XAI:** Coefficient-based feature importance
- **Pipeline:** Load → Clean → Split → Train → Evaluate → Plot

### Cell 3 (Markdown) — Section Label
> *"WITH BLACK_BOX MODEL and XAI (FEATURE IMPORTANCE → PDP)"*

### Cell 4 — RandomForest + Partial Dependence Plots
- **Model:** `RandomForestClassifier`
- **XAI:** `PartialDependenceDisplay.from_estimator` (sklearn)
- **Key Functions:**
  - `load_and_preprocess(filepath)` — dedup columns, drop non-predictive, coerce to numeric, fill NaN with 0
  - `split_features_target(df)` — stratified 80/20 split on `FLAG`
  - `train_random_forest(X_train, y_train)`
  - `evaluate_model(model, X_test, y_test)` — prints classification report + confusion matrix heatmap
  - `plot_feature_importance(model, feature_names, top_n=10)` → saves `feature_importance_plot.png`
  - `plot_partial_dependence_plots(model, X, features, top_n=3)` → saves `partial_dependence_plots.png`

### Cell 5 (Markdown) — Section Label
> *"WITH WHITE_BOX MODEL and XAI (FEATURE IMPORTANCE → PDP)"*

### Cell 6 — **Decision Tree + PDP + Tree Visualization** *(Primary/Final Model)*
- **Model:** `DecisionTreeClassifier`
- **Hyperparameters:**
  ```python
  DecisionTreeClassifier(
      max_depth=15,
      min_samples_split=20,
      min_samples_leaf=10,
      class_weight='balanced',
      random_state=42
  )
  ```
- **XAI:** Feature importance + Partial Dependence Plots + `plot_tree()`
- **Key Functions (same preprocessing helpers, plus):**
  - `train_decision_tree(X_train, y_train)`
  - `plot_tree_structure(model, feature_names, max_depth_to_show=3)` → saves `decision_tree_structure.png` at 300 DPI
- **Outputs:** `confusion_matrix_heatmap.png`, `feature_importance_plot.png`, `partial_dependence_plots.png`, `decision_tree_structure.png`

---

## 🔬 XAI Notebook: `FraudDetectionEthereum_(XAI).ipynb`

**2 cells — designed for Google Colab.**

- **Cell 0:** Commented-out `drive.mount('/content/drive')` — Colab setup
- **Cell 1:** Same Decision Tree pipeline as Cell 6 above (full standalone version)

> **Note:** This notebook appears to be a Colab-ready export of the final Decision Tree experiment.

---

## 🌐 Streamlit App: `streamlit-app/app.py`

**634 lines — production web application.**

### Architecture

```
EthereumFraudDetector (class)
├── FEATURE_NAMES: List[str]          # 43 feature definitions
├── __init__()                        # Trains synthetic model + creates SHAP explainer
├── _create_synthetic_model()         # Generates synthetic training data + trains DT
├── predict(features) → (prob, shap_vals, features)
└── get_base_value(explainer) → float

SAMPLE_PROFILES: Dict                 # 5 pre-configured test wallets

Helper Functions:
├── generate_fraud_alerts() → pd.DataFrame   # Mock live alert feed
├── create_gauge_chart(fraud_prob) → (go.Figure, risk_level)
└── generate_shap_explanation(shap_vals, features, feature_names, base_value) → str

main():
├── Sidebar: Profile selector + 8 collapsible feature input sections
├── Row 1: Plotly gauge chart + Key metrics + Quick stats
├── Row 2: SHAP waterfall plot (matplotlib) + Feature impact table
├── Row 3: Plain-English AI summary
└── Row 4: Live fraud alerts feed (mock)
```

### Model Details (Synthetic — used in App)

```python
# Training data: 10,000 synthetic samples
# - 7,900 legitimate (loc=typical legit values)
# - 2,100 fraudulent (loc=typical fraud values — high txn count, low avg times)
# np.abs() applied to clip negatives

DecisionTreeClassifier(
    max_depth=15, min_samples_split=20,
    min_samples_leaf=10, class_weight='balanced', random_state=42
)
```

> ⚠️ **App uses a synthetic model** — not the real trained model from the notebook. Pickle loading instructions are provided in the app's README for replacing it.

### Sample Profiles

| Profile | Expected Fraud Prob | Description |
|---|---|---|
| 🟢 Legitimate User | ~15-30% | Normal retail patterns |
| 🟡 High-Frequency Trader | Moderate | Active but legitimate |
| 🔴 Suspicious Wallet | ~70-85% | Rapid/unusual patterns |
| 🟠 Mixer/Tumbler Activity | High | Automated mixing service |
| 🟠 Wash Trading Pattern | High | Circular transaction patterns |

### Risk Thresholds

| Probability | Level | Color |
|---|---|---|
| < 30% | LOW RISK ✓ | Green |
| 30–60% | MEDIUM RISK ⚠️ | Orange |
| > 60% | HIGH RISK 🚨 | Red |

### SHAP Integration

- **Explainer:** `shap.TreeExplainer(model)`
- **Visualization:** `shap.waterfall_plot(shap.Explanation(...))`
- **Fallback:** Text-based feature impact table if waterfall plot fails
- **Top-5 features** shown in plain-English summary; **Top-8** in impact table

### UI Sections (Sidebar Expanders)

| Expander | Features | Indices |
|---|---|---|
| ⏱️ Temporal Features | 3 features | 0–2 |
| 📤 Transaction Counts | 3 features | 3–5 |
| 👥 Network Features | 2 features | 6–7 |
| 💰 Value (Received) | 3 features | 8–10 |
| 💸 Value (Sent) | 3 features | 11–13 |
| 🔗 Contract Transactions | 3 features | 14–16 |
| 📊 Total Aggregations | 5 features | 17–21 |
| 🪙 ERC20 Token Features | 21 features | 22–42 |

---

## ⚙️ Configuration

### `streamlit-app/streamlit_config.toml`

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
font = "sans serif"

[server]
headless = true
runOnSave = true
port = 8501
enableXsrfProtection = true
enableCORS = false
maxUploadSize = 200

[browser]
gatherUsageStats = false
```

---

## 📦 Dependencies (`streamlit-app/requirements.txt`)

| Package | Version | Role |
|---|---|---|
| `streamlit` | 1.28.1 | Web framework |
| `scikit-learn` | 1.3.2 | Decision Tree, PDP |
| `shap` | 0.43.0 | XAI / TreeExplainer |
| `numpy` | 1.24.3 | Numerical ops |
| `pandas` | 2.1.3 | Data handling |
| `scipy` | 1.11.4 | Scientific computing |
| `matplotlib` | 3.8.2 | Static plots |
| `plotly` | 5.18.0 | Interactive gauge |
| `seaborn` | 0.13.0 | Confusion matrix heatmap |
| `streamlit-option-menu` | 0.3.6 | UI component |

> Python 3.9–3.11 supported. All versions are pinned for reproducibility.

---

## 📈 Model Performance (Test Set: 1,969 samples)

```
              Precision  Recall  F1-Score
Not Fraud (0):  0.96     0.93     0.95
Fraud (1):      0.78     0.87     0.82

Overall Accuracy: 92%
```

---

## 🚀 How to Run

### Local Development
```bash
cd streamlit-app
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
streamlit run app.py
# Opens at http://localhost:8501
```

### Deploy Options
- **Streamlit Community Cloud** — push to GitHub, connect at share.streamlit.io
- **Hugging Face Spaces** — Docker-based, port 7860

---

## 🔄 Upgrade Path (Notebook → App Integration)

To use the real trained model in the app instead of the synthetic one:

```python
# In app.py → EthereumFraudDetector.__init__()
import pickle
with open('model.pkl', 'rb') as f:
    self.model = pickle.load(f)
self.explainer = shap.TreeExplainer(self.model)
```

---

## 📝 Notes & Observations

- The dataset file is named `etheruem_transaction_dataset.csv` (typo — missing 'h' in ethereum)
- The main notebook (`.ipynb`) references `./dataset/transaction_dataset.csv` in Cell 0, but `./dataset/etheruem_transaction_dataset.csv` in Cells 4 & 6 — **Cell 0 path may be stale/incorrect**
- The XAI notebook (`_XAI`) is essentially a Colab-ready duplicate of Cell 6 from the main notebook
- The Streamlit app trains a **synthetic** model on startup — it does not load `etheruem_transaction_dataset.csv`
- `DSML.pptx` is a presentation file (~5.4 MB) — not yet analyzed (requires external reader)
- `Outputs/` contains pre-generated plots from the notebook experiments
