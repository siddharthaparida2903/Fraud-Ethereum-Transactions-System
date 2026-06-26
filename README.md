# 🛡️ Fraud Detection in Ethereum using Explainable AI (XAI)

[![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-success.svg?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/XAI-SHAP-orange.svg?style=for-the-badge)](https://shap.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Deployment-Streamlit-FF4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning pipeline and interactive web application designed to identify fraudulent Ethereum addresses. By leveraging advanced tree-based ensemble methods alongside Explainable AI (XAI) frameworks, this system moves beyond "black-box" predictions to deliver fully transparent, auditable fraud risk assessments in real-time.

---

## 🚀 Key Features

- **High-Performance Detection:** Utilizes an optimized **XGBoost Classifier** configured with cost-sensitive learning (`scale_pos_weight`) to manage the severe class imbalance typical of blockchain fraud data.
- **Explainable AI Integration:** Employs **SHAP (SHapley Additive exPlanations)** to break down individual predictions, showing exactly which features (e.g., token anomalies, transaction speed) drove the risk score.
- **Interactive Dashboard:** Built with **Streamlit** to feature custom user input profiles (e.g., Legitimate User vs. Phishing Wallet) and real-time prediction gauges.
- **Automated Insight Summaries:** Translates complex mathematical SHAP values into simple, plain-English summaries for fraud analysts and blockchain investigators.

---

## 🧠 System Architecture

```text
┌──────────────────────┐     ┌────────────────────────┐     ┌───────────────────────┐
│ Ethereum Account Data│ ──> │  XGBoost ML Pipeline   │ ──> │ SHAP Explainability   │
│ (Volumes, Freq, ERC20)│     │  (Imbalance Optimized) │     │ (Force/Waterfall Plots)│
└──────────────────────┘     └────────────────────────┘     └───────────────────────┘
                                                                        │
                                                                        ▼
                                                            ┌───────────────────────┐
                                                            │  Interactive Dashboard│
                                                            │   (Streamlit Live Web)│
                                                            └───────────────────────┘