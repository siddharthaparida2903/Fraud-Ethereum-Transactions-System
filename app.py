"""
================================================================================
ETHEREUM FRAUD DETECTION - PRODUCTION STREAMLIT APPLICATION
================================================================================
Real-time web application for detecting fraudulent Ethereum transactions
using Decision Tree classification and SHAP explainable AI visualizations.

Based on: Fraud Detection in Ethereum using Explainable AI
Dataset: Ethereum Transaction Features (47 features)
Model: Decision Tree Classifier (max_depth=15, balanced class weights)
Explainability: SHAP TreeExplainer with Waterfall Plots

Author: Fraud Detection Team
Version: 2.0.0 (Updated with Decision Tree & Real Features)
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import shap
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import plotly.graph_objects as go
from typing import Dict, Tuple, List
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Ethereum Fraud Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .fraud-alert {
        background-color: #ffcccc;
        border-left: 4px solid #ff0000;
        padding: 1rem;
        border-radius: 0.25rem;
    }
    .legit-safe {
        background-color: #ccffcc;
        border-left: 4px solid #00aa00;
        padding: 1rem;
        border-radius: 0.25rem;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# FEATURE DEFINITIONS & MODEL INITIALIZATION
# ============================================================================

class EthereumFraudDetector:
    """
    Production-ready Ethereum fraud detection system using Decision Tree.
    Encapsulates Decision Tree model, SHAP explainer, and prediction logic.
    Features extracted from: etheruem_transaction_dataset.csv (47 features)
    """
    
    # Exact feature definitions from the Ethereum transaction dataset
    # These are the 43 features used after dropping non-predictive columns
    FEATURE_NAMES = [
        'Avg min between sent tnx',
        'Avg min between received tnx',
        'Time Diff between first and last (Mins)',
        'Sent tnx',
        'Received Tnx',
        'Number of Created Contracts',
        'Unique Received From Addresses',
        'Unique Sent To Addresses',
        'min value received',
        'max value received',
        'avg val received',
        'min val sent',
        'max val sent',
        'avg val sent',
        'min value sent to contract',
        'max val sent to contract',
        'avg value sent to contract',
        'total transactions (including tnx to create contract',
        'total Ether sent',
        'total ether received',
        'total ether sent contracts',
        'total ether balance',
        'Total ERC20 tnxs',
        'ERC20 total Ether received',
        'ERC20 total ether sent',
        'ERC20 total Ether sent contract',
        'ERC20 uniq sent addr',
        'ERC20 uniq rec addr',
        'ERC20 uniq sent addr.1',
        'ERC20 uniq rec contract addr',
        'ERC20 avg time between sent tnx',
        'ERC20 avg time between rec tnx',
        'ERC20 avg time between rec 2 tnx',
        'ERC20 avg time between contract tnx',
        'ERC20 min val rec',
        'ERC20 max val rec',
        'ERC20 avg val rec',
        'ERC20 min val sent',
        'ERC20 max val sent',
        'ERC20 avg val sent',
        'ERC20 min val sent contract',
        'ERC20 max val sent contract',
        'ERC20 avg val sent contract'
    ]
    
    def __init__(self):
        """Initialize the detector with a synthetic but realistic Decision Tree model."""
        self.model = self._create_synthetic_model()
        self.explainer = shap.TreeExplainer(self.model)
        self.feature_names = self.FEATURE_NAMES
        
    def _create_synthetic_model(self) -> DecisionTreeClassifier:
        """
        Create a synthetic Decision Tree model trained on realistic fraud patterns.
        This uses the same hyperparameters as the original notebook:
        - max_depth=15
        - min_samples_split=20
        - min_samples_leaf=10
        - class_weight='balanced'
        """
        # Initialize Decision Tree with production-ready hyperparameters
        model = DecisionTreeClassifier(
            max_depth=15,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight='balanced',
            random_state=42
        )
        
        # Create synthetic training data with realistic patterns based on dataset statistics
        np.random.seed(42)
        n_samples = 10000
        
        # Generate legitimate transaction patterns (78.86% of dataset)
        legit_data = np.random.normal(
            loc=[1000, 1500, 150000, 100, 80, 0, 30, 50, 0.1, 20, 5,
                 0.05, 25, 8, 0.1, 10, 2, 200, 50, 30, 1, 50,
                 5, 3, 1, 2, 20, 15, 8, 5, 100, 50, 80, 60,
                 0.01, 0.5, 0.1, 0.01, 0.05, 0.01, 0.1, 0.05, 0.01],
            scale=[500, 800, 80000, 50, 40, 0, 15, 25, 0.05, 10, 2.5,
                   0.03, 15, 4, 0.05, 5, 1, 100, 30, 20, 0.5, 30,
                   3, 2, 0.5, 1, 10, 8, 4, 2, 50, 25, 40, 30,
                   0.005, 0.25, 0.05, 0.005, 0.03, 0.005, 0.05, 0.03, 0.005],
            size=(int(n_samples * 0.79), len(self.FEATURE_NAMES))
        )
        legit_labels = np.zeros(int(n_samples * 0.79))
        
        # Generate fraudulent transaction patterns (21.14% of dataset)
        fraud_data = np.random.normal(
            loc=[50, 100, 10000, 500, 600, 30, 200, 300, 0.001, 150, 50,
                 0.001, 500, 100, 0.001, 400, 50, 1000, 500, 300, 20, 500,
                 50, 100, 10, 30, 200, 150, 100, 60, 1000, 500, 800, 600,
                 0.1, 5, 1, 0.1, 0.5, 0.1, 1, 0.5, 0.1],
            scale=[30, 50, 5000, 200, 200, 15, 100, 150, 0.0005, 50, 20,
                   0.0005, 200, 40, 0.0005, 200, 25, 500, 250, 150, 10, 250,
                   25, 50, 5, 15, 100, 75, 50, 30, 500, 250, 400, 300,
                   0.05, 2.5, 0.5, 0.05, 0.25, 0.05, 0.5, 0.25, 0.05],
            size=(int(n_samples * 0.21), len(self.FEATURE_NAMES))
        )
        fraud_labels = np.ones(int(n_samples * 0.21))
        
        # Combine and train
        X_train = np.vstack([legit_data, fraud_data])
        y_train = np.hstack([legit_labels, fraud_labels])
        
        # Clip unrealistic negative values
        X_train = np.abs(X_train)
        
        model.fit(X_train, y_train)
        return model
    
    def predict(self, features: np.ndarray) -> Tuple[float, Dict]:
        """
        Generate fraud prediction and SHAP explanation for input features.
        
        Args:
            features: 1D numpy array of feature values
            
        Returns:
            Tuple of (fraud_probability, shap_values_dict)
        """
        # Reshape for single prediction
        features_reshaped = features.reshape(1, -1)
        
        # Get prediction probability
        fraud_probability = float(self.model.predict_proba(features_reshaped)[0, 1])
        
        # Calculate SHAP values for explainability
        shap_values = self.explainer.shap_values(features_reshaped)
        
        # For Decision Tree, shap_values is a single array for binary classification
        # Extract SHAP values for the fraud class (typically the positive class)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use fraud class SHAP values
        
        return fraud_probability, shap_values[0], features_reshaped[0]
    
    @staticmethod
    def get_base_value(explainer) -> float:
        """Get the base (expected) value from Decision Tree SHAP explainer."""
        return explainer.expected_value


# ============================================================================
# SAMPLE PROFILES FOR QUICK TESTING
# ============================================================================

SAMPLE_PROFILES = {
    "🟢 Legitimate User": {
        "description": "Regular retail user with normal transaction patterns",
        "values": [1200, 1800, 200000, 95, 75, 0, 35, 45, 0.15, 22, 6,
                   0.08, 28, 9, 0.12, 12, 3, 180, 60, 40, 2, 60,
                   8, 4, 2, 3, 25, 18, 10, 6, 150, 60, 100, 70,
                   0.02, 0.6, 0.15, 0.02, 0.06, 0.02, 0.12, 0.06, 0.02]
    },
    "🟡 High-Frequency Trader": {
        "description": "Active trader with frequent transactions, legitimate patterns",
        "values": [500, 600, 100000, 250, 200, 8, 100, 150, 0.08, 50, 20,
                   0.04, 80, 25, 0.06, 60, 15, 500, 200, 150, 10, 300,
                   25, 50, 8, 15, 80, 60, 30, 20, 400, 200, 300, 250,
                   0.08, 2, 0.4, 0.08, 0.2, 0.08, 0.4, 0.2, 0.08]
    },
    "🔴 Suspicious Wallet": {
        "description": "Flagged account: rapid pattern, unusual activity",
        "values": [80, 120, 15000, 450, 480, 40, 180, 250, 0.003, 90, 20,
                   0.001, 350, 40, 0.001, 300, 30, 800, 400, 200, 15, 450,
                   55, 120, 12, 25, 250, 180, 120, 70, 600, 350, 500, 400,
                   0.12, 4, 0.8, 0.12, 0.4, 0.12, 0.8, 0.4, 0.12]
    },
    "🟠 Mixer/Tumbler Activity": {
        "description": "Automated mixing service characteristics",
        "values": [50, 100, 8000, 550, 700, 20, 300, 400, 0.0001, 200, 50,
                   0.0001, 600, 60, 0.0001, 500, 50, 1200, 600, 400, 25, 650,
                   80, 180, 18, 40, 350, 250, 180, 100, 800, 450, 700, 550,
                   0.15, 5, 1, 0.15, 0.5, 0.15, 1, 0.5, 0.15]
    },
    "🟠 Wash Trading Pattern": {
        "description": "High circular transaction pattern (potentially suspicious)",
        "values": [200, 250, 50000, 600, 650, 25, 250, 280, 5, 120, 130,
                   0.05, 450, 70, 0.04, 420, 65, 1500, 500, 450, 18, 550,
                   60, 140, 15, 35, 280, 240, 160, 100, 700, 400, 600, 500,
                   0.1, 3.5, 0.7, 0.1, 0.35, 0.1, 0.7, 0.35, 0.1]
    }
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_fraud_alerts() -> pd.DataFrame:
    """Generate mock recent fraud alerts for live feed display."""
    now = datetime.now()
    alerts = []
    
    alert_types = ["High-volume transfers", "Mixing pattern detected", 
                   "Rapid account creation", "Gas price anomaly"]
    
    for i in range(15):
        timestamp = now - timedelta(minutes=np.random.randint(1, 120))
        alert_risk = np.random.choice(["🔴 Critical", "🟠 High", "🟡 Medium"])
        
        alerts.append({
            "Timestamp": timestamp.strftime("%H:%M:%S"),
            "Address": f"0x{np.random.randint(100000, 999999):06x}...",
            "Alert Type": np.random.choice(alert_types),
            "Risk Level": alert_risk,
            "Score": f"{np.random.uniform(0.65, 0.99):.2%}"
        })
    
    return pd.DataFrame(alerts)


def create_gauge_chart(fraud_probability: float) -> go.Figure:
    """
    Create an interactive gauge chart showing fraud risk probability.
    
    Args:
        fraud_probability: Fraud likelihood (0-1)
        
    Returns:
        Plotly gauge figure
    """
    # Determine color based on probability
    if fraud_probability < 0.3:
        color = "green"
        risk_level = "LOW RISK ✓"
    elif fraud_probability < 0.6:
        color = "orange"
        risk_level = "MEDIUM RISK ⚠️"
    else:
        color = "red"
        risk_level = "HIGH RISK 🚨"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=fraud_probability * 100,
        title={'text': "Fraud Risk Score"},
        delta={'reference': 50, 'suffix': "%"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 60], 'color': "lightyellow"},
                {'range': [60, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        },
        number={'suffix': "%"}
    ))
    
    fig.update_layout(
        height=300,
        font={'size': 14},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    return fig, risk_level


def generate_shap_explanation(shap_values: np.ndarray, features: np.ndarray, 
                              feature_names: List[str], base_value: float) -> str:
    """
    Convert SHAP values into human-readable explanation.
    
    Args:
        shap_values: SHAP values from explainer
        features: Input features
        feature_names: Names of features
        base_value: Base value from model
        
    Returns:
        Plain English explanation string
    """
    # Identify top contributing features
    abs_shap = np.abs(shap_values)
    top_indices = np.argsort(abs_shap)[-5:][::-1]  # Top 5 features
    
    explanation = "**Key Factors in This Prediction:**\n\n"
    
    for rank, idx in enumerate(top_indices, 1):
        feature_name = feature_names[idx]
        shap_value = shap_values[idx]
        feature_value = features[idx]
        direction = "⬆️ increases" if shap_value > 0 else "⬇️ decreases"
        
        explanation += f"{rank}. **{feature_name}** ({direction} fraud risk)\n"
        explanation += f"   - Current value: `{feature_value:.4f}`\n"
        explanation += f"   - Impact magnitude: `{abs_shap[idx]:.4f}`\n\n"
    
    # Add contextual interpretation
    if base_value < 0.3:
        baseline = "significantly below average fraud probability"
    elif base_value < 0.7:
        baseline = "near the average fraud probability"
    else:
        baseline = "significantly above average fraud probability"
    
    explanation += f"**Model Context:** The model's baseline expectation is {baseline}. "
    explanation += "The factors above push the prediction higher or lower from this baseline."
    
    return explanation


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main Streamlit application entry point."""
    
    # Initialize session state
    if 'detector' not in st.session_state:
        with st.spinner("⏳ Initializing XGBoost model and SHAP explainer..."):
            st.session_state.detector = EthereumFraudDetector()
    
    detector = st.session_state.detector
    
    # Header
    st.markdown("<h1 class='main-header'>🔍 Ethereum Fraud Detection System</h1>", 
                unsafe_allow_html=True)
    st.markdown(
        "**Real-time fraud detection powered by XGBoost + SHAP Explainable AI**"
    )
    st.markdown("---")
    
    # Sidebar: Input Panel
    with st.sidebar:
        st.header("📋 Input Panel")
        
        # Sample profile selector
        profile_selected = st.selectbox(
            "📊 Quick Start - Load Sample Profile:",
            list(SAMPLE_PROFILES.keys())
        )
        
        if profile_selected:
            profile_data = SAMPLE_PROFILES[profile_selected]
            st.info(f"**Profile:** {profile_data['description']}")
            default_values = profile_data["values"]
        else:
            default_values = [1.0] * len(detector.FEATURE_NAMES)
        
        # Feature input section
        st.subheader("🔧 Transaction Features")
        st.caption("Adjust values to see real-time fraud detection predictions")
        
        input_values = []
        
        # Organize features into collapsible sections
        with st.expander("⏱️ Temporal Features", expanded=True):
            for i in range(3):  # First 3 features are temporal
                input_values.append(st.slider(
                    detector.FEATURE_NAMES[i],
                    min_value=0.0, max_value=300000.0, value=float(default_values[i]),
                    step=100.0, key=f"feature_{i}"
                ))
        
        with st.expander("📤 Transaction Counts", expanded=True):
            for i in range(3, 6):  # Features 3-5: Sent/Received counts, Contracts
                input_values.append(st.slider(
                    detector.FEATURE_NAMES[i],
                    min_value=0, max_value=1000, value=int(default_values[i]),
                    step=10, key=f"feature_{i}"
                ))
        
        with st.expander("👥 Network Features", expanded=True):
            for i in range(6, 8):  # Features 6-7: Unique addresses
                input_values.append(st.slider(
                    detector.FEATURE_NAMES[i],
                    min_value=0, max_value=500, value=int(default_values[i]),
                    step=5, key=f"feature_{i}"
                ))
        
        with st.expander("💰 Value Features (ETH) - Received", expanded=True):
            for i in range(8, 11):  # Features 8-10: min/max/avg received
                input_values.append(st.slider(
                    detector.FEATURE_NAMES[i],
                    min_value=0.0, max_value=1000.0, value=float(default_values[i]),
                    step=0.1, key=f"feature_{i}"
                ))
        
        with st.expander("💸 Value Features (ETH) - Sent", expanded=True):
            for i in range(11, 14):  # Features 11-13: min/max/avg sent
                input_values.append(st.slider(
                    detector.FEATURE_NAMES[i],
                    min_value=0.0, max_value=1000.0, value=float(default_values[i]),
                    step=0.1, key=f"feature_{i}"
                ))
        
        with st.expander("🔗 Contract Transactions", expanded=False):
            for i in range(14, 17):  # Features 14-16: contract values
                input_values.append(st.slider(
                    detector.FEATURE_NAMES[i],
                    min_value=0.0, max_value=1000.0, value=float(default_values[i]),
                    step=0.1, key=f"feature_{i}"
                ))
        
        with st.expander("📊 Total Aggregations", expanded=True):
            for i in range(17, 22):  # Features 17-21: Total ether sent/received/balance
                input_values.append(st.slider(
                    detector.FEATURE_NAMES[i],
                    min_value=0.0, max_value=2000.0, value=float(default_values[i]),
                    step=1.0, key=f"feature_{i}"
                ))
        
        with st.expander("🪙 ERC20 Token Features", expanded=False):
            for i in range(22, len(detector.FEATURE_NAMES)):
                input_values.append(st.slider(
                    detector.FEATURE_NAMES[i],
                    min_value=0.0, max_value=2000.0, value=float(default_values[i]),
                    step=0.1, key=f"feature_{i}"
                ))
    
    # ========================================================================
    # MAIN CONTENT AREA
    # ========================================================================
    
    # Convert input to numpy array
    features_array = np.array(input_values, dtype=np.float32)
    
    # Get prediction and SHAP explanation
    fraud_prob, shap_vals, input_features = detector.predict(features_array)
    base_value = detector.get_base_value(detector.explainer)
    
    # Row 1: Prediction Gauge & Key Metrics
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.subheader("📊 Fraud Risk Assessment")
        gauge_fig, risk_level = create_gauge_chart(fraud_prob)
        st.plotly_chart(gauge_fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Key Metrics")
        
        if fraud_prob < 0.3:
            st.success(f"**Risk Level:** {risk_level}")
        elif fraud_prob < 0.6:
            st.warning(f"**Risk Level:** {risk_level}")
        else:
            st.error(f"**Risk Level:** {risk_level}")
        
        st.metric("Fraud Probability", f"{fraud_prob:.2%}")
        st.metric("Confidence", f"{(1 - abs(0.5 - fraud_prob)) * 2:.0%}")
        st.metric("Assessment", 
                 "FLAGGED 🚨" if fraud_prob > 0.7 else "MONITORED ⚠️" if fraud_prob > 0.4 else "CLEAN ✓")
    
    with col3:
        st.subheader("🎯 Quick Stats")
        st.info(f"""
        **Timestamp:** {datetime.now().strftime('%H:%M:%S')}
        
        **Transactions:** {int(input_features[3] + input_features[4])}
        
        **Age (hours):** {int(input_features[2])}
        
        **Unique Contacts:** {int(input_features[6] + input_features[7])}
        """)
    
    st.markdown("---")
    
    # Row 2: SHAP Explanation
    st.subheader("🧠 Explainable AI - SHAP Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**SHAP Waterfall Plot**")
        st.caption("Shows how each feature contributes to the fraud prediction")
        
        # Create SHAP waterfall visualization
        try:
            # Create explanation object for waterfall
            explainer_data = shap.Explanation(
                values=shap_vals,
                base_values=base_value,
                data=input_features,
                feature_names=detector.FEATURE_NAMES
            )
            
            # Use matplotlib for SHAP waterfall
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.waterfall_plot(explainer_data, show=False)
            st.pyplot(fig, use_container_width=True)
            plt.close()
        except Exception as e:
            st.warning(f"Waterfall visualization: {str(e)}")
            
            # Fallback to text-based explanation
            st.info("📊 Displaying top contributing features...")
    
    with col2:
        st.markdown("**Feature Impact Summary**")
        st.caption("Text explanation of SHAP values")
        
        # Get top contributing features
        abs_shap = np.abs(shap_vals)
        top_indices = np.argsort(abs_shap)[-8:][::-1]
        
        # Create impact DataFrame
        impact_data = []
        for idx in top_indices:
            impact_data.append({
                "Feature": detector.FEATURE_NAMES[idx],
                "Value": f"{input_features[idx]:.4f}",
                "SHAP Impact": f"{shap_vals[idx]:+.4f}",
                "Direction": "📈 Fraud↑" if shap_vals[idx] > 0 else "📉 Legit↑"
            })
        
        impact_df = pd.DataFrame(impact_data)
        st.dataframe(impact_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Row 3: Plain English Explanation
    st.subheader("📝 AI Summary")
    explanation_text = generate_shap_explanation(
        shap_vals, input_features, detector.FEATURE_NAMES, base_value
    )
    st.markdown(explanation_text)
    
    st.markdown("---")
    
    # Row 4: Live Alerts Log
    st.subheader("🚨 Live Fraud Alerts Feed")
    st.caption("Mock real-time feed of recently flagged suspicious addresses")
    
    alerts_df = generate_fraud_alerts()
    st.dataframe(alerts_df, use_container_width=True, hide_index=True)
    
    # Footer
    st.markdown("---")
    st.caption(
        "🔐 **Note:** This is a demonstration system. "
        "Predictions are based on synthetic training data. "
        "Integrate with real Ethereum data and model artifacts for production use."
    )


if __name__ == "__main__":
    main()
