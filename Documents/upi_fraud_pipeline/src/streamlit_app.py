"""
streamlit_app.py
-----------------
Interactive dashboard for the UPI fraud detection pipeline.

Run locally with:
    pip install streamlit plotly
    streamlit run streamlit_app.py

Expects the pipeline to have already been run (generate_data.py ->
sql_features.py -> train_model.py) so that data/features.csv,
outputs/flagged_transactions.csv, and outputs/model_metrics.json exist.
"""

import json
import pandas as pd
import streamlit as st
import plotly.express as px

DATA_DIR = "data"
OUT_DIR = "outputs"

st.set_page_config(page_title="UPI Fraud Detection", layout="wide")
st.title("🔍 Real-Time UPI Fraud & Anomaly Detection")
st.caption("Isolation Forest + Logistic Regression pipeline over engineered velocity, "
           "location, and amount-deviation features.")


@st.cache_data
def load_data():
    features = pd.read_csv(f"{DATA_DIR}/features.csv", parse_dates=["timestamp"])
    flagged = pd.read_csv(f"{OUT_DIR}/flagged_transactions.csv", parse_dates=["timestamp"])
    with open(f"{OUT_DIR}/model_metrics.json") as f:
        metrics = json.load(f)
    return features, flagged, metrics


features, flagged, metrics = load_data()

# ---------------- Top-line metrics ----------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Transactions analyzed", f"{metrics['n_transactions']:,}")
c2.metric("Fraud rate", f"{metrics['fraud_rate_pct']}%")
c3.metric("Model AUC-ROC", metrics["logistic_regression"]["auc_roc"])
c4.metric("FP reduction vs. rule baseline",
          f"{metrics['false_positive_comparison']['fp_reduction_pct']}%")

st.divider()

# ---------------- Sidebar filters ----------------
st.sidebar.header("Filters")
min_prob = st.sidebar.slider("Minimum fraud probability", 0.0, 1.0, 0.8, 0.01)
show_only_true_fraud = st.sidebar.checkbox("Show only confirmed fraud (ground truth)", False)

view = flagged[flagged["lr_fraud_probability"] >= min_prob]
if show_only_true_fraud:
    view = view[view["is_fraud"] == 1]

# ---------------- Flagged transactions table ----------------
st.subheader(f"🚩 Flagged Transactions ({len(view):,})")
st.dataframe(
    view[["transaction_id", "user_id", "timestamp", "amount",
          "txns_last_1min", "implied_speed_kmh", "amount_zscore",
          "lr_fraud_probability", "fraud_type"]]
    .sort_values("lr_fraud_probability", ascending=False),
    use_container_width=True,
    height=350,
)

# ---------------- Visuals ----------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Fraud probability distribution")
    fig = px.histogram(flagged, x="lr_fraud_probability", nbins=40,
                        color="fraud_type", title=None)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Flagged transactions by fraud type")
    counts = view["fraud_type"].value_counts().reset_index()
    counts.columns = ["fraud_type", "count"]
    fig2 = px.bar(counts, x="fraud_type", y="count", color="fraud_type")
    st.plotly_chart(fig2, use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    st.subheader("Velocity vs. implied travel speed")
    fig3 = px.scatter(
        flagged, x="txns_last_1min", y="implied_speed_kmh",
        color="fraud_type", opacity=0.6, log_y=True,
        labels={"txns_last_1min": "Transactions in last 1 min",
                "implied_speed_kmh": "Implied travel speed (km/h, log scale)"}
    )
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.subheader("Feature importance (Logistic Regression)")
    fi = pd.Series(metrics["feature_importance_lr_coefs"]).sort_values()
    fig4 = px.bar(fi, orientation="h",
                  labels={"value": "Standardized coefficient", "index": "Feature"})
    st.plotly_chart(fig4, use_container_width=True)

st.divider()
st.caption(
    f"Isolation Forest AUC (unsupervised, sanity check): "
    f"{metrics['isolation_forest']['auc_roc_vs_ground_truth']} | "
    f"Decision threshold: {metrics['logistic_regression']['decision_threshold']} | "
    f"Precision: {metrics['logistic_regression']['precision']} | "
    f"Recall: {metrics['logistic_regression']['recall']}"
)
