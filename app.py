
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Manufacturing Quality Intelligence",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "quality_model.joblib"
METRICS_PATH = BASE_DIR / "outputs" / "metrics.json"
PREDICTIONS_PATH = BASE_DIR / "outputs" / "test_predictions.csv"
IMPORTANCE_PATH = BASE_DIR / "outputs" / "feature_importance.csv"
CONFUSION_PATH = BASE_DIR / "outputs" / "confusion_matrix.csv"

# --------------------------------------------------
# 2. PROFESSIONAL DASHBOARD STYLING
# --------------------------------------------------

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128, 128, 128, 0.25);
        padding: 15px;
        border-radius: 10px;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem;
    }
    .dashboard-subtitle {
        color: #808895;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# 3. LOAD PROJECT OUTPUTS
# --------------------------------------------------

@st.cache_data
def load_metrics(path, modified):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_predictions(path, modified):
    return pd.read_csv(path)


@st.cache_data
def load_importance(path, modified):
    return pd.read_csv(path)


def load_confusion_matrix(path):
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0)


required_files = [
    MODEL_PATH,
    METRICS_PATH,
    PREDICTIONS_PATH,
    IMPORTANCE_PATH,
]

missing_files = [
    str(path.relative_to(BASE_DIR))
    for path in required_files
    if not path.exists()
]

if missing_files:
    st.error("Required project outputs are missing.")
    st.write("Run the training script first:")
    st.code("python train_model.py")
    st.write("Missing files:")
    for file in missing_files:
        st.write(f"- `{file}`")
    st.stop()

try:
    model_bundle = joblib.load(MODEL_PATH)

    # Supports the bundled model produced by the revised
    # train_model.py and the older pipeline-only format.
    if isinstance(model_bundle, dict):
        model = model_bundle["model"]
        saved_threshold = float(
            model_bundle.get("threshold", 0.5)
        )
    else:
        model = model_bundle
        saved_threshold = 0.5

    metrics = load_metrics(
        METRICS_PATH,
        METRICS_PATH.stat().st_mtime,
    )

    predictions_df = load_predictions(
        PREDICTIONS_PATH,
        PREDICTIONS_PATH.stat().st_mtime,
    )

    importance_df = load_importance(
        IMPORTANCE_PATH,
        IMPORTANCE_PATH.stat().st_mtime,
    )

except Exception as error:
    st.error(f"Unable to load project outputs: {error}")
    st.stop()

required_prediction_columns = {
    "actual",
    "failure_probability",
}

if not required_prediction_columns.issubset(
    predictions_df.columns
):
    st.error(
        "test_predictions.csv must contain actual and "
        "failure_probability columns. Rerun train_model.py."
    )
    st.stop()

predictions_df["actual"] = pd.to_numeric(
    predictions_df["actual"], errors="coerce"
)

predictions_df["failure_probability"] = pd.to_numeric(
    predictions_df["failure_probability"],
    errors="coerce",
)

predictions_df = predictions_df.dropna(
    subset=["actual", "failure_probability"]
).copy()

predictions_df["actual"] = predictions_df["actual"].astype(int)

if not predictions_df["actual"].isin([0, 1]).all():
    st.error("Actual labels must be 0 (Pass) or 1 (Fail).")
    st.stop()

if predictions_df.empty:
    st.error("No valid test predictions were found.")
    st.stop()

# --------------------------------------------------
# 4. SIDEBAR CONTROLS
# --------------------------------------------------

st.sidebar.title("🏭 Quality Intelligence")
st.sidebar.caption("Manufacturing analytics and defect prediction")

st.sidebar.markdown("---")
st.sidebar.subheader("Dashboard settings")

min_threshold = 0.05
max_threshold = 0.50

default_threshold = float(
    np.clip(saved_threshold, min_threshold, max_threshold)
)

threshold = st.sidebar.slider(
    "Failure-risk threshold",
    min_value=min_threshold,
    max_value=max_threshold,
    value=default_threshold,
    step=0.01,
    help=(
        "A unit is flagged when its predicted failure "
        "probability is at least this value. This changes "
        "the test-results display, not the trained model."
    ),
)

st.sidebar.caption(
    "Lower thresholds generally flag more units, which "
    "may increase both failure detection and false alarms."
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Project information**")
st.sidebar.write("Dataset: UCI SECOM")
st.sidebar.write("Model: Random Forest")
st.sidebar.write(f"Sensor features: {metrics['sensor_features']}")
st.sidebar.write(
    f"Test observations: {len(predictions_df):,}"
)

# --------------------------------------------------
# 5. INTERACTIVE TEST PREDICTIONS
# --------------------------------------------------

df = predictions_df.copy()

df["predicted"] = (
    df["failure_probability"] >= threshold
).astype(int)

df["actual_label"] = df["actual"].map({
    0: "Pass",
    1: "Fail",
})

df["predicted_label"] = df["predicted"].map({
    0: "Pass",
    1: "Flagged",
})

df["risk_category"] = pd.cut(
    df["failure_probability"],
    bins=[-0.001, 0.10, 0.25, 0.50, 1.001],
    labels=[
        "Low",
        "Moderate",
        "High",
        "Very High",
    ],
    include_lowest=True,
)

# --------------------------------------------------
# 6. HEADER
# --------------------------------------------------

st.title("Manufacturing Quality Intelligence")
st.markdown(
    '<p class="dashboard-subtitle">'
    'Machine learning-based defect risk analysis '
    'for semiconductor manufacturing observations.'
    '</p>',
    unsafe_allow_html=True,
)

st.caption(
    "Metrics and inspection records below refer to the "
    "held-out test dataset unless explicitly stated otherwise."
)

st.markdown("---")

# --------------------------------------------------
# 7. KPI CARDS
# --------------------------------------------------

total_units = len(df)
actual_failures = int(df["actual"].sum())
flagged_units = int(df["predicted"].sum())

true_positives = int(
    ((df["actual"] == 1) & (df["predicted"] == 1)).sum()
)

false_negatives = int(
    ((df["actual"] == 1) & (df["predicted"] == 0)).sum()
)

false_positives = int(
    ((df["actual"] == 0) & (df["predicted"] == 1)).sum()
)

actual_failure_rate = df["actual"].mean() * 100
flagged_rate = df["predicted"].mean() * 100

k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "Test observations",
    f"{total_units:,}",
)

k2.metric(
    "Actual failures",
    f"{actual_failures:,}",
    f"{actual_failure_rate:.2f}% of test set",
)

k3.metric(
    "Flagged for inspection",
    f"{flagged_units:,}",
    f"{flagged_rate:.2f}% of test set",
)

k4.metric(
    "Missed failures",
    f"{false_negatives:,}",
    "Actual failures predicted as pass",
    delta_color="inverse",
)

st.markdown("---")

# --------------------------------------------------
# 8. MODEL PERFORMANCE
# --------------------------------------------------

st.subheader("Model performance")

y_true = df["actual"].to_numpy()
y_pred = df["predicted"].to_numpy()
y_prob = df["failure_probability"].to_numpy()

accuracy = accuracy_score(y_true, y_pred)
balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0,
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0,
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0,
)

if len(np.unique(y_true)) == 2:
    roc_auc = roc_auc_score(y_true, y_prob)
else:
    roc_auc = float("nan")

m1, m2, m3, m4, m5 = st.columns(5)

m1.metric("Accuracy", f"{accuracy:.1%}")
m2.metric("Balanced accuracy", f"{balanced_accuracy:.1%}")
m3.metric("Failure precision", f"{precision:.1%}")
m4.metric("Failure recall", f"{recall:.1%}")
m5.metric("Failure F1-score", f"{f1:.3f}")

st.caption(
    f"ROC-AUC: {roc_auc:.3f} | "
    f"Saved training threshold: {saved_threshold:.2f} | "
    f"Dashboard threshold: {threshold:.2f}"
)

if recall == 0:
    st.warning(
        "No actual failures are detected at the current "
        "threshold. Consider a lower threshold and review "
        "the resulting false alarms."
    )
elif false_negatives > 0:
    st.warning(
        f"The current threshold misses {false_negatives} "
        "actual failures. Review false negatives before "
        "using this model for inspection decisions."
    )
else:
    st.success(
        "All failures in this test set are flagged at the "
        "current threshold. Check the number of false alarms "
        "before drawing conclusions."
    )

# --------------------------------------------------
# 9. CONFUSION MATRIX
# --------------------------------------------------

st.markdown("---")
st.subheader("Confusion matrix")

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1],
)

cm_display = pd.DataFrame(
    cm,
    index=["Actual Pass", "Actual Fail"],
    columns=["Predicted Pass", "Predicted Fail"],
)

left, right = st.columns([1.1, 1])

with left:
    fig_cm = px.imshow(
        cm,
        x=["Predicted Pass", "Predicted Fail"],
        y=["Actual Pass", "Actual Fail"],
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        labels={
            "x": "Prediction",
            "y": "Actual result",
            "color": "Observations",
        },
        title="Test-set classification results",
    )

    fig_cm.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    st.plotly_chart(
        fig_cm,
        use_container_width=True,
    )

with right:
    st.markdown("**Interpretation**")
    st.write(
        f"- Correctly identified failures: **{true_positives}**"
    )
    st.write(
        f"- Missed failures: **{false_negatives}**"
    )
    st.write(
        f"- Passing units flagged: **{false_positives}**"
    )
    st.write(
        f"- Correctly identified passes: **{int(cm[0, 0])}**"
    )

    st.dataframe(
        cm_display,
        use_container_width=True,
    )

    st.caption(
        "A false negative is particularly important in "
        "quality inspection because a failed unit may "
        "be incorrectly classified as passing."
    )

# --------------------------------------------------
# 10. FAILURE PROBABILITY DISTRIBUTION
# --------------------------------------------------

st.markdown("---")
st.subheader("Failure-risk distribution")

risk_left, risk_right = st.columns(2)

with risk_left:
    fig_hist = px.histogram(
        df,
        x="failure_probability",
        color="actual_label",
        nbins=30,
        barmode="overlay",
        opacity=0.75,
        color_discrete_map={
            "Pass": "#2878B5",
            "Fail": "#D9534F",
        },
        title="Predicted probabilities by actual result",
        labels={
            "failure_probability": "Predicted failure probability",
            "actual_label": "Actual result",
        },
    )

    fig_hist.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Threshold {threshold:.2f}",
    )

    fig_hist.update_layout(height=390)

    st.plotly_chart(
        fig_hist,
        use_container_width=True,
    )

with risk_right:
    risk_counts = (
        df["risk_category"]
        .astype("object")
        .fillna("Unknown")
        .value_counts()
        .reindex(
            ["Low", "Moderate", "High", "Very High", "Unknown"],
            fill_value=0,
        )
        .rename_axis("Risk category")
        .reset_index(name="Observations")
    )

    fig_risk = px.bar(
        risk_counts,
        x="Risk category",
        y="Observations",
        color="Risk category",
        title="Observations by probability band",
        color_discrete_map={
            "Low": "#2E8B57",
            "Moderate": "#E6B655",
            "High": "#E67E22",
            "Very High": "#C0392B",
            "Unknown": "#808080",
        },
    )

    fig_risk.update_layout(
        height=390,
        showlegend=False,
    )

    st.plotly_chart(
        fig_risk,
        use_container_width=True,
    )

st.caption(
    "Risk bands are descriptive probability ranges, not "
    "validated operational risk categories."
)

# --------------------------------------------------
# 11. SENSOR FEATURE IMPORTANCE
# --------------------------------------------------

st.markdown("---")
st.subheader("Sensor feature importance")

st.write(
    "These features were selected during training. "
    "Random Forest importance indicates predictive usefulness "
    "within this fitted model; it does not establish causation."
)

if (
    not importance_df.empty
    and {"sensor_feature", "importance"}.issubset(
        importance_df.columns
    )
):
    top_n = st.slider(
        "Number of sensor features to display",
        min_value=5,
        max_value=min(40, len(importance_df)),
        value=min(15, len(importance_df)),
        step=1,
    )

    top_features = importance_df.nlargest(
        top_n,
        "importance",
    ).sort_values("importance", ascending=True)

    fig_importance = px.bar(
        top_features,
        x="importance",
        y="sensor_feature",
        orientation="h",
        title=f"Top {top_n} selected sensor features",
        labels={
            "importance": "Relative feature importance",
            "sensor_feature": "Sensor",
        },
    )

    fig_importance.update_layout(
        height=max(420, top_n * 27),
        showlegend=False,
    )

    st.plotly_chart(
        fig_importance,
        use_container_width=True,
    )

    st.download_button(
        "Download feature importance CSV",
        data=importance_df.to_csv(index=False).encode("utf-8"),
        file_name="feature_importance.csv",
        mime="text/csv",
    )
else:
    st.info(
        "Feature importance data is unavailable or has an "
        "unexpected format. Run train_model.py again."
    )

# --------------------------------------------------
# 12. ROC CURVE
# --------------------------------------------------

st.markdown("---")
st.subheader("ROC curve")

if len(np.unique(y_true)) == 2:
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    fig_roc = go.Figure()

    fig_roc.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"Model ROC-AUC = {roc_auc:.3f}",
            line=dict(width=3),
        )
    )

    fig_roc.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random classifier reference",
            line=dict(dash="dash"),
        )
    )

    fig_roc.update_layout(
        title="Discrimination across probability thresholds",
        xaxis_title="False positive rate",
        yaxis_title="True positive rate",
        height=420,
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
    )

    st.plotly_chart(
        fig_roc,
        use_container_width=True,
    )
else:
    st.info(
        "ROC curve requires both passing and failing examples."
    )

# --------------------------------------------------
# 13. INSPECTION EXPLORER
# --------------------------------------------------

st.markdown("---")
st.subheader("Inspection explorer")

st.write(
    "Explore the held-out test records. These are historical "
    "test observations, not live production sensor readings."
)

filter_col1, filter_col2, filter_col3 = st.columns(3)

with filter_col1:
    actual_filter = st.selectbox(
        "Actual result",
        ["All", "Pass", "Fail"],
    )

with filter_col2:
    prediction_filter = st.selectbox(
        "Prediction",
        ["All", "Pass", "Flagged"],
    )

with filter_col3:
    risk_filter = st.selectbox(
        "Probability band",
        ["All", "Low", "Moderate", "High", "Very High"],
    )

filtered = df.copy()

if actual_filter != "All":
    expected_actual = 1 if actual_filter == "Fail" else 0
    filtered = filtered[
        filtered["actual"] == expected_actual
    ]

if prediction_filter != "All":
    expected_prediction = (
        1 if prediction_filter == "Flagged" else 0
    )
    filtered = filtered[
        filtered["predicted"] == expected_prediction
    ]

if risk_filter != "All":
    filtered = filtered[
        filtered["risk_category"].astype(str) == risk_filter
    ]

# Highest risk first.
filtered = filtered.sort_values(
    "failure_probability",
    ascending=False,
).copy()

# Add an observation number for convenient reference.
filtered.insert(
    0,
    "test_observation",
    filtered.index + 1,
)

display_columns = [
    "test_observation",
    "actual_label",
    "predicted_label",
    "failure_probability",
    "risk_category",
]

if "timestamp" in filtered.columns:
    display_columns.append("timestamp")

st.caption(f"Showing {len(filtered):,} matching observations")

st.dataframe(
    filtered[display_columns],
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "Download filtered inspection results",
    data=filtered[display_columns].to_csv(
        index=False
    ).encode("utf-8"),
    file_name="inspection_results.csv",
    mime="text/csv",
)

# --------------------------------------------------
# 14. MODEL METADATA AND LIMITATIONS
# --------------------------------------------------

with st.expander("Model metadata and limitations"):
    st.json(metrics)

    st.markdown(
        """
        **Important limitations**

        - The UCI SECOM dataset is historical semiconductor
          manufacturing data, not a live production feed.
        - Sensor importance does not demonstrate causation.
        - The classification threshold changes how probabilities
          are converted into flags; it does not retrain the model.
        - The dashboard's interactive threshold metrics are
          calculated on the held-out test set for exploration.
          Do not use this test set repeatedly to optimize the
          model and report the same results as independent
          final performance.
        - Before deployment, assess false-negative costs,
          probability calibration, time-based validation,
          process drift, and the consequences of inspection
          decisions with domain experts.
        """
    )

st.markdown("---")
st.caption(
    "Manufacturing Quality Intelligence | "
    "Python · Scikit-learn · Plotly · Streamlit"
)
