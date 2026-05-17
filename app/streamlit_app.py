from __future__ import annotations

import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from churn_utils import METADATA_PATH, MODEL_PATH


st.set_page_config(
    page_title="ChurnPredict AI",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --bg-1: #06101d;
        --bg-2: #0c1930;
        --panel: rgba(13, 22, 39, 0.86);
        --panel-strong: rgba(18, 29, 50, 0.96);
        --text: #edf4ff;
        --muted: #a8b8d1;
        --accent-1: #7c3aed;
        --accent-2: #06b6d4;
        --accent-3: #f97316;
        --accent-4: #10b981;
        --danger: #ef4444;
        --warning: #f59e0b;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(124, 58, 237, 0.24), transparent 30%),
            radial-gradient(circle at top right, rgba(6, 182, 212, 0.20), transparent 28%),
            radial-gradient(circle at bottom left, rgba(249, 115, 22, 0.12), transparent 24%),
            linear-gradient(180deg, var(--bg-1) 0%, var(--bg-2) 100%);
        color: var(--text);
    }

    .main {
        padding: 1rem 1.5rem 2rem 1.5rem;
    }

    .block-container {
        padding-top: 1.25rem;
        max-width: 1500px;
    }

    .header-gradient {
        background:
            linear-gradient(135deg, rgba(124, 58, 237, 0.95) 0%, rgba(6, 182, 212, 0.9) 48%, rgba(249, 115, 22, 0.82) 100%);
        padding: 2.4rem 2.2rem;
        border-radius: 2rem;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 24px 60px rgba(0, 0, 0, 0.34);
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.16);
    }

    .header-gradient::after {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(135deg, rgba(255,255,255,0.18), transparent 34%, rgba(255,255,255,0.05));
        pointer-events: none;
    }

    .header-gradient h1 {
        margin: 0;
        font-size: 2.9rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        line-height: 1.05;
    }

    .header-gradient p {
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
        opacity: 0.92;
    }

    .hero-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        margin-top: 1rem;
        padding: 0.5rem 0.9rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.14);
        border: 1px solid rgba(255, 255, 255, 0.18);
        color: white;
        font-weight: 600;
        backdrop-filter: blur(10px);
    }

    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        color: #eff6ff;
        border-left: 4px solid var(--accent-2);
        padding-left: 1rem;
    }

    .prediction-card {
        padding: 2rem;
        border-radius: 1.5rem;
        color: white;
        box-shadow: 0 18px 40px rgba(0,0,0,0.26);
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.14);
        backdrop-filter: blur(12px);
    }

    .prediction-card h3 { margin: 0; font-size: 1.2rem; opacity: 0.9; }
    .prediction-card .status { font-size: 2rem; margin: 0.5rem 0; font-weight: 700; }
    .prediction-card .label { font-size: 1.8rem; font-weight: 700; margin: 0.5rem 0 0 0; }

    .result-panel {
        background: linear-gradient(180deg, rgba(8, 15, 29, 0.88), rgba(12, 22, 39, 0.96));
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 1.6rem;
        padding: 1.25rem;
        box-shadow: 0 22px 48px rgba(0, 0, 0, 0.28);
        margin-bottom: 1rem;
    }

    .result-hero {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        flex-wrap: wrap;
    }

    .result-title {
        font-size: 1.7rem;
        font-weight: 800;
        margin: 0;
        color: white;
    }

    .result-subtitle {
        margin: 0.25rem 0 0 0;
        color: var(--muted);
    }

    .pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.5rem 0.85rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.9rem;
        border: 1px solid rgba(255, 255, 255, 0.12);
    }

    .form-container {
        background: linear-gradient(180deg, rgba(12, 20, 36, 0.82) 0%, rgba(8, 14, 26, 0.9) 100%);
        padding: 2rem;
        border-radius: 1.8rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin: 1rem 0 2rem 0;
        box-shadow: 0 20px 48px rgba(0, 0, 0, 0.24);
    }

    .stButton > button {
        background: linear-gradient(135deg, #7c3aed 0%, #06b6d4 52%, #f97316 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 0.9rem !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        box-shadow: 0 10px 24px rgba(6, 182, 212, 0.24) !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 16px 30px rgba(124, 58, 237, 0.3) !important;
    }

    .metric-box {
        background: linear-gradient(180deg, rgba(18, 27, 44, 0.96), rgba(12, 20, 35, 0.98));
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 4px solid var(--accent-2);
        box-shadow: 0 12px 28px rgba(0,0,0,0.22);
        border: 1px solid rgba(255,255,255,0.06);
    }

    .metric-label {
        font-size: 0.8rem;
        color: var(--muted);
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--text);
    }

    .info-panel {
        background: linear-gradient(180deg, rgba(17, 26, 43, 0.96), rgba(10, 18, 35, 0.94));
        border-radius: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 1rem;
        color: var(--text);
    }

    .footer {
        text-align: center;
        color: var(--muted);
        font-size: 0.875rem;
        margin-top: 3rem;
        padding-top: 2rem;
        border-top: 1px solid rgba(255,255,255,0.1);
    }

    .stDataFrame, .stDataFrame [data-testid="stTable"] {
        border-radius: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="header-gradient">
        <h1>ChurnPredict AI</h1>
        <p>Intelligent customer churn prediction and risk analysis engine</p>
        <div class="hero-pill">Advanced visual analytics | live percentage chart | SHAP explanation</div>
    </div>
    """,
    unsafe_allow_html=True,
)


def fmt_number(value: object) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "N/A"


def build_percentage_donut(probability: float, risk_color: str, label: str):
    fig, ax = plt.subplots(figsize=(4.2, 4.2), subplot_kw={"aspect": "equal"})
    fig.patch.set_facecolor("#0b1324")
    ax.set_facecolor("#0b1324")
    remaining = max(0.0, 1.0 - probability)
    ax.pie(
        [probability, remaining],
        startangle=90,
        counterclock=False,
        colors=[risk_color, "#1f2a44"],
        wedgeprops={"width": 0.28, "edgecolor": "#0b1324", "linewidth": 2},
    )
    ax.text(0, 0.1, f"{probability:.0%}", ha="center", va="center", fontsize=24, fontweight="bold", color="white")
    ax.text(0, -0.15, label, ha="center", va="center", fontsize=10.5, color="#cbd5e1")
    ax.set_title("Churn percentage", fontsize=12, color="white", pad=14, fontweight="bold")
    ax.axis("off")
    return fig


@st.cache_resource
def load_model():
    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


@st.cache_data
def load_metadata() -> dict:
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.error("Missing trained model artifacts. Run app/train_model.py first to build the pipeline.")
    st.stop()


model = load_model()
metadata = load_metadata()
feature_specs = metadata["features"]


DISPLAY_LABELS = {
    "Gender": "Gender",
    "Senior Citizen": "Senior citizen",
    "Partner": "Has partner",
    "Dependents": "Has dependents",
    "Tenure Months": "Tenure with company (months)",
    "Phone Service": "Phone service",
    "Multiple Lines": "Multiple phone lines",
    "Internet Service": "Internet service type",
    "Online Security": "Online security",
    "Online Backup": "Online backup",
    "Device Protection": "Device protection",
    "Tech Support": "Tech support",
    "Streaming TV": "Streaming TV",
    "Streaming Movies": "Streaming movies",
    "Contract": "Contract type",
    "Paperless Billing": "Paperless billing",
    "Payment Method": "Payment method",
    "Monthly Charges": "Monthly charges ($)",
    "Total Charges": "Total charges ($)",
    "CLTV": "Customer lifetime value",
}

FIELD_HINTS = {
    "Gender": "Customer gender as recorded by the telecom provider.",
    "Senior Citizen": "Choose Yes if the customer is 65 or older.",
    "Partner": "Whether the customer has a partner/spouse.",
    "Dependents": "Whether the customer has dependents.",
    "Tenure Months": "How long the customer has been with the company.",
    "Phone Service": "Whether the customer has phone service.",
    "Multiple Lines": "Whether multiple phone lines are active.",
    "Internet Service": "Type of internet service the customer uses.",
    "Online Security": "Whether online security is included.",
    "Online Backup": "Whether online backup is included.",
    "Device Protection": "Whether device protection is included.",
    "Tech Support": "Whether technical support is included.",
    "Streaming TV": "Whether streaming TV is included.",
    "Streaming Movies": "Whether streaming movies is included.",
    "Contract": "Current contract duration.",
    "Paperless Billing": "Whether the customer uses paperless billing.",
    "Payment Method": "How the customer usually pays.",
    "Monthly Charges": "Current monthly bill amount.",
    "Total Charges": "Total charged so far.",
    "CLTV": "Estimated lifetime value score.",
}

FIELD_GROUPS = [
    ("Customer Profile", ["Gender", "Senior Citizen", "Partner", "Dependents"]),
    ("Services", ["Phone Service", "Multiple Lines", "Internet Service", "Online Security", "Online Backup", "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies"]),
    ("Contract & Billing", ["Contract", "Paperless Billing", "Payment Method"]),
    ("Value & Tenure", ["Tenure Months", "Monthly Charges", "Total Charges", "CLTV"]),
]


with st.sidebar:
    st.markdown("### Model Information")
    st.markdown(
        f"""
        <div class="info-panel">
            <div><strong>Model Performance</strong></div>
            <div>Cross-Val ROC-AUC: {fmt_number(metadata.get('cv_mean'))}</div>
            <div>Test ROC-AUC: {fmt_number(metadata.get('test_roc_auc'))}</div>
            <div style="margin-top:0.6rem;">Total Features: {len(feature_specs)}</div>
            <div>Algorithm: Random Forest (200 estimators)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("### Advanced Controls")
    threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, step=0.01)

    theme = st.radio("Theme", ["Aurora", "Midnight"], index=0)
    if theme == "Midnight":
        st.markdown(
            """
            <style>
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(40, 80, 180, 0.18), transparent 28%),
                    radial-gradient(circle at top right, rgba(6, 182, 212, 0.15), transparent 24%),
                    linear-gradient(180deg, #030816 0%, #08111f 100%);
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("### Presets & Tips")
    presets = {
        "Default": {},
        "Example - High Risk": {
            "Contract": "Month-to-month",
            "Tenure Months": 2,
            "Monthly Charges": 110.0,
            "Internet Service": "Fiber optic",
        },
        "Example - Low Risk": {
            "Contract": "Two year",
            "Tenure Months": 48,
            "Monthly Charges": 50.0,
            "Internet Service": "DSL",
        },
    }
    preset_key = st.selectbox("Select preset", list(presets.keys()))
    if st.button("Apply preset"):
        for key, value in presets[preset_key].items():
            st.session_state[key] = value

    st.divider()
    st.markdown("### Quick Tips")
    st.caption("• Month-to-month contracts have higher churn risk")
    st.caption("• High monthly charges increase churn likelihood")
    st.caption("• Long tenure customers are more loyal")
    st.caption("• Premium services reduce churn risk")


st.markdown('<div class="form-container">', unsafe_allow_html=True)

col1, col2 = st.columns([1, 3])
with col1:
    st.markdown("### Customer Info")
with col2:
    st.write("Choose the values that match the customer. The model uses these details to estimate churn risk.")
    st.caption("Tip: If you are unsure, use the preset examples in the sidebar to see the form populated automatically.")

with st.form("churn_form"):
    input_values: dict[str, object] = {}
    spec_map = {spec["name"]: spec for spec in feature_specs}

    for section_name, field_names in FIELD_GROUPS:
        st.markdown(f"#### {section_name}")
        section_columns = st.columns(2)

        for index, field_name in enumerate(field_names):
            spec = spec_map[field_name]
            target_column = section_columns[index % 2]
            with target_column:
                label = DISPLAY_LABELS.get(field_name, field_name)
                help_text = FIELD_HINTS.get(field_name, "")
                if spec["kind"] == "numeric":
                    input_values[field_name] = st.number_input(
                        label,
                        min_value=float(spec["min"]),
                        max_value=float(spec["max"]),
                        value=float(st.session_state.get(field_name, spec["default"])),
                        step=float(spec["step"]),
                        help=help_text,
                        key=field_name,
                    )
                else:
                    options = spec["options"]
                    current_value = st.session_state.get(field_name, spec["default"])
                    current_index = options.index(current_value) if current_value in options else 0
                    input_values[field_name] = st.selectbox(
                        label,
                        options=options,
                        index=current_index,
                        help=help_text,
                        key=field_name,
                    )

        st.divider()

    submit_col1, submit_col2, submit_col3 = st.columns([1, 2, 1])
    with submit_col2:
        submitted = st.form_submit_button("Predict churn risk", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)


if submitted:
    input_frame = pd.DataFrame([input_values])
    probability = float(model.predict_proba(input_frame)[0][1])
    prediction = int(probability >= threshold)

    if probability >= 0.7:
        risk_level = "HIGH"
        risk_indicator = "High"
        risk_color = "#ef4444"
    elif probability >= 0.4:
        risk_level = "MEDIUM"
        risk_indicator = "Medium"
        risk_color = "#f59e0b"
    else:
        risk_level = "LOW"
        risk_indicator = "Low"
        risk_color = "#10b981"

    result_label = "WILL CHURN" if prediction else "WILL STAY"
    result_badge_color = "#ef4444" if prediction else "#10b981"
    top_banner = "High retention risk detected" if prediction else "Customer appears stable"
    percent_chart = build_percentage_donut(probability, risk_color, result_label)

    st.markdown("---")
    st.markdown('<h2 class="section-header"> Prediction Results</h2>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="result-panel">
            <div class="result-hero">
                <div>
                    <p class="result-title">{top_banner}</p>
                    <p class="result-subtitle">The model converted the customer profile into a churn probability and risk band.</p>
                </div>
                <div class="pill" style="background:{result_badge_color}22;color:{result_badge_color};">
                    <span>&bull;</span>
                    <span>{risk_level} risk</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1.1, 1.5, 1.1])

    with col1:
        card_bg = "#ef4444" if prediction else "#10b981"
        status_text = "At risk" if prediction else "Stable"
        st.markdown(
            f'<div class="prediction-card" style="background: linear-gradient(135deg, {card_bg} 0%, #00000020 100%);"><h3>Prediction</h3><div class="status">{status_text}</div><div class="label">{result_label}</div></div>',
            unsafe_allow_html=True,
        )

    with col2:
        st.pyplot(percent_chart, use_container_width=True)
        plt.close(percent_chart)
        st.markdown(
            f"""
            <div style="text-align:center;color:#cbd5e1;margin-top:-0.3rem;">
                Decision threshold: <strong>{threshold:.2f}</strong> · Probability: <strong>{probability:.1%}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f'<div style="text-align: center; padding: 2rem; background: linear-gradient(180deg, rgba(18, 27, 44, 0.98), rgba(11, 19, 36, 0.98)); border-radius: 1rem; border: 1px solid rgba(255,255,255,0.08); box-shadow: 0 14px 32px rgba(0,0,0,0.24);"><p style="margin: 0; font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; font-weight: 600;">Risk Level</p><p style="margin: 1rem 0 0 0; font-size: 1.6rem; font-weight: 700; color: {risk_color};">{risk_indicator}</p><p style="margin: 0.5rem 0 0 0; font-size: 1.3rem; font-weight: 600; color: #f8fafc;">{risk_level}</p><p style="margin:0.5rem 0 0 0;color:#94a3b8">Threshold: {threshold:.2f}</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<h3 class="section-header"> Key Metrics</h3>', unsafe_allow_html=True)
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.metric("Churn Probability", f"{probability:.1%}")

    with metric_col2:
        st.metric("Stay Probability", f"{(1 - probability):.1%}")

    with metric_col3:
        tenure = input_values.get("Tenure Months", 0)
        st.metric("Customer Tenure", f"{int(tenure)} mo" if tenure else "N/A")

    with metric_col4:
        monthly = input_values.get("Monthly Charges", 0)
        st.metric("Monthly Charges", f"${monthly:.2f}" if monthly else "N/A")

    st.markdown('<h3 class="section-header">Local Explanation (SHAP)</h3>', unsafe_allow_html=True)
    try:
        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["classifier"]
        X_trans = preprocessor.transform(input_frame)

        numeric_features = preprocessor.named_transformers_["numeric"].get_feature_names_out()
        ohe = preprocessor.named_transformers_["categorical"].named_steps["one_hot"]
        categorical_features = ohe.get_feature_names_out()
        all_feature_names = np.concatenate([numeric_features, categorical_features])

        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_trans)

        if isinstance(shap_values, list):
            sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            sv = shap_values

        instance_sv = sv[0]
        idx = np.argsort(np.abs(instance_sv))[-10:][::-1]
        top_feats_list = [str(feature) for feature in all_feature_names[idx]]
        top_vals_list = [float(value) for value in instance_sv[idx]]

        fig, ax = plt.subplots(figsize=(6, 3))
        cols = ["#ef4444" if value > 0 else "#10b981" for value in top_vals_list]
        ax.barh(range(len(top_feats_list)), top_vals_list, color=cols)
        ax.set_yticks(range(len(top_feats_list)))
        ax.set_yticklabels(top_feats_list, fontsize=9)
        ax.set_xlabel("SHAP value", fontsize=9)
        ax.set_title("Top local feature contributions", fontsize=10)
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        report = {
            "inputs": input_values,
            "probability": probability,
            "prediction": int(prediction),
            "threshold": threshold,
            "top_features": {feature: value for feature, value in zip(top_feats_list, top_vals_list)},
        }
        st.download_button(
            "Download report (JSON)",
            data=json.dumps(report, indent=2),
            file_name="churn_report.json",
            mime="application/json",
        )
    except Exception as exc:
        st.info(f"SHAP explanation unavailable: {exc}")

    st.markdown('<h3 class="section-header">Risk Assessment & Factors</h3>', unsafe_allow_html=True)
    risk_col1, risk_col2 = st.columns(2)

    with risk_col1:
        st.markdown("#### Positive Factors (Lower Risk)")
        positive_factors = []

        if input_values.get("Tenure Months", 0) > 24:
            positive_factors.append("Long customer tenure (24+ months)")
        if input_values.get("Contract") == "Two year":
            positive_factors.append("Long-term contract commitment")
        if input_values.get("Online Security") == "Yes":
            positive_factors.append("Security services active")
        if input_values.get("Tech Support") == "Yes":
            positive_factors.append("Premium tech support")
        if input_values.get("Partner") == "Yes":
            positive_factors.append("Has family/partner account")

        if positive_factors:
            for factor in positive_factors:
                st.success(factor)
        else:
            st.info("No strong positive factors")

    with risk_col2:
        st.markdown("#### Risk Factors (Higher Risk)")
        risk_factors = []

        if input_values.get("Monthly Charges", 0) > 80:
            risk_factors.append("High monthly charges (>$80)")
        if input_values.get("Tenure Months", 0) < 12:
            risk_factors.append("New customer (low tenure)")
        if input_values.get("Contract") == "Month-to-month":
            risk_factors.append("Month-to-month contract")
        if input_values.get("Internet Service") == "Fiber optic":
            risk_factors.append("Fiber optic service")
        if input_values.get("Tech Support") == "No":
            risk_factors.append("No technical support")

        if risk_factors:
            for factor in risk_factors:
                st.warning(factor)
        else:
            st.info("No significant risk factors")

    st.markdown('<h3 class="section-header">Customer Profile & Recommendations</h3>', unsafe_allow_html=True)
    profile_col1, profile_col2 = st.columns(2)

    with profile_col1:
        st.markdown("#### Customer Input Details")
        profile_df = pd.DataFrame(
            {
                "Feature": list(input_values.keys()),
                "Value": [str(value) for value in input_values.values()],
            }
        )
        st.dataframe(profile_df, use_container_width=True, hide_index=True, height=400)

    with profile_col2:
        st.markdown("#### Recommended Actions")
        if prediction:
            st.markdown(
                """
                <div style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); padding: 2rem; border-radius: 1rem; color: white;">
                <h4 style="margin: 0 0 1rem 0;">High Risk - Immediate Action Required</h4>
                <ul style="margin: 0; padding-left: 1.5rem;">
                <li><strong>Urgent:</strong> Contact customer within 24 hours</li>
                <li><strong>Assess:</strong> Understand satisfaction and pain points</li>
                <li><strong>Offer:</strong> Premium discount or service upgrade</li>
                <li><strong>Loyalty:</strong> Extended contract with incentives</li>
                <li><strong>Monitor:</strong> Track engagement daily</li>
                </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); padding: 2rem; border-radius: 1rem; color: white;">
                <h4 style="margin: 0 0 1rem 0;">Low Risk - Growth Opportunity</h4>
                <ul style="margin: 0; padding-left: 1.5rem;">
                <li><strong>Upsell:</strong> Premium services and bundles</li>
                <li><strong>Engagement:</strong> Regular check-ins and updates</li>
                <li><strong>Loyalty:</strong> VIP rewards program</li>
                <li><strong>Growth:</strong> Cross-sell opportunities</li>
                <li><strong>Retention:</strong> Maintain satisfaction levels</li>
                </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="footer">
            ChurnPredict AI v2.1 | Powered by Advanced ML | Random Forest Classification
        </div>
        """,
        unsafe_allow_html=True,
    )
