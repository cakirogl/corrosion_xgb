"""
Rebar Corrosion Parameter Predictor
=====================================
Streamlit app that predicts i_corr and E_corr from experimentally
controlled variables using optimised XGBoost models.

Reference:
  Li et al. (2023), Construction and Building Materials 409, 134160.
  Hyperparameters tuned via 300-trial Bayesian optimisation (TPE, seed 42).
"""

import os
import pickle
import numpy as np
import streamlit as st

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rebar Corrosion Predictor",
    page_icon="🔬",
    layout="centered",
)

# ── load pre-trained models from pickle files ─────────────────────────────────
@st.cache_resource
def load_models():
    base = os.path.dirname(__file__)
    with open(os.path.join(base, "models", "model_icorr.pkl"), "rb") as f:
        mi = pickle.load(f)
    with open(os.path.join(base, "models", "model_ecorr.pkl"), "rb") as f:
        me = pickle.load(f)
    with open(os.path.join(base, "models", "model_cols.pkl"), "rb") as f:
        cols = pickle.load(f)
    return mi, me, cols

mi, me, MODEL_COLS = load_models()

# ── corrosion risk helpers (ASTM C876 / literature thresholds) ────────────────
def icorr_risk(val):
    if val < 0.1:
        return "Passive", "#27AE60"
    elif val < 0.5:
        return "Low", "#F1C40F"
    elif val < 1.0:
        return "Moderate", "#E67E22"
    else:
        return "High", "#E74C3C"

def ecorr_risk(val_mv):
    if val_mv > -200:
        return "Low probability of corrosion (ASTM C876)", "#27AE60"
    elif val_mv > -350:
        return "Uncertain — further testing recommended", "#F1C40F"
    else:
        return "High probability of corrosion (ASTM C876)", "#E74C3C"

# ── encode a single observation ───────────────────────────────────────────────
def encode_input(sample_type, carb_state, condition, cl_level):
    row = {c: 0 for c in MODEL_COLS}
    for col in MODEL_COLS:
        if col == f"SampleType_{sample_type}":
            row[col] = 1
        elif col == f"CarbonationState_{carb_state}":
            row[col] = 1
        elif col == f"Condition_{condition}":
            row[col] = 1
        elif col == "ClLevel_wtPct_Cl_by_cement":
            row[col] = cl_level
    return np.array([[row[c] for c in MODEL_COLS]], dtype=float)

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("Rebar Corrosion Parameter Predictor")
st.markdown(
    "Predict **corrosion current density** (*i*_corr) and "
    "**corrosion potential** (*E*_corr) of steel rebar from "
    "experimentally controlled conditions, using optimised XGBoost models "
    "trained on the dataset of Li et al. (2023)."
)
st.divider()

col_in, col_out = st.columns([1, 1], gap="large")

with col_in:
    st.subheader("Input parameters")

    sample_type = st.selectbox(
        "Sample type",
        ["Mortar", "SPS"],
        help="Mortar: rebar embedded in cement mortar. "
             "SPS: bare rebar in simulated pore solution.",
    )

    carb_state = st.selectbox(
        "Carbonation state",
        ["Non-carbonated", "Carbonated"],
        help="Carbonated specimens have undergone CO₂-induced pH reduction.",
    )

    if sample_type == "Mortar":
        cond_opts = ["RH65", "RH85", "RH95", "Submerged"]
        cond_help = "Relative humidity of the curing/exposure environment, or submerged."
    else:
        cond_opts = ["72h", "20d", "36h"]
        cond_help = "Duration of immersion in simulated pore solution."

    condition = st.selectbox("Condition", cond_opts, help=cond_help)

    cl_level = st.slider(
        "Chloride content (wt% Cl by cement mass)",
        min_value=0.0, max_value=2.5, value=0.65, step=0.01,
        help="Chloride ion content expressed as wt% Cl relative to cement mass.",
    )

    predict_btn = st.button("Predict", type="primary", use_container_width=True)

with col_out:
    st.subheader("Predictions")

    if predict_btn:
        Xq     = encode_input(sample_type, carb_state, condition, cl_level)
        i_pred = 10 ** mi.predict(Xq)[0]
        e_pred = me.predict(Xq)[0] * 1000          # V → mV

        i_label, i_col = icorr_risk(i_pred)
        e_label, e_col = ecorr_risk(e_pred)

        st.metric(
            label="Corrosion current density  (*i*_corr)",
            value=f"{i_pred:.4f}  µA/cm²",
        )
        st.markdown(
            f"<span style='color:{i_col}; font-weight:600;'>"
            f"Corrosion activity: {i_label}</span>",
            unsafe_allow_html=True,
        )

        st.divider()

        st.metric(
            label="Corrosion potential  (*E*_corr)",
            value=f"{e_pred:.1f}  mV vs. SCE",
        )
        st.markdown(
            f"<span style='color:{e_col}; font-weight:600;'>"
            f"{e_label}</span>",
            unsafe_allow_html=True,
        )

        st.divider()
        st.caption(
            "Thresholds: *i*_corr — passive < 0.1, low 0.1–0.5, "
            "moderate 0.5–1.0, high > 1.0 µA/cm²  |  "
            "*E*_corr — ASTM C876."
        )
    else:
        st.info("Set the input parameters and click **Predict**.")

st.divider()
st.caption(
    "Models trained on: Li et al. (2023), *Construction and Building Materials* **409**, 134160.  "
    "Hyperparameters optimised via 300-trial Bayesian optimisation (Optuna TPE, seed 42).  "
    "5-fold CV R² — *i*_corr: 0.936 | *E*_corr: 0.953."
)
