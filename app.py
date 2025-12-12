# app.py
"""
Streamlit app — Classification YES/NO + Regression -> Future_Price_5Y

Features:
- Classification: single bold YES/NO for "Good Investment"
  - Optional: show probability(s) via a checkbox
  - Protects against leakage by NOT auto-creating price-derived features
- Regression: returns estimated Future_Price_5Y
  - If regressor predicts current price (Price_in_Lakhs), converts using (1+g)^years
  - Shows units (Lakhs) and rupee conversion

Place pretrained pipelines in the app folder:
 - best_classification_model_retrained.pkl  (recommended)
 - best_regression_model.pkl   (if you have a regressor)
"""

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from typing import List

# ---- Streamlit page config (must be first Streamlit command) ----
st.set_page_config(page_title="Housing — Good Investment (YES/NO) + 5Y Price", layout="wide")

# ---- Model file path candidates (edit if your files are named differently) ----
CLF_PIPE_PATHS = [
    "best_classification_model_retrained.pkl",
    "best_classification_model.pkl",
    "models/best_classification_model_retrained.pkl",
    "models/best_classification_model.pkl",
]
REG_PIPE_PATHS = [
    "best_regression_model.pkl",
    "models/best_regression_model.pkl",
]

# ---- Helpers ----
def try_load_pipeline(paths: List[str]):
    for p in paths:
        if os.path.exists(p):
            try:
                pipe = joblib.load(p)
                return pipe, p
            except Exception as e:
                st.sidebar.warning(f"Failed to load {p}: {e}")
    return None, None

def get_pipeline_input_cols(pipeline) -> List[str]:
    """Return expected input columns from a pipeline's 'preprocess' ColumnTransformer if present."""
    if pipeline is None:
        return []
    pre = None
    if hasattr(pipeline, "named_steps") and "preprocess" in pipeline.named_steps:
        pre = pipeline.named_steps["preprocess"]
    else:
        if hasattr(pipeline, "steps"):
            for name, step in pipeline.steps:
                if hasattr(step, "transformers_"):
                    pre = step
                    break
    if pre is None:
        return []
    cols = []
    for name, trans, cols_spec in getattr(pre, "transformers_", []):
        if cols_spec == "remainder":
            continue
        try:
            cols.extend(list(cols_spec))
        except Exception:
            # skip non-iterable specs
            pass
    # dedupe while preserving order
    seen = set(); out = []
    for c in cols:
        if c not in seen:
            out.append(c); seen.add(c)
    return out

def align_input(df: pd.DataFrame, expected_cols: List[str], dont_add=None, fill_value=np.nan) -> pd.DataFrame:
    """Ensure df contains expected cols. Add missing with fill_value (except those in dont_add)."""
    if expected_cols is None or len(expected_cols) == 0:
        return df.copy()
    if dont_add is None:
        dont_add = set()
    expected = [c for c in expected_cols if c not in dont_add]
    df2 = df.copy()
    missing = [c for c in expected if c not in df2.columns]
    for c in missing:
        df2[c] = fill_value
    # reorder to expected (this is helpful for display and some pipelines)
    df2 = df2.reindex(columns=expected)
    return df2

# Safe derived features (non-price) to create at inference
PTA_MAP = {"low": 0, "medium": 1, "high": 2, "poor": 0, "none": 0}
def add_safe_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add only non-price derived features used by classifier: PTA numeric mapping, others optional."""
    df = df.copy()
    if "Public_Transport_Accessibility_Num" not in df.columns and "Public_Transport_Accessibility" in df.columns:
        df["Public_Transport_Accessibility_Num"] = (
            df["Public_Transport_Accessibility"].astype(str).str.strip().str.lower().map(PTA_MAP).fillna(1)
        )
    # Do NOT compute Price_per_SqFt, Future_Price_5Y, Price_Growth_Percent, etc. here.
    return df

def pretty_currency_lakhs(x):
    """Format lakhs -> INR string e.g., 12.345 -> ₹1,234,500"""
    try:
        rupees = float(x) * 100000.0
        return "₹{:,.0f}".format(rupees)
    except Exception:
        return str(x)

# ---- Load pipelines ----
clf_pipe, clf_path = try_load_pipeline(CLF_PIPE_PATHS)
reg_pipe, reg_path = try_load_pipeline(REG_PIPE_PATHS)

st.sidebar.header("Model status")
st.sidebar.write(f"Classifier loaded: {'Yes' if clf_pipe is not None else 'No'}")
if clf_path:
    st.sidebar.write(f"  {clf_path}")
st.sidebar.write(f"Regressor loaded: {'Yes' if reg_pipe is not None else 'No'}")
if reg_path:
    st.sidebar.write(f"  {reg_path}")

# Show brief pipeline info in sidebar (optional)
def show_pipeline_summary(pipe, title="pipeline"):
    try:
        st.sidebar.subheader(title)
        st.sidebar.text(str(pipe))
    except Exception:
        pass

show_pipeline_summary(clf_pipe, "Classifier pipeline")
show_pipeline_summary(reg_pipe, "Regression pipeline")

# ---- UI ----
st.title("Housing: Good Investment (YES/NO)  &  Estimated Price after 5 Years")

st.markdown(
    "Upload a CSV where each row is a property. "
    "Classification returns a single YES/NO verdict (big banner). "
    "Regression returns Estimated Future Price (5 years)."
)

uploaded = st.file_uploader("Upload CSV (properties)", type=["csv"])
if uploaded:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        df = pd.DataFrame([{}])
else:
    df = pd.DataFrame([{}])  # empty sample

st.subheader("Preview — first rows")
st.write(df.head())

# Add safe derived features only (no price-derived features)
df_safe = add_safe_derived(df)
st.subheader("Input sent to models (after safe derived features) — first rows")
st.write(df_safe.head())

# ---------------------------------------------------------
# Classification block: single big YES / NO + optional probabilities
# ---------------------------------------------------------
st.markdown("## Classification — Is this a Good Investment?")
show_probs = st.checkbox("Show classification probabilities (per-row)", value=False, help="If checked, the app will show probability(s) for Good Investment alongside decisions.")

if st.button("Classify: Good Investment (YES/NO)"):
    if clf_pipe is None:
        st.error("Classifier pipeline not found. Place best_classification_model_retrained.pkl in the app folder or update CLF_PIPE_PATHS.")
    else:
        # Align inputs to classifier's expected columns. Do NOT add known leakage columns.
        clf_expected = get_pipeline_input_cols(clf_pipe)
        dont_add = set(["Future_Price_5Y", "Price_Growth_Percent", "Price_in_Lakhs", "Price_per_SqFt", "Good_Investment", "ID"])
        df_clf_aligned = align_input(df_safe, clf_expected, dont_add=dont_add)
        # Optional debug: show aligned shape (comment out in prod)
        st.write("Aligned shape sent to classifier:", df_clf_aligned.shape)

        try:
            if hasattr(clf_pipe, "predict_proba"):
                probs = clf_pipe.predict_proba(df_clf_aligned)[:, 1]
                labels = (probs >= 0.5).astype(int)
            else:
                labels = clf_pipe.predict(df_clf_aligned)
                probs = None
        except Exception as e:
            st.error(f"Classification failed: {e}")
            raise

        # Single-row: show big banner (YES/NO) and optional probability below
        if len(labels) == 1:
            if labels[0] == 1:
                st.success("👍 GOOD INVESTMENT")
            else:
                st.error("👎 NOT A GOOD INVESTMENT")
            if show_probs and probs is not None:
                st.write(f"Probability (Good Investment): {probs[0]:.4f}")
        else:
            # multiple rows -> show counts and majority verdict, plus optional per-row table when checkbox enabled
            ones = int((labels == 1).sum()); zeros = int((labels == 0).sum())
            majority = "👍 GOOD INVESTMENT" if ones >= zeros else "👎 NOT A GOOD INVESTMENT"
            st.write(f"Total rows: {len(labels)} — 👍 {ones} | 👎 {zeros}")
            if majority.startswith("👍"):
                st.success(majority)
            else:
                st.error(majority)

            if show_probs:
                out_df = pd.DataFrame({"decision": labels, "label_text": pd.Series(labels).map({1: "GOOD", 0: "NOT GOOD"})})
                if probs is not None:
                    out_df["prob_good"] = np.round(probs, 6)
                st.subheader("Per-row classification results (first 200 rows)")
                st.write(out_df.head(200))

# ---------------------------------------------------------
# Regression block: always return Future_Price_5Y
# ---------------------------------------------------------
st.markdown("## Regression — Estimated Price after 5 Years")
st.write("If your regression model predicts current price (Price_in_Lakhs), the app will convert to 5-year estimate by compounding at the chosen annual growth rate.")

# user controls for future price conversion
col_g, col_y = st.columns(2)
with col_g:
    annual_growth_pct = st.number_input("Annual growth rate (%) for 5-year estimate", min_value=0.0, max_value=100.0, value=6.0, step=0.5)
with col_y:
    years_for_estimate = st.number_input("Years to project (for Future Price)", min_value=1, max_value=20, value=5, step=1)

if st.button("Predict (Regression -> Future_Price_5Y)"):
    if reg_pipe is None:
        st.error("Regression pipeline not found. Place your regression pipeline file in the app folder or update REG_PIPE_PATHS.")
    else:
        reg_expected = get_pipeline_input_cols(reg_pipe)
        # If the regressor expects Future_Price_5Y explicitly as a column, we assume model predicts that target already.
        # Otherwise assume regressor predicts current Price_in_Lakhs and convert.
        predict_outputs_future_directly = ("Future_Price_5Y" in reg_expected)
        df_reg_aligned = align_input(df_safe, reg_expected, dont_add=set(["Good_Investment"]), fill_value=np.nan)
        st.write("Aligned shape sent to regressor:", df_reg_aligned.shape)

        try:
            preds = reg_pipe.predict(df_reg_aligned)  # model output (interpretation depends)
        except Exception as e:
            st.error(f"Regression predict failed: {e}")
            raise

        # Decide whether to convert
        if predict_outputs_future_directly:
            future_lakhs = np.array(preds, dtype=float)
            st.success("Regression model appears to predict Future_Price_5Y directly (shown in Lakhs).")
        else:
            # assume preds are current price (Price_in_Lakhs) -> convert to future
            current_lakhs = np.array(preds, dtype=float)
            rate = float(annual_growth_pct) / 100.0
            future_lakhs = current_lakhs * ((1.0 + rate) ** float(years_for_estimate))
            st.success(f"Regression model predicted current price (assumed in Lakhs). Converted to {years_for_estimate}-year estimate using {annual_growth_pct}% annual growth.")

        # Display results
        if len(future_lakhs) == 0:
            st.write("No rows to predict.")
        elif len(future_lakhs) == 1:
            v = float(future_lakhs[0])
            st.subheader("Estimated Future Price (single row)")
            st.write(f"Estimated price after {int(years_for_estimate)} years: **{v:.3f} Lakhs**")
            st.write(f"In rupees (approx): **{pretty_currency_lakhs(v)}**")
        else:
            out = pd.DataFrame({"Future_Price_Lakhs": np.round(future_lakhs, 4)})
            out["Future_Price_Rupees"] = out["Future_Price_Lakhs"].apply(pretty_currency_lakhs)
            st.subheader(f"Estimated Future Prices (first 200 rows) — units: Lakhs (and rupees approx)")
            st.write(out.head(200))

# Footer / notes
st.sidebar.markdown("---")
st.sidebar.header("Notes & tips")
st.sidebar.write("- Classification: the app avoids creating price-derived features at inference to prevent leakage.")
st.sidebar.write("- Regression: the app converts current price -> future price if model predicts current price. Confirm your regression model target when possible.")
st.sidebar.write("- Units: prices are shown in Lakhs. Multiply by 100,000 to get rupees (app also shows rupee conversion).")
st.sidebar.write("- If your model expects specific column names, ensure your CSV uses the same names (use the aligned preview above to verify).")
