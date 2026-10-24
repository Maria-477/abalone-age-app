"""
Abalone Age Prediction - Streamlit app

Run locally:  streamlit run app.py
"""
import json
import os
import pickle

import pandas as pd
import sklearn
import streamlit as st

from model_utils import (
    ALL_FEATURES, INFO_FILE, MODEL_FILE, NUMERIC_FEATURES,
    build_info, fit_final_model, load_data, run_cross_validation,
)

st.set_page_config(page_title="Abalone Age Predictor", page_icon="🐚", layout="centered")

# Example abalone from the notebook (Step 11.1), used as the default inputs
DEFAULTS = {
    "Length": 0.455, "Diameter": 0.365, "Height": 0.095,
    "Whole weight": 0.514, "Shucked weight": 0.2245,
    "Viscera weight": 0.1010, "Shell weight": 0.150,
}
SEX_LABELS = {"M": "Male (M)", "F": "Female (F)", "I": "Infant (I)"}


@st.cache_resource(show_spinner="Loading the model...")
def load_model():
    """Load the saved model. If it can't be loaded (e.g. a different scikit-learn
    version on the server), re-train it from abalone.csv with the same workflow."""
    info = None
    if os.path.exists(INFO_FILE):
        with open(INFO_FILE) as f:
            info = json.load(f)

    if info and os.path.exists(MODEL_FILE) and info.get("sklearn_version") == sklearn.__version__:
        try:
            with open(MODEL_FILE, "rb") as f:
                return pickle.load(f), info
        except Exception:
            pass  # fall through to re-training

    df = load_data()
    if info is None:
        cv_results = run_cross_validation(df)
        best = min(cv_results, key=lambda n: cv_results[n]["mean_rmse"])
        info = build_info(df, best, cv_results)
    model = fit_final_model(df, info["best_algorithm"])
    return model, info


st.title("🐚 Abalone Age Predictor")
st.write(
    "Enter an abalone's physical measurements to estimate its age in years. "
    "Age is estimated as **rings + 1.5**, the same target used in the project notebook."
)

try:
    model, info = load_model()
except Exception as error:
    st.error(f"The model could not be loaded: {error}")
    st.stop()

with st.form("abalone_form"):
    sex_options = info["sex_values"]
    sex = st.selectbox(
        "Sex", options=sex_options,
        index=sex_options.index("M") if "M" in sex_options else 0,
        format_func=lambda s: SEX_LABELS.get(s, s),
    )

    values = {}
    for feature in NUMERIC_FEATURES:
        low, high, _ = info["feature_ranges"][feature]
        values[feature] = st.number_input(
            feature,
            min_value=0.0,
            max_value=round(high * 1.5, 4),
            value=float(min(max(DEFAULTS[feature], low), high)),
            step=0.001,
            format="%.4f",
            help=f"Range seen in the training data: {low:.4f} to {high:.4f}",
        )
    submitted = st.form_submit_button("Predict", type="primary")

if submitted:
    try:
        # Same columns, same order as training
        features = pd.DataFrame([{**values, "Sex": sex}])[ALL_FEATURES]

        outside = [
            f for f in NUMERIC_FEATURES
            if not info["feature_ranges"][f][0] <= values[f] <= info["feature_ranges"][f][1]
        ]
        if outside:
            st.warning(
                "These values are outside the range of the training data, so the prediction "
                "may be less reliable: " + ", ".join(outside)
            )

        predicted_age = float(model.predict(features)[0])
        if predicted_age <= 0:
            st.error("The model returned an invalid age. Please check the measurements.")
        else:
            st.success(f"Predicted Abalone Age: {predicted_age:.2f} years")
            st.caption(
                f"Typical error (cross-validation RMSE of {info['best_algorithm']}): "
                f"about {info['cv_mean_rmse']:.2f} years. Estimated rings: {predicted_age - 1.5:.1f}."
            )
    except Exception as error:
        st.error(f"Something went wrong while predicting: {error}")

with st.expander("About the model"):
    st.write(
        f"- **Algorithm:** {info['best_algorithm']} (lowest mean RMSE in 10-fold cross validation)\n"
        f"- **Trained on:** {info['n_samples']:,} abalones\n"
        f"- **Mean RMSE:** {info['cv_mean_rmse']:.3f} years (± {info['cv_std_rmse']:.3f})\n"
        "- **Pre-processing:** median imputation + standard scaling for numbers, "
        "one-hot encoding for Sex (built into the model pipeline)"
    )
    comparison = pd.DataFrame(info["cv_results"]).T.rename(
        columns={"mean_rmse": "Mean RMSE (years)", "std_rmse": "Std RMSE (years)"}
    ).sort_values("Mean RMSE (years)")
    st.dataframe(comparison.style.format("{:.3f}"))
