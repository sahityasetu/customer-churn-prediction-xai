"""Streamlit app: score one customer or a whole CSV, and explain the score.

Run:  streamlit run app.py   (train first with: python -m src.train)
"""
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import shap
import streamlit as st

from src.features import CATEGORICAL_COLS, load_raw, prepare, split_xy

st.set_page_config(page_title="Churn Risk Scorer", layout="wide")


@st.cache_resource
def load_artifacts():
    bundle = joblib.load("models/churn_model.joblib")
    pipe = bundle["pipeline"]
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    X_bg, _ = split_xy(prepare(load_raw()).sample(300, random_state=0))
    bg = prep.transform(X_bg)
    if hasattr(model, "coef_"):
        explainer = shap.LinearExplainer(model, bg)
    else:
        explainer = shap.TreeExplainer(model)
    names = [c.split("__", 1)[1] for c in prep.get_feature_names_out()]
    return bundle, explainer, names


def readable(feature: str, row: pd.Series) -> str:
    """Turn 'Contract_Two year' into 'Contract is not Two year' for this customer."""
    for col in CATEGORICAL_COLS:
        if feature.startswith(col + "_"):
            level = feature[len(col) + 1:]
            return f"{col} is {'' if str(row[col]) == level else 'not '}{level}"
    return f"{feature} = {row[feature]:,.1f}" if isinstance(row[feature], float) else f"{feature} = {row[feature]}"


def reason_chart(contrib: pd.Series, row: pd.Series):
    d = pd.DataFrame({"reason": [readable(f, row) for f in contrib.index], "impact": contrib.values})
    d["effect"] = np.where(d["impact"] > 0, "raises risk", "lowers risk")
    fig = px.bar(d.iloc[::-1], x="impact", y="reason", orientation="h", color="effect",
                 color_discrete_map={"raises risk": "#d62728", "lowers risk": "#2ca02c"})
    fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="",
                      xaxis_title="impact on churn risk (SHAP, log-odds)", legend_title="")
    # Biggest impact on top, whatever its colour.
    fig.update_yaxes(categoryorder="array", categoryarray=d["reason"].iloc[::-1].tolist())
    return fig


bundle, explainer, feature_names = load_artifacts()
pipe, threshold = bundle["pipeline"], bundle["threshold"]

st.title("Customer Churn Risk Scorer")
st.caption(f"Model: {type(pipe.named_steps['model']).__name__} | contact threshold = {threshold:.2f} "
           "(chosen to maximise expected retention-campaign profit)")

tab_one, tab_batch = st.tabs(["Score one customer", "Score a CSV"])

with tab_one:
    c1, c2, c3 = st.columns(3)
    with c1:
        tenure = st.slider("Tenure (months)", 0, 72, 5)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        monthly = st.number_input("Monthly charges ($)", 18.0, 120.0, 80.0)
        payment = st.selectbox("Payment method", ["Electronic check", "Mailed check",
                                                  "Bank transfer (automatic)", "Credit card (automatic)"])
    with c2:
        internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        security = st.selectbox("Online security", ["No", "Yes", "No internet service"])
        support = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    with c3:
        senior = st.selectbox("Senior citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        streaming = st.selectbox("Streaming TV / movies", ["No", "Yes", "No internet service"])

    row = pd.DataFrame([{
        "customerID": "new", "gender": "Female", "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner, "Dependents": dependents, "tenure": tenure, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": internet, "OnlineSecurity": security,
        "OnlineBackup": "No" if internet != "No" else "No internet service",
        "DeviceProtection": "No" if internet != "No" else "No internet service",
        "TechSupport": support, "StreamingTV": streaming, "StreamingMovies": streaming,
        "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": monthly * tenure,
    }])
    X_row = split_xy(prepare(row).assign(Churn=0))[0]
    p = float(pipe.predict_proba(X_row)[0, 1])

    m1, m2 = st.columns([1, 2])
    m1.metric("Churn probability", f"{p:.0%}")
    if p >= threshold:
        m1.error("Action: include in retention campaign")
    else:
        m1.success("Action: no offer needed")

    sv = explainer(pipe.named_steps["prep"].transform(X_row))
    vals = sv.values[0] if sv.values.ndim == 2 else sv.values[0, :, 1]
    contrib = pd.Series(vals, index=feature_names).sort_values(key=np.abs, ascending=False).head(8)
    m2.subheader("Top reasons for this score")
    m2.plotly_chart(reason_chart(contrib, X_row.iloc[0]), width="stretch")

with tab_batch:
    st.write("Upload a CSV with the same columns as the Telco dataset to get a ranked call list.")
    file = st.file_uploader("CSV file", type="csv")
    if file is not None:
        raw = pd.read_csv(file)
        df = prepare(raw.assign(Churn=raw.get("Churn", "No")))
        df["churn_probability"] = pipe.predict_proba(split_xy(df)[0])[:, 1]
        df["contact"] = df["churn_probability"] >= threshold
        out = df.sort_values("churn_probability", ascending=False)[
            ["customerID", "tenure", "Contract", "MonthlyCharges", "churn_probability", "contact"]]
        st.write(f"{out['contact'].sum():,} of {len(out):,} customers flagged for the campaign")
        st.dataframe(out, width="stretch")
        st.download_button("Download call list", out.to_csv(index=False), "call_list.csv")
