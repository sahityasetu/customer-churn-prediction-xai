"""Data loading, cleaning and feature engineering for the Telco churn data."""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "Telco-Customer-Churn.csv"

SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]
NUMERIC_COLS = [
    "tenure", "MonthlyCharges", "TotalCharges", "num_services",
    "avg_monthly_spend", "charge_increase",
]
CATEGORICAL_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod", "tenure_band",
]
TARGET = "Churn"


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 11 new customers (tenure = 0) have a blank TotalCharges: they have not been billed yet.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce").fillna(0.0)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"}).fillna(df["SeniorCitizen"])
    if TARGET in df.columns:
        df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["num_services"] = (df[SERVICE_COLS] == "Yes").sum(axis=1)
    # Average historical spend; for brand-new customers fall back to the current charge.
    df["avg_monthly_spend"] = np.where(df["tenure"] > 0, df["TotalCharges"] / df["tenure"].clip(lower=1), df["MonthlyCharges"])
    # Positive value = the customer now pays more than their historical average (a common churn trigger).
    df["charge_increase"] = df["MonthlyCharges"] - df["avg_monthly_spend"]
    df["tenure_band"] = pd.cut(
        df["tenure"], bins=[-1, 6, 12, 24, 48, 100],
        labels=["0-6m", "7-12m", "13-24m", "25-48m", "49m+"],
    ).astype(str)
    return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    return add_features(clean(df))


def split_xy(df: pd.DataFrame):
    return df[NUMERIC_COLS + CATEGORICAL_COLS], df[TARGET]
