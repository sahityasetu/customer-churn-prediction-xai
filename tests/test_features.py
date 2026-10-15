import numpy as np
import pandas as pd

from src.business import CampaignAssumptions, best_threshold, expected_profit
from src.features import prepare


def _row(**kw):
    base = {
        "customerID": "x", "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "tenure": 10, "PhoneService": "Yes", "MultipleLines": "Yes", "InternetService": "DSL",
        "OnlineSecurity": "Yes", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "No", "StreamingMovies": "No", "Contract": "One year", "PaperlessBilling": "No",
        "PaymentMethod": "Mailed check", "MonthlyCharges": 50.0, "TotalCharges": "400", "Churn": "No",
    }
    base.update(kw)
    return pd.DataFrame([base])


def test_blank_total_charges_becomes_zero():
    df = prepare(_row(tenure=0, TotalCharges=" "))
    assert df.loc[0, "TotalCharges"] == 0.0
    assert df.loc[0, "avg_monthly_spend"] == 50.0


def test_engineered_features():
    df = prepare(_row())
    assert df.loc[0, "num_services"] == 3          # phone, multiple lines, online security
    assert df.loc[0, "avg_monthly_spend"] == 40.0  # 400 / 10
    assert df.loc[0, "charge_increase"] == 10.0
    assert df.loc[0, "tenure_band"] == "7-12m"
    assert df.loc[0, "Churn"] == 0


def test_profit_counts_offer_cost_and_saved_value():
    a = CampaignAssumptions(offer_cost=10, save_rate=0.5, value_months=2)
    y = np.array([1, 0])
    proba = np.array([0.9, 0.9])
    charges = np.array([100.0, 100.0])
    # contact both: saved value = 100*2*0.5 = 100, cost = 2*10
    assert expected_profit(y, proba, charges, 0.5, a) == 80.0


def test_best_threshold_skips_low_value_contacts():
    y = np.array([1, 0, 0, 0])
    proba = np.array([0.9, 0.3, 0.3, 0.3])
    charges = np.full(4, 100.0)
    thr, _, _ = best_threshold(y, proba, charges)
    assert 0.3 < thr <= 0.9
