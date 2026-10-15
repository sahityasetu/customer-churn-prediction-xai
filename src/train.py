"""Train, compare and explain churn models.

Run from the project root:  python -m src.train
Outputs: models/churn_model.joblib, reports/metrics.json, reports/figures/*.png
"""
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.business import CampaignAssumptions, best_threshold, expected_profit
from src.features import CATEGORICAL_COLS, NUMERIC_COLS, load_raw, prepare, split_xy

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
SEED = 42


def make_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False), CATEGORICAL_COLS),
    ])


def make_models(pos_weight: float):
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=5, class_weight="balanced",
                                                n_jobs=-1, random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=4, subsample=0.8,
                                 colsample_bytree=0.8, min_child_weight=3, scale_pos_weight=pos_weight,
                                 eval_metric="logloss", random_state=SEED),
    }


def classification_metrics(y, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": round(roc_auc_score(y, proba), 4),
        "pr_auc": round(average_precision_score(y, proba), 4),
        "precision": round(precision_score(y, pred), 4),
        "recall": round(recall_score(y, pred), 4),
        "f1": round(f1_score(y, pred), 4),
    }


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    df = prepare(load_raw())
    X, y = split_xy(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    assumptions = CampaignAssumptions()

    results, fitted, oof = {}, {}, {}
    for name, model in make_models(pos_weight).items():
        pipe = Pipeline([("prep", make_preprocessor()), ("model", model)])
        # Out-of-fold predictions on the training set are used to pick the threshold,
        # so the test set is touched only once, for the final numbers.
        oof[name] = cross_val_predict(pipe, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
        thr, _, cv_profits = best_threshold(y_tr, oof[name], X_tr["MonthlyCharges"], assumptions)
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        results[name] = {
            "cv_roc_auc": round(roc_auc_score(y_tr, oof[name]), 4),
            "cv_profit_usd": round(float(cv_profits.max())),
            "threshold": thr,
            "test": classification_metrics(y_te, proba, thr),
            "test_profit_usd": round(expected_profit(y_te, proba, X_te["MonthlyCharges"], thr, assumptions)),
        }
        fitted[name] = (pipe, proba)
        print(f"{name:20s} CV AUC={results[name]['cv_roc_auc']:.3f}  test={results[name]['test']}  "
              f"profit=${results[name]['test_profit_usd']:,}")

    # Model is chosen on cross-validated profit (training data only), never on the test set.
    best = max(results, key=lambda n: results[n]["cv_profit_usd"])
    pipe, proba = fitted[best]
    thr = results[best]["threshold"]

    # Baselines on the same test set, for context.
    n_te = len(y_te)
    contact_all = expected_profit(y_te, np.ones(n_te), X_te["MonthlyCharges"], 0.5, assumptions)
    m2m = (X_te["Contract"] == "Month-to-month").astype(float).values
    contact_m2m = expected_profit(y_te, m2m, X_te["MonthlyCharges"], 0.5, assumptions)

    # Lift: share of all churners found in the top-k% riskiest customers.
    order = np.argsort(-proba)
    y_sorted = y_te.values[order]
    capture = {f"top_{p}pct": round(y_sorted[: int(n_te * p / 100)].sum() / y_sorted.sum(), 4) for p in (10, 20, 30)}

    metrics = {
        "dataset": {"rows": int(len(df)), "churn_rate": round(float(y.mean()), 4), "test_rows": int(n_te)},
        "assumptions": assumptions.__dict__,
        "models": results,
        "selected_model": best,
        "baselines_test_profit_usd": {
            "contact_nobody": 0,
            "contact_everyone": round(contact_all),
            "contact_all_month_to_month": round(contact_m2m),
            "model_targeting": results[best]["test_profit_usd"],
        },
        "churner_capture": capture,
    }
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2))
    joblib.dump({"pipeline": pipe, "threshold": thr, "assumptions": assumptions.__dict__}, ROOT / "models" / "churn_model.joblib")

    # ---------- figures ----------
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (_, p) in fitted.items():
        fpr, tpr, _ = roc_curve(y_te, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_te, p):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve - test set")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "roc_curves.png", dpi=150); plt.close(fig)

    _, grid, profits = best_threshold(y_te, proba, X_te["MonthlyCharges"], assumptions)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grid, profits / 1000)
    ax.axvline(thr, color="crimson", ls="--", label=f"chosen threshold {thr:.2f} (picked on train CV)")
    ax.axvline(0.5, color="grey", ls=":", label="default 0.50")
    ax.set(xlabel="Contact customers with churn probability >=", ylabel="Expected campaign profit ($k)",
           title=f"Profit vs threshold - {best}, test set")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "profit_curve.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    pct = np.arange(1, n_te + 1) / n_te * 100
    ax.plot(pct, np.cumsum(y_sorted) / y_sorted.sum() * 100, label=best)
    ax.plot([0, 100], [0, 100], "--", color="grey", label="random")
    ax.set(xlabel="% of customers contacted (riskiest first)", ylabel="% of churners reached", title="Cumulative gains")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "gains_chart.png", dpi=150); plt.close(fig)

    # SHAP explanations (tree models explained in transformed feature space).
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    X_te_t = pd.DataFrame(prep.transform(X_te), columns=prep.get_feature_names_out())
    X_te_t.columns = [c.split("__", 1)[1] for c in X_te_t.columns]
    if best == "Logistic Regression":
        explainer = shap.LinearExplainer(model, X_te_t)
    else:
        explainer = shap.TreeExplainer(model)
    sv = explainer(X_te_t)
    if sv.values.ndim == 3:
        sv = sv[:, :, 1]
    plt.figure()
    shap.plots.beeswarm(sv, max_display=15, show=False)
    plt.title("What drives churn (SHAP, test set)")
    plt.tight_layout(); plt.savefig(FIG / "shap_summary.png", dpi=150, bbox_inches="tight"); plt.close()
    plt.figure()
    shap.plots.waterfall(sv[int(np.argmax(proba))], max_display=10, show=False)
    plt.title("Why the highest-risk customer is flagged")
    plt.tight_layout(); plt.savefig(FIG / "shap_single_customer.png", dpi=150, bbox_inches="tight"); plt.close()

    # Churn rate by contract and tenure (EDA figure for the README).
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    df.groupby("Contract")["Churn"].mean().mul(100).plot.bar(ax=axes[0], color="#4C72B0", rot=0)
    axes[0].set(ylabel="Churn rate (%)", xlabel="", title="Churn by contract type")
    order_b = ["0-6m", "7-12m", "13-24m", "25-48m", "49m+"]
    df.groupby("tenure_band")["Churn"].mean().reindex(order_b).mul(100).plot.bar(ax=axes[1], color="#DD8452", rot=0)
    axes[1].set(ylabel="Churn rate (%)", xlabel="Tenure", title="Churn by customer tenure")
    fig.tight_layout(); fig.savefig(FIG / "eda_churn_rates.png", dpi=150); plt.close(fig)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
