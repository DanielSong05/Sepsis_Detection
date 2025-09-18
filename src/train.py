# src/train.py
import json
import os
import numpy as np
import pandas as pd
from joblib import dump

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from models import MODEL_BUILDERS  # must include "voting" with voting="soft"

SEED = 42
K_OUTER = 5
K_INNER = 4
ARTIFACTS_DIR = "artifacts"
SPLITS_DIR = "splits"
DATA_DIR = "data/processed"

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(SPLITS_DIR, exist_ok=True)
os.makedirs(f"{ARTIFACTS_DIR}/cv", exist_ok=True)
os.makedirs(f"{ARTIFACTS_DIR}/models", exist_ok=True)
os.makedirs(f"{ARTIFACTS_DIR}/thresholds", exist_ok=True)

def load_data():
    X = pd.read_csv(f"{DATA_DIR}/features.csv")
    y = pd.read_csv(f"{DATA_DIR}/labels.csv")
    df = X.merge(y, on="patient_id", validate="one_to_one")
    pid = df["patient_id"].values
    y = df["label"].values.astype(int)
    X = df.drop(columns=["patient_id", "label"])
    return X, y, pid

def build_outer_folds(y, groups):
    sgkf = StratifiedGroupKFold(n_splits=K_OUTER, shuffle=True, random_state=SEED)
    folds = list(sgkf.split(np.zeros_like(y), y, groups))
    train_idx, test_idx = folds[0]  # reserve a test fold for later scripts
    pd.Series(groups[test_idx]).to_csv(f"{SPLITS_DIR}/test_ids.csv", index=False, header=["patient_id"])
    return train_idx, test_idx

def get_inner_cv_splits(y_train, groups_train):
    sgkf = StratifiedGroupKFold(n_splits=K_INNER, shuffle=True, random_state=SEED)
    return list(sgkf.split(np.zeros_like(y_train), y_train, groups_train))

def model_candidates():
    return {
        "logistic": MODEL_BUILDERS["logistic"](),
        "tree": MODEL_BUILDERS["tree"](criterion="gini", max_depth=None, min_samples_leaf=5),
        "bagging": MODEL_BUILDERS["bagging"](n_estimators=150, base_max_depth=3),
        "rf": MODEL_BUILDERS["rf"](n_estimators=300, max_depth=None, min_samples_leaf=3, max_features="sqrt"),
        "adaboost": MODEL_BUILDERS["adaboost"](n_estimators=300, learning_rate=0.05),
        "gb": MODEL_BUILDERS["gb"](n_estimators=400, learning_rate=0.05, max_depth=3, subsample=0.8),
        "voting": MODEL_BUILDERS["voting"](),  # ensure voting="soft" so predict_proba exists
    }

def build_pipeline(est):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", est),
    ])

def oof_for_model(name, estimator, X_train, y_train, pid_train, inner_splits):
    """Return AUPRC, AUROC, and OOF probabilities; also save per-patient OOF CSV."""
    oof = np.zeros_like(y_train, dtype=float)
    for tr, va in inner_splits:
        X_tr, X_va = X_train.iloc[tr], X_train.iloc[va]
        y_tr = y_train[tr]
        pipe = build_pipeline(estimator)
        pipe.fit(X_tr, y_tr)
        oof[va] = pipe.predict_proba(X_va)[:, 1]

    # Save lean OOF file for visualizations/diagnostics
    pd.DataFrame({
        "patient_id": pid_train,
        "y": y_train,
        "oof_proba": oof
    }).to_csv(f"{ARTIFACTS_DIR}/cv/{name}_oof.csv", index=False)

    ap = average_precision_score(y_train, oof)
    auc = roc_auc_score(y_train, oof)
    return ap, auc, oof

def pick_threshold(y_true, proba, method="max_f1"):
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    thresholds = np.append(thresholds, 1.0)  # align arrays with precision/recall
    if method == "max_f1":
        f1s = (2 * precision * recall) / (precision + recall + 1e-12)
        best_idx = int(np.nanargmax(f1s))
        return float(thresholds[best_idx])
    return float(np.median(proba))

def main():
    X, y, pid = load_data()

    # Reserve a test fold for later prediction/evaluation scripts
    train_pool_idx, _ = build_outer_folds(y, pid)
    X_pool, y_pool, g_pool, pid_pool = (
        X.iloc[train_pool_idx], y[train_pool_idx], pid[train_pool_idx], pid[train_pool_idx]
    )
    print(f"Train-pool size: {len(X_pool)} (held-out test IDs saved to {SPLITS_DIR}/test_ids.csv)")

    # Inner CV on training pool: compute OOF per model
    inner_splits = get_inner_cv_splits(y_pool, g_pool)
    results, oof_store = [], {}

    for name, est in model_candidates().items():
        ap, auc, oof = oof_for_model(name, est, X_pool, y_pool, pid_pool, inner_splits)
        thr = pick_threshold(y_pool, oof, method="max_f1")  # compute per-model threshold

        results.append({
            "model": name,
            "cv_auprc": ap,
            "cv_auroc": auc,
            "best_threshold": thr,  # add to leaderboard
        })
        oof_store[name] = oof
        print(f"[{name}] AUPRC={ap:.4f} | AUROC={auc:.4f} | thr={thr:.4f}")

    # Save CV leaderboard and choose best by AUPRC
    leaderboard_path = f"{ARTIFACTS_DIR}/cv/leaderboard.csv"
    pd.DataFrame(results).sort_values("cv_auprc", ascending=False).to_csv(leaderboard_path, index=False)
    best = max(results, key=lambda d: d["cv_auprc"])["model"]
    print(f"Selected best model: {best} (leaderboard saved to {leaderboard_path})")

    # Save best model name and threshold (picked from its OOF)
    with open(f"{ARTIFACTS_DIR}/models/best_model_name.json", "w") as f:
        json.dump({"best_model": best}, f)

    thr = pick_threshold(y_pool, oof_store[best], method="max_f1")
    with open(f"{ARTIFACTS_DIR}/thresholds/threshold.json", "w") as f:
        json.dump({"threshold": thr}, f)
    print(f"Chosen decision threshold (from OOF): {thr:.4f}")

    # Retrain best model on FULL training pool and save fitted pipeline
    final_est = model_candidates()[best]
    final_pipe = build_pipeline(final_est)
    final_pipe.fit(X_pool, y_pool)
    dump(final_pipe, f"{ARTIFACTS_DIR}/models/best_model.joblib")
    print(f"Saved trained model to {ARTIFACTS_DIR}/models/best_model.joblib")

if __name__ == "__main__":
    main()
