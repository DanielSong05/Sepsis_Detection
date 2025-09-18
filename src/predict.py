# src/predict.py
import os
import json
import pandas as pd
import numpy as np
from joblib import load

ARTIFACTS_DIR = "artifacts"
PRED_DIR = f"{ARTIFACTS_DIR}/predictions"
DATA_DIR = "data/processed"
SPLITS_DIR = "splits"

os.makedirs(PRED_DIR, exist_ok=True)

def load_threshold():
    with open(f"{ARTIFACTS_DIR}/thresholds/threshold.json", "r") as f:
        return float(json.load(f)["threshold"])

def load_model():
    return load(f"{ARTIFACTS_DIR}/models/best_model.joblib")

def load_test_ids():
    # saved by train.py with header ["patient_id"]
    ids = pd.read_csv(f"{SPLITS_DIR}/test_ids.csv")
    ids = ids["patient_id"].astype(str).values
    return ids

def load_features():
    # same features used in train.py
    feats = pd.read_csv(f"{DATA_DIR}/features.csv")
    # keep patient_id as string to avoid merge/filt surprises
    feats["patient_id"] = feats["patient_id"].astype(str)
    return feats

def main():
    thr = load_threshold()
    model = load_model()
    test_ids = load_test_ids()
    feats = load_features()

    X_test = feats[feats["patient_id"].isin(test_ids)].copy()
    pid = X_test["patient_id"].values
    X = X_test.drop(columns=["patient_id"])

    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= thr).astype(int)

    out = pd.DataFrame({
        "patient_id": pid,
        "proba": proba,
        "pred": pred
    }).sort_values("patient_id")

    out_path = f"{PRED_DIR}/test_predictions.csv"
    out.to_csv(out_path, index=False)

if __name__ == "__main__":
    main()
