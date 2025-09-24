# src/evaluate.py
import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix
)

ARTIFACTS_DIR = "artifacts"
PRED_DIR = f"{ARTIFACTS_DIR}/predictions"
EVAL_DIR = f"{ARTIFACTS_DIR}/eval"
VIZ_DIR = f"{ARTIFACTS_DIR}/visualizations"
DATA_DIR = "data/processed"
SPLITS_DIR = "splits"

os.makedirs(EVAL_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)

def load_threshold():
    with open(f"{ARTIFACTS_DIR}/thresholds/threshold.json", "r") as f:
        return float(json.load(f)["threshold"])

def load_labels():
    # must contain patient_id, label (as in train.py)
    y = pd.read_csv(f"{DATA_DIR}/labels.csv")
    y["patient_id"] = y["patient_id"].astype(str)
    y["label"] = y["label"].astype(int)
    return y

def load_test_ids():
    ids = pd.read_csv(f"{SPLITS_DIR}/test_ids.csv")
    ids = ids["patient_id"].astype(str).values
    return ids

def discover_prediction_files():
    """
    Finds all prediction CSVs that match the expected schema:
    columns: patient_id (string), proba (float), pred (int/bool).
    Returns list of (path, tag).
    tag is derived from filename:
      - artifacts/predictions/test_predictions.csv        -> 'main'
      - artifacts/predictions/test_predictions_XXX.csv    -> 'XXX'
    """
    paths = sorted(glob.glob(os.path.join(PRED_DIR, "test_predictions*.csv")))
    discovered = []
    for p in paths:
        fname = os.path.basename(p)
        if fname == "test_predictions.csv":
            tag = "main"
        else:
            # e.g., test_predictions_logreg.csv -> 'logreg'
            tag = fname.replace("test_predictions_", "").replace(".csv", "")
        discovered.append((p, tag))
    return discovered

def load_predictions(path):
    df = pd.read_csv(path)
    required = {"patient_id", "proba", "pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    df["patient_id"] = df["patient_id"].astype(str)
    return df

def compute_metrics(y_true, proba, y_pred, thr):
    auprc = float(average_precision_score(y_true, proba))
    auroc = float(roc_auc_score(y_true, proba))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    return {
        "test_auprc": auprc,
        "test_auroc": auroc,
        "precision_at_thr": precision,
        "recall_at_thr": recall,
        "f1_at_thr": f1,
        "threshold": thr,
        "n_test": int(len(y_true)),
        "positive_rate_test": float(np.mean(y_true)),
        "pred_positive_rate": float(np.mean(y_pred)),
    }

def plot_roc(models_data, outpath, title="ROC Curve – Sepsis Prediction"):
    fig, ax = plt.subplots(figsize=(7.5, 6.0))

    for md in models_data:
        y_true = md["y_true"]
        proba = md["proba"]
        tag   = md["tag"]

        fpr, tpr, _ = roc_curve(y_true, proba)
        au = roc_auc_score(y_true, proba)

        # main curve
        ax.plot(fpr, tpr, linewidth=2.5, label=f"{tag} (AUROC={au:.3f})")

        # operating point from your chosen threshold (via y_pred)
        if "y_pred" in md:
            y_pred = md["y_pred"]
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            fpr_op = fp / (fp + tn) if (fp + tn) else 0.0
            tpr_op = tp / (tp + fn) if (tp + fn) else 0.0
            thr     = md.get("thr", None)

            ax.scatter([fpr_op], [tpr_op], s=60, edgecolor="black", zorder=5)
            label_txt = f"thr={thr:.2f}" if thr is not None else "operating point"
            ax.annotate(label_txt, (fpr_op, tpr_op),
                        xytext=(8, -10), textcoords="offset points",
                        fontsize=11)

    # random baseline
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.8, alpha=0.7)

    # styling
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("False Positive Rate (1 − Specificity)", fontsize=13, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity/Recall)", fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=8)
    leg = ax.legend(loc="lower right", frameon=False, fontsize=11)
    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pr(models_data, outpath, title="Precision–Recall Curve – Sepsis Prediction"):
    fig, ax = plt.subplots(figsize=(7.5, 6.0))

    # Use prevalence from the first model's y_true (same test set for all models)
    baseline = float(np.mean(models_data[0]["y_true"]))

    for md in models_data:
        y_true = md["y_true"]
        proba  = md["proba"]
        tag    = md["tag"]

        rec, prec, _ = precision_recall_curve(y_true, proba)
        ap = average_precision_score(y_true, proba)

        ax.plot(rec, prec, linewidth=2.5, label=f"{tag} (AUPRC={ap:.3f})")

        # operating point from y_pred (precision/recall at chosen threshold)
        if "y_pred" in md:
            y_pred = md["y_pred"]
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            prec_op = tp / (tp + fp) if (tp + fp) else 0.0
            rec_op  = tp / (tp + fn) if (tp + fn) else 0.0
            thr     = md.get("thr", None)

            ax.scatter([rec_op], [prec_op], s=60, edgecolor="black", zorder=5)
            label_txt = f"thr={thr:.2f}" if thr is not None else "operating point"
            ax.annotate(label_txt, (rec_op, prec_op),
                        xytext=(8, -10), textcoords="offset points",
                        fontsize=11)

    # prevalence baseline
    ax.hlines(y=baseline, xmin=0, xmax=1, linestyle="--", linewidth=1.8, alpha=0.7,
              label=f"Prevalence = {baseline:.3f}")

    # styling
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Recall", fontsize=13, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=8)
    leg = ax.legend(loc="upper right", frameon=False, fontsize=11)
    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    thr = load_threshold()
    y_all = load_labels()
    test_ids = load_test_ids()

    # Discover all prediction files
    discovered = discover_prediction_files()
    if not discovered:
        raise FileNotFoundError(
            f"No prediction files found in {PRED_DIR}. "
            "Run src/predict.py to generate e.g. test_predictions.csv "
            "and optionally test_predictions_logreg.csv for baseline."
        )

    # Keep only test IDs for which we have labels
    y_test = y_all[y_all["patient_id"].isin(test_ids)][["patient_id", "label"]]

    all_metrics = []
    models_data_for_curves = []

    # Evaluate each discovered prediction set
    for pred_path, tag in discovered:
        preds = load_predictions(pred_path)
        df = y_test.merge(preds, on="patient_id", validate="one_to_one")
        y_true = df["label"].values
        proba = df["proba"].values
        y_pred = df["pred"].values  # assumed thresholded at the same thr in predict.py

        # Save per-model joined df for traceability
        df_out_path = f"{EVAL_DIR}/test_with_labels_{tag}.csv"
        df.to_csv(df_out_path, index=False)

        # Compute metrics
        m = compute_metrics(y_true, proba, y_pred, thr)
        m["model"] = tag
        all_metrics.append(m)

        # Save per-model metrics JSON (for backward compat with your original single-model output)
        per_json_path = f"{EVAL_DIR}/test_metrics_{tag}.json"
        with open(per_json_path, "w") as f:
            json.dump({
                "test_auprc": m["test_auprc"],
                "test_auroc": m["test_auroc"],
                "precision_at_thr": m["precision_at_thr"],
                "recall_at_thr": m["recall_at_thr"],
                "f1_at_thr": m["f1_at_thr"],
                "threshold": m["threshold"],
                "n_test": m["n_test"],
                "positive_rate_test": m["positive_rate_test"],
                "pred_positive_rate": m["pred_positive_rate"],
                "model": tag
            }, f, indent=2)

        # For ROC/PR overlays
        models_data_for_curves.append({"tag": tag, "y_true": y_true, "proba": proba, "y_pred": y_pred, "thr": thr})

    # Also write a combined table for easy comparison
    metrics_df = pd.DataFrame([{
        "model": m["model"],
        "AUROC": m["test_auroc"],
        "AUPRC": m["test_auprc"],
        "Precision": m["precision_at_thr"],
        "Recall": m["recall_at_thr"],
        "F1": m["f1_at_thr"],
        "Threshold": m["threshold"],
        "PosRate(Test)": m["positive_rate_test"],
        "PredPosRate": m["pred_positive_rate"],
        "N_test": m["n_test"],
    } for m in all_metrics]).sort_values(by="AUPRC", ascending=False)

    metrics_csv = f"{EVAL_DIR}/test_metrics_comparison.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    # Keep legacy single-file output for the top row (best AUPRC) if you want
    best_row = metrics_df.iloc[0]
    legacy_metrics = {
        "test_auprc": float(best_row["AUPRC"]),
        "test_auroc": float(best_row["AUROC"]),
        "precision_at_thr": float(best_row["Precision"]),
        "recall_at_thr": float(best_row["Recall"]),
        "f1_at_thr": float(best_row["F1"]),
        "threshold": float(best_row["Threshold"]),
        "n_test": int(best_row["N_test"]),
        "positive_rate_test": float(best_row["PosRate(Test)"]),
        "pred_positive_rate": float(best_row["PredPosRate"]),
        "model": str(best_row["model"])
    }
    with open(f"{EVAL_DIR}/test_metrics.json", "w") as f:
        json.dump(legacy_metrics, f, indent=2)

    # Plots
    plot_roc(models_data_for_curves, os.path.join(VIZ_DIR, "roc_overlay.png"))
    plot_pr(models_data_for_curves, os.path.join(VIZ_DIR, "pr_overlay.png"))

if __name__ == "__main__":
    main()
