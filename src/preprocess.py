import os
import pandas as pd

RAW_DATA_DIRS = [
    "data/raw/training_setA/training",
    "data/raw/training_setB/training_setB"
]

PROCESSED_DATA_DIR = "data/processed"

MISSING_VALUE_THRESHOLD = 0.8

def drop_missing_features():
    #sample missingness from folders for quickness
    sample_frames = []

    for folder in RAW_DATA_DIRS:
        for i, fname in enumerate(os.listdir(folder)):
            if not fname.endswith(".psv"):
                continue
            df = pd.read_csv(os.path.join(folder, fname), sep="|")
            sample_frames.append(df)
            if i > 200:
                break
    
    big_df = pd.concat(sample_frames, ignore_index=True)
    missing_pct = big_df.isna().mean()
    kept_features = missing_pct[missing_pct < MISSING_VALUE_THRESHOLD].index.tolist()
    # Ensure label is kept for downstream processing
    if "SepsisLabel" in big_df.columns and "SepsisLabel" not in kept_features:
        kept_features.append("SepsisLabel")
    return kept_features

def extract_features(patient_df, kept_features):
    df = patient_df[kept_features].copy()
    label = int(df["SepsisLabel"].max())
    #Extracts features from patient-level dataframe
    #dont include predictor
    X = df.drop(columns=["SepsisLabel"]) 
    #fill missing values with median
    X = X.fillna(X.median())

    #aggregate statistics
    features = {}
    features.update(X.mean().add_suffix("_mean"))
    features.update(X.std().add_suffix("_std"))
    features.update(X.min().add_suffix("_min"))
    features.update(X.max().add_suffix("_max"))
    features.update(X.iloc[-1].add_suffix("_last"))

    return features, label

def prepare_data():
    kept_features = drop_missing_features()
    #loads all raw data, extracts patient-level features, saves processed CSVs.
    all_features = []
    all_labels = []
    all_ids = []

    for folder in RAW_DATA_DIRS:
        for fname in os.listdir(folder):
            if not fname.endswith(".psv"):
                continue
            pid = fname.replace(".psv", "")
            df = pd.read_csv(os.path.join(folder, fname), sep="|")
            features, label = extract_features(df, kept_features)
            features["patient_id"] = pid
            all_features.append(features)
            all_labels.append(label)
            all_ids.append(pid)
    #Build Dataframe

    X = pd.DataFrame(all_features).set_index("patient_id")
    y = pd.Series(all_labels, index=all_ids, name = "label")
    y.index.name = "patient_id"  # Set index name so CSV has proper column header
    
    os.makedirs(PROCESSED_DATA_DIR, exist_ok = True)
    X.to_csv(os.path.join(PROCESSED_DATA_DIR, "features.csv"))
    y.to_csv(os.path.join(PROCESSED_DATA_DIR, "labels.csv"))

    print(f"Saved process data to {PROCESSED_DATA_DIR}")
    print(f"Shape: X={X.shape}, y = {y.shape}")
    return X,y

if __name__ == "__main__":
    prepare_data()