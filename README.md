# Sepsis_Detection

Sepsis Prediction is a machine learning pipeline for early detection of sepsis using patient data. The project covers end-to-end development, including data preprocessing, stratified train/test splits, model training with cross-validation, performance evaluation (AUROC, AUPRC, precision, recall, F1), and generating predictions. Artifacts such as trained models, metrics, and visualizations are stored for reproducibility and further analysis.


## Structure
- `src/`: pipelines for preprocess, train, evaluate, and predict
- `data/`: raw and processed datasets (raw data ignored)
- `artifacts/`: models, metrics, and visualizations (ignored)
- `splits/`: dataset splits

## Quickstart
1. Create and activate a virtual environment
2. Install dependencies: `pip install -r requirements.txt`
3. See `src/train.py`, `src/predict.py`, `src/evaluate.py` for entry points
