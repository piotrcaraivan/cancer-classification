# Breast Cancer Classification (LogReg + Scikit-Learn)

A small, reproducible CLI project for binary classification of breast tumors (malignant vs benign) using the `sklearn` breast cancer dataset.

## Quickstart

```powershell
# From project root
X:/cancer_classification/.venv/Scripts/python.exe -m pip install -r requirements.txt

# Train (saves model, metrics, plots)
X:/cancer_classification/.venv/Scripts/python.exe cancer_prediction.py train

# Evaluate saved model (reuses saved test split)
X:/cancer_classification/.venv/Scripts/python.exe cancer_prediction.py evaluate --model artifacts/model.joblib --split artifacts/split.joblib

# Cross-validation (5 folds by default) for stability
X:/cancer_classification/.venv/Scripts/python.exe cancer_prediction.py cv --folds 5

# Single prediction by dataset index
X:/cancer_classification/.venv/Scripts/python.exe cancer_prediction.py predict --model artifacts/model.joblib --sample-index 0
```

Artifacts (after `train`):
- `artifacts/model.joblib` — trained pipeline (StandardScaler + LogisticRegression)
- `artifacts/split.joblib` — saved test split (X_test, y_test, feature names)
- `artifacts/metrics.json` — metrics and curves data
- `artifacts/plots/` — PNGs: confusion matrix, ROC, PR, calibration, probability histogram, top coefficients

Cross-validation report: `artifacts/cv.json` (means/stds per metric, per-fold values).

## What the metrics mean (layperson)
- **Accuracy / Balanced accuracy**: overall correctness; balanced accuracy is robust to class imbalance.
- **Precision / Recall / F1**: precision = how many predicted "benign" are truly benign; recall = how many true class we caught; F1 = balance of both.
- **ROC-AUC**: how well the model ranks malignant vs benign across all thresholds (closer to 1 is better).
- **Average Precision (PR-AUC)**: quality of positive predictions when threshold varies; high AP means few false positives at high confidence.
- **Log loss / Brier loss**: penalty for bad probabilities; lower is better; indicates how calibrated/confident the model is.
- **Confusion matrix**: counts of correct/incorrect decisions; especially watch false negatives (missed malignant).

## What the plots show
- `confusion_matrix.png`: where the model is right/wrong; false negatives are the critical cell.
- `roc_curve.png`: trade-off between sensitivity and false alarms; area under it is ROC-AUC.
- `pr_curve.png`: usefulness of positive predictions; good when curve is near top-right.
- `calibration_curve.png`: whether predicted probabilities match reality (ideally close to diagonal).
- `prob_hist.png`: distribution of predicted probabilities by class; well-separated peaks mean the model is confident.
- `top_coefficients.png`: most influential features of logistic regression (after scaling); blue pushes to benign, red to malignant.

## Code walkthrough (key parts of `cancer_prediction.py`)
- **Data loading**: `_load_dataset()` wraps `sklearn.datasets.load_breast_cancer()` and returns features, targets, names.
- **Model build**: `_build_model()` creates a `Pipeline(StandardScaler -> LogisticRegression)`; scaling is important for linear models.
- **Evaluation**: `_evaluate()` computes core metrics plus ROC/PR/calibration curves and stores curve points in JSON.
- **Plot helpers**: `_maybe_save_confusion_plot`, `_maybe_save_curve_plot`, `_maybe_save_probability_hist`, `_maybe_save_top_coefficients` save PNGs if matplotlib is available (headless-friendly).
- **Commands** (CLI):
  - `train`: split data, fit pipeline, save `model.joblib`, `split.joblib`, `metrics.json`, plots.
  - `evaluate`: load saved model (+ optionally saved split) and recompute metrics/plots.
  - `predict`: single-sample prediction (by dataset index or JSON with feature values) and output JSON with probabilities.
  - `cv`: Stratified K-Fold cross-validation, saves `cv.json` with mean/std per metric and per-fold values.

## Why this is useful
- Fast baseline for medical-style binary classification with transparent metrics and visuals.
- Reproducible (fixed random_state, saved splits and artifacts).
- Layperson-friendly plots explain why and how the model decides, and how trustworthy its probabilities are.

## Next steps (optional)
- Add threshold tuning to prioritize recall for malignant.
- Try stronger models (e.g., Gradient Boosting) and compare via the same `cv` command.
- Wrap into a small API/UI if you need interactive demos.
