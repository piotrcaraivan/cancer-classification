
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, cast

import numpy as np
from joblib import dump, load
from sklearn.calibration import calibration_curve
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    brier_score_loss,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch
import os


@dataclass
class SplitArtifact:
    feature_names: list[str]
    target_names: list[str]
    X_test: np.ndarray
    y_test: np.ndarray


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_dataset() -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    # sklearn stubs can be imprecise across versions; cast keeps Pylance happy.
    data = cast(Bunch, load_breast_cancer())
    X = data.data
    y = data.target
    feature_names = list(data.feature_names)
    target_names = list(data.target_names)
    return X, y, feature_names, target_names


def _build_model(random_state: int, max_iter: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    random_state=random_state,
                    max_iter=max_iter,
                    n_jobs=None,
                ),
            ),
        ]
    )


def _get_proba(model: Pipeline, X: np.ndarray) -> Optional[np.ndarray]:
    if hasattr(model, "predict_proba"):
        return cast(np.ndarray, model.predict_proba(X))
    return None


def _evaluate(model: Pipeline, X_test: np.ndarray, y_test: np.ndarray, target_names: list[str]) -> Dict[str, Any]:
    y_pred = model.predict(X_test)
    proba = _get_proba(model, X_test)
    y_score = proba[:, 1] if proba is not None else None

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        ),
    }

    if y_score is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_score))
        metrics["average_precision"] = float(average_precision_score(y_test, y_score))
        metrics["log_loss"] = float(log_loss(y_test, y_score, labels=[0, 1]))
        metrics["brier_loss"] = float(brier_score_loss(y_test, y_score))

        fpr, tpr, _ = roc_curve(y_test, y_score)
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_score)
        prob_true, prob_pred = calibration_curve(y_test, y_score, n_bins=10)

        metrics["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        metrics["pr_curve"] = {
            "precision": precision_curve.tolist(),
            "recall": recall_curve.tolist(),
        }
        metrics["calibration_curve"] = {
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
        }

    return metrics


def _maybe_save_confusion_plot(
    *,
    cm: list[list[int]],
    target_names: list[str],
    out_path: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    _ensure_parent_dir(out_path)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(target_names)),
        yticks=np.arange(len(target_names)),
        xticklabels=target_names,
        yticklabels=target_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = np.max(cm) / 2.0 if np.max(cm) > 0 else 0
    for i in range(len(target_names)):
        for j in range(len(target_names)):
            ax.text(
                j,
                i,
                str(cm[i][j]),
                ha="center",
                va="center",
                color="white" if cm[i][j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=140)
    plt.close(fig)


def _maybe_save_curve_plot(
    *,
    x: list[float],
    y: list[float],
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    _ensure_parent_dir(out_path)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(x, y, label=title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=140)
    plt.close(fig)


def _maybe_save_probability_hist(
    *,
    y_true: np.ndarray,
    y_score: np.ndarray,
    out_path: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    _ensure_parent_dir(out_path)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(y_score[y_true == 0], bins=20, alpha=0.6, label="malignant (0)")
    ax.hist(y_score[y_true == 1], bins=20, alpha=0.6, label="benign (1)")
    ax.set_xlabel("Predicted probability for class 1 (benign)")
    ax.set_ylabel("Count")
    ax.set_title("Probability distribution by class")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=140)
    plt.close(fig)


def _maybe_save_top_coefficients(
    *,
    model: Pipeline,
    feature_names: list[str],
    top_k: int,
    out_path: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    clf = cast(LogisticRegression, model.named_steps.get("clf"))
    if clf is None or clf.coef_ is None:
        return

    coefs = clf.coef_[0]
    idx = np.argsort(np.abs(coefs))[::-1][:top_k]
    top_features = [feature_names[i] for i in idx]
    top_values = coefs[idx]

    _ensure_parent_dir(out_path)
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#1f77b4" if v >= 0 else "#d62728" for v in top_values]
    ax.barh(range(len(top_features))[::-1], top_values[::-1], color=colors[::-1])
    ax.set_yticks(range(len(top_features))[::-1])
    ax.set_yticklabels(top_features[::-1])
    ax.set_xlabel("Coefficient value (after scaling)")
    ax.set_title(f"Top {top_k} influential features (LogReg)")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=160)
    plt.close(fig)


def cmd_train(args: argparse.Namespace) -> int:
    X, y, feature_names, target_names = _load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    model = _build_model(random_state=args.random_state, max_iter=args.max_iter)
    model.fit(X_train, y_train)

    metrics = _evaluate(model, X_test, y_test, target_names)
    payload: Dict[str, Any] = {
        "created_at_utc": _utc_now_iso(),
        "dataset": "sklearn.datasets.load_breast_cancer",
        "model": "LogisticRegression + StandardScaler (sklearn Pipeline)",
        "params": {
            "test_size": args.test_size,
            "random_state": args.random_state,
            "max_iter": args.max_iter,
        },
        "metrics": metrics,
        "target_mapping": {"0": "malignant", "1": "benign"},
    }

    model_path = Path(args.model_out)
    split_path = Path(args.split_out)
    metrics_path = Path(args.metrics_out)

    _ensure_parent_dir(model_path)
    _ensure_parent_dir(split_path)
    _ensure_parent_dir(metrics_path)

    dump(model, model_path)
    dump(
        SplitArtifact(
            feature_names=feature_names,
            target_names=target_names,
            X_test=X_test,
            y_test=y_test,
        ),
        split_path,
    )
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.no_plots:
        plots_dir = Path(args.plots_dir)
        cm_plot_path = plots_dir / "confusion_matrix.png"
        _maybe_save_confusion_plot(
            cm=metrics["confusion_matrix"],
            target_names=target_names,
            out_path=cm_plot_path,
        )

        if "roc_curve" in metrics:
            _maybe_save_curve_plot(
                x=metrics["roc_curve"]["fpr"],
                y=metrics["roc_curve"]["tpr"],
                xlabel="False Positive Rate",
                ylabel="True Positive Rate",
                title="ROC curve",
                out_path=plots_dir / "roc_curve.png",
            )
        if "pr_curve" in metrics:
            _maybe_save_curve_plot(
                x=metrics["pr_curve"]["recall"],
                y=metrics["pr_curve"]["precision"],
                xlabel="Recall",
                ylabel="Precision",
                title="Precision-Recall curve",
                out_path=plots_dir / "pr_curve.png",
            )
        if "calibration_curve" in metrics:
            _maybe_save_curve_plot(
                x=metrics["calibration_curve"]["prob_pred"],
                y=metrics["calibration_curve"]["prob_true"],
                xlabel="Predicted probability (bin avg)",
                ylabel="Observed frequency",
                title="Calibration curve",
                out_path=plots_dir / "calibration_curve.png",
            )
        if "roc_curve" in metrics:
            proba = _get_proba(model, X_test)
            if proba is not None:
                _maybe_save_probability_hist(
                    y_true=y_test,
                    y_score=proba[:, 1],
                    out_path=plots_dir / "prob_hist.png",
                )

        _maybe_save_top_coefficients(
            model=model,
            feature_names=feature_names,
            top_k=min(12, len(feature_names)),
            out_path=plots_dir / "top_coefficients.png",
        )

    print("Train finished")
    print(f"Model:   {model_path}")
    print(f"Split:   {split_path}")
    print(f"Metrics: {metrics_path}")
    if not args.no_plots:
        print(f"Plots:   {Path(args.plots_dir)}")
    print(f"Accuracy: {payload['metrics']['accuracy']:.4f}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    model = load(Path(args.model))

    if args.split:
        split: SplitArtifact = load(Path(args.split))
        X_test, y_test = split.X_test, split.y_test
        target_names = split.target_names
    else:
        X, y, _, target_names = _load_dataset()
        _, X_test, _, y_test = train_test_split(
            X,
            y,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=y,
        )

    metrics = _evaluate(model, X_test, y_test, target_names)

    out_path = Path(args.out)
    _ensure_parent_dir(out_path)
    payload: Dict[str, Any] = {
        "created_at_utc": _utc_now_iso(),
        "model_path": str(Path(args.model)),
        "metrics": metrics,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.no_plots:
        plots_dir = Path(args.plots_dir)
        cm_plot_path = plots_dir / "confusion_matrix_eval.png"
        _maybe_save_confusion_plot(
            cm=metrics["confusion_matrix"],
            target_names=target_names,
            out_path=cm_plot_path,
        )

        if "roc_curve" in metrics:
            _maybe_save_curve_plot(
                x=metrics["roc_curve"]["fpr"],
                y=metrics["roc_curve"]["tpr"],
                xlabel="False Positive Rate",
                ylabel="True Positive Rate",
                title="ROC curve",
                out_path=plots_dir / "roc_curve_eval.png",
            )
        if "pr_curve" in metrics:
            _maybe_save_curve_plot(
                x=metrics["pr_curve"]["recall"],
                y=metrics["pr_curve"]["precision"],
                xlabel="Recall",
                ylabel="Precision",
                title="Precision-Recall curve",
                out_path=plots_dir / "pr_curve_eval.png",
            )
        if "calibration_curve" in metrics:
            _maybe_save_curve_plot(
                x=metrics["calibration_curve"]["prob_pred"],
                y=metrics["calibration_curve"]["prob_true"],
                xlabel="Predicted probability (bin avg)",
                ylabel="Observed frequency",
                title="Calibration curve",
                out_path=plots_dir / "calibration_curve_eval.png",
            )
        proba = _get_proba(model, X_test)
        if proba is not None:
            _maybe_save_probability_hist(
                y_true=y_test,
                y_score=proba[:, 1],
                out_path=plots_dir / "prob_hist_eval.png",
            )

    print("Evaluate finished")
    print(f"Report:  {out_path}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    return 0


def _features_from_json(payload: Dict[str, Any], feature_names: list[str]) -> np.ndarray:
    missing = [name for name in feature_names if name not in payload]
    extra = [name for name in payload.keys() if name not in set(feature_names)]
    if missing:
        raise ValueError(f"Missing features: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    if extra:
        raise ValueError(f"Unknown features: {extra[:5]}{' ...' if len(extra) > 5 else ''}")

    row = [float(payload[name]) for name in feature_names]
    return np.array(row, dtype=float).reshape(1, -1)


def cmd_predict(args: argparse.Namespace) -> int:
    model: Pipeline = load(Path(args.model))

    X, y, feature_names, target_names = _load_dataset()
    target_mapping = {0: "malignant", 1: "benign"}

    if args.sample_index is not None:
        idx = int(args.sample_index)
        if idx < 0 or idx >= len(X):
            raise ValueError(f"sample-index out of range: 0..{len(X) - 1}")
        x = X[idx].reshape(1, -1)
        true_label = int(y[idx])
    elif args.input_json:
        payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("input-json must contain a JSON object (feature -> value)")
        x = _features_from_json(payload, feature_names)
        true_label = None
    else:
        raise ValueError("Provide either --sample-index or --input-json")

    pred = int(model.predict(x)[0])
    out: Dict[str, Any] = {
        "created_at_utc": _utc_now_iso(),
        "prediction": {
            "class_id": pred,
            "class_name": target_mapping.get(pred, str(pred)),
        },
    }

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)[0]
        out["prediction"]["probabilities"] = {target_mapping[i]: float(proba[i]) for i in range(len(proba))}

    if true_label is not None:
        out["ground_truth"] = {
            "class_id": true_label,
            "class_name": target_mapping.get(true_label, str(true_label)),
        }

    if args.out:
        out_path = Path(args.out)
        _ensure_parent_dir(out_path)
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved: {out_path}")
    else:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0

def cmd_cv(args):
    """
    Запускает кросс-валидацию (проверку стабильности).
    """
    print(f"Loading dataset for Cross-Validation (folds={args.folds})...")
    
    X, y, _, _ = _load_dataset()

    # Строим модель (новую, чистую)
    model = _build_model(random_state=args.random_state, max_iter=args.max_iter)

    print("Running cross_validate...")
    # Запускаем проверку
    # scoring - какие оценки нам нужны
    scores = cross_validate(
        model, 
        X, 
        y, 
        cv=args.folds, 
        scoring=['accuracy', 'roc_auc', 'average_precision'],
        n_jobs=-1  # использовать все процессоры
    )

    # Собираем красивый отчет
    summary = {
        "folds": args.folds,
        "metrics": {}
    }

    # Считаем среднее и отклонение для каждой метрики
    for key in scores:
        # scores[key] - это массив из 5 чисел (по одному на фолд)
        mean_val = float(np.mean(scores[key]))
        std_val = float(np.std(scores[key]))
        
        summary["metrics"][key] = {
            "mean": mean_val,
            "std": std_val,
            "per_fold": [float(x) for x in scores[key]]
        }
        print(f"  {key}: {mean_val:.4f} ± {std_val:.4f}")

    # Сохраняем в файл cv.json
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"CV results saved to {args.out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cancer_prediction",
        description="Breast cancer classification (scikit-learn) with reproducible CLI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_cv = sub.add_parser("cv", help="Cross-validation to see stability across folds")
    
    p_cv.add_argument("--folds", type=int, default=5, help="Number of folds")
    p_cv.add_argument("--random-state", type=int, default=42)
    p_cv.add_argument("--max-iter", type=int, default=2000)
    p_cv.add_argument("--out", default="artifacts/cv.json")
    
    p_cv.set_defaults(func=cmd_cv)

    p_train = sub.add_parser("train", help="Train model and save artifacts")
    p_train.add_argument("--test-size", type=float, default=0.2)
    p_train.add_argument("--random-state", type=int, default=42)
    p_train.add_argument("--max-iter", type=int, default=2000)
    p_train.add_argument("--model-out", default="artifacts/model.joblib")
    p_train.add_argument("--split-out", default="artifacts/split.joblib")
    p_train.add_argument("--metrics-out", default="artifacts/metrics.json")
    p_train.add_argument("--plots-dir", default="artifacts/plots")
    p_train.add_argument("--no-plots", action="store_true")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="Evaluate a saved model")
    p_eval.add_argument("--model", required=True)
    p_eval.add_argument("--split", help="Path to split.joblib created by train")
    p_eval.add_argument("--test-size", type=float, default=0.2)
    p_eval.add_argument("--random-state", type=int, default=42)
    p_eval.add_argument("--out", default="artifacts/eval.json")
    p_eval.add_argument("--plots-dir", default="artifacts/plots")
    p_eval.add_argument("--no-plots", action="store_true")
    p_eval.set_defaults(func=cmd_evaluate)

    p_pred = sub.add_parser("predict", help="Predict for one sample")
    p_pred.add_argument("--model", required=True)
    p_pred.add_argument("--sample-index", type=int)
    p_pred.add_argument("--input-json", help="JSON with features -> values")
    p_pred.add_argument("--out", help="Output JSON path")
    p_pred.set_defaults(func=cmd_predict)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())