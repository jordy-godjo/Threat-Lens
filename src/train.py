"""
Training pipeline: loads CICIDS2017, trains RF + XGBoost, evaluates, saves artifacts.
Usage: python -m src.train --data-dir data/raw --model-dir models
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

from src.preprocessing import (
    build_preprocessor,
    clean,
    encode_labels,
    load_cicids2017,
    save_artifacts,
    select_features,
    LABEL_COLUMN,
    SELECTED_FEATURES,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def plot_confusion_matrix(cm: np.ndarray, class_names: list, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Confusion Matrix",
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"Confusion matrix saved → {out_path}")


def plot_feature_importance(model, feature_names: list, out_path: Path, top_n: int = 20):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        [feature_names[i] for i in indices[::-1]],
        importances[indices[::-1]],
        color="steelblue",
    )
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"Feature importance plot saved → {out_path}")


def shap_summary(model, X_sample: pd.DataFrame, out_path: Path):
    logger.info("Computing SHAP values (this may take a minute)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # For multi-class, shap_values is a list; pick the most frequent non-benign class
    if isinstance(shap_values, list):
        shap_vals = np.abs(np.array(shap_values)).mean(axis=0)
    else:
        shap_vals = shap_values

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_vals, X_sample, show=False, plot_type="bar")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"SHAP summary saved → {out_path}")


def train(data_dir: str, model_dir: str, reports_dir: str, sample_frac: float = 1.0):
    data_dir = Path(data_dir)
    model_dir = Path(model_dir)
    reports_dir = Path(reports_dir)
    (reports_dir / "figures").mkdir(parents=True, exist_ok=True)

    # ── Load & preprocess ──────────────────────────────────────────────────────
    raw = load_cicids2017(data_dir)
    data = clean(raw)
    data = select_features(data)

    if sample_frac < 1.0:
        data = data.groupby(LABEL_COLUMN, group_keys=False).apply(
            lambda x: x.sample(frac=sample_frac, random_state=42)
        )
        logger.info(f"Sampled {sample_frac*100:.0f}% → {len(data)} rows")

    X = data.drop(columns=[LABEL_COLUMN])
    y_raw = data[LABEL_COLUMN]

    feature_names = X.columns.tolist()
    y, encoder = encode_labels(y_raw)
    class_names = encoder.classes_.tolist()
    logger.info(f"Classes: {class_names}")
    logger.info(f"Class distribution:\n{pd.Series(y_raw).value_counts()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = build_preprocessor(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ── Random Forest ──────────────────────────────────────────────────────────
    logger.info("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
    )
    rf.fit(X_train_s, y_train)
    rf_preds = rf.predict(X_test_s)
    rf_f1 = f1_score(y_test, rf_preds, average="weighted")
    logger.info(f"Random Forest — weighted F1: {rf_f1:.4f}")
    logger.info("\n" + classification_report(y_test, rf_preds, target_names=class_names))

    # ── XGBoost ───────────────────────────────────────────────────────────────
    logger.info("Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=42,
    )
    xgb.fit(X_train_s, y_train, eval_set=[(X_test_s, y_test)], verbose=False)
    xgb_preds = xgb.predict(X_test_s)
    xgb_f1 = f1_score(y_test, xgb_preds, average="weighted")
    logger.info(f"XGBoost — weighted F1: {xgb_f1:.4f}")
    logger.info("\n" + classification_report(y_test, xgb_preds, target_names=class_names))

    # ── Pick best model ────────────────────────────────────────────────────────
    if xgb_f1 >= rf_f1:
        best_model, best_name, best_preds = xgb, "xgboost", xgb_preds
        logger.info("Best model: XGBoost")
    else:
        best_model, best_name, best_preds = rf, "random_forest", rf_preds
        logger.info("Best model: Random Forest")

    # ── Persist ────────────────────────────────────────────────────────────────
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_dir / f"{best_name}.pkl")
    joblib.dump(rf, model_dir / "random_forest.pkl")
    joblib.dump(xgb, model_dir / "xgboost.pkl")
    save_artifacts(scaler, encoder, model_dir)

    # Save feature names for the dashboard
    with open(model_dir / "feature_names.json", "w") as fh:
        json.dump(feature_names, fh)

    # ── Reports ────────────────────────────────────────────────────────────────
    metrics = {
        "random_forest": {"weighted_f1": rf_f1},
        "xgboost": {"weighted_f1": xgb_f1},
        "best_model": best_name,
    }
    with open(reports_dir / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    cm = confusion_matrix(y_test, best_preds)
    plot_confusion_matrix(cm, class_names, reports_dir / "figures" / "confusion_matrix.png")
    plot_feature_importance(rf, feature_names, reports_dir / "figures" / "feature_importance.png")

    # SHAP on a balanced 500-row sample to keep it fast
    sample_idx = []
    for cls in np.unique(y_test):
        idx = np.where(y_test == cls)[0]
        n = min(50, len(idx))
        sample_idx.extend(np.random.choice(idx, n, replace=False))
    X_shap = pd.DataFrame(X_test_s[sample_idx], columns=feature_names)
    shap_summary(rf, X_shap, reports_dir / "figures" / "shap_summary.png")

    logger.info("Training complete.")
    return best_model, scaler, encoder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Threat-Lens IDS models")
    parser.add_argument("--data-dir", default="data/raw", help="Directory with CICIDS2017 CSVs")
    parser.add_argument("--model-dir", default="models", help="Where to save trained models")
    parser.add_argument("--reports-dir", default="reports", help="Where to save evaluation reports")
    parser.add_argument("--sample-frac", type=float, default=1.0,
                        help="Fraction of data to use (0-1), useful for quick experiments")
    args = parser.parse_args()

    train(args.data_dir, args.model_dir, args.reports_dir, args.sample_frac)
