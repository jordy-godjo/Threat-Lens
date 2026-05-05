"""
Inference module: loads a trained model and classifies new network flows.
Can be used standalone or imported by the dashboard.
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.preprocessing import SELECTED_FEATURES, load_artifacts

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

SEVERITY = {
    "BENIGN": "info",
    "DoS": "high",
    "DDoS": "critical",
    "PortScan": "medium",
    "BruteForce": "high",
    "Botnet": "high",
    "WebAttack": "medium",
    "Infiltration": "critical",
    "Heartbleed": "critical",
}


class ThreatDetector:
    def __init__(self, model_dir: str | Path = "models"):
        model_dir = Path(model_dir)
        self.scaler, self.encoder = load_artifacts(model_dir)

        # Prefer XGBoost, fall back to RF
        xgb_path = model_dir / "xgboost.pkl"
        rf_path = model_dir / "random_forest.pkl"
        best_path = next(
            (p for p in [xgb_path, rf_path] if p.exists()),
            None,
        )
        if best_path is None:
            raise FileNotFoundError(f"No model found in {model_dir}. Run src/train.py first.")
        self.model = joblib.load(best_path)
        logger.info(f"Loaded model from {best_path}")

        feature_path = model_dir / "feature_names.json"
        if feature_path.exists():
            with open(feature_path) as fh:
                self.feature_names = json.load(fh)
        else:
            self.feature_names = SELECTED_FEATURES

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Classify a batch of network flows.

        Parameters
        ----------
        X : DataFrame with columns matching the training features.

        Returns
        -------
        DataFrame with columns: label, probability, severity
        """
        X_aligned = X.reindex(columns=self.feature_names, fill_value=0.0)
        X_scaled = self.scaler.transform(X_aligned)

        y_pred = self.model.predict(X_scaled)
        labels = self.encoder.inverse_transform(y_pred)

        proba = None
        if hasattr(self.model, "predict_proba"):
            proba_matrix = self.model.predict_proba(X_scaled)
            proba = proba_matrix.max(axis=1)

        results = pd.DataFrame({"label": labels})
        if proba is not None:
            results["confidence"] = np.round(proba * 100, 1)
        results["severity"] = results["label"].map(SEVERITY).fillna("unknown")
        results["is_attack"] = results["label"] != "BENIGN"

        return results

    def predict_single(self, flow: dict) -> dict:
        """Classify a single flow dict. Convenience wrapper around predict()."""
        df = pd.DataFrame([flow])
        result = self.predict(df)
        return result.iloc[0].to_dict()
