"""
Data loading and feature engineering for CICIDS2017 dataset.
Handles the known data quality issues: infinite values, NaNs, duplicate columns.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

# CICIDS2017 column name normalization (files use inconsistent casing/spacing)
LABEL_COLUMN = "Label"

ATTACK_FAMILY = {
    "BENIGN": "BENIGN",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "DDoS": "DDoS",
    "PortScan": "PortScan",
    "FTP-Patator": "BruteForce",
    "SSH-Patator": "BruteForce",
    "Bot": "Botnet",
    "Web Attack \x96 Brute Force": "WebAttack",
    "Web Attack \x96 XSS": "WebAttack",
    "Web Attack \x96 Sql Injection": "WebAttack",
    "Web Attack – Brute Force": "WebAttack",
    "Web Attack – XSS": "WebAttack",
    "Web Attack – Sql Injection": "WebAttack",
    "Infiltration": "Infiltration",
    "Heartbleed": "Heartbleed",
}

# Features selected after correlation analysis and variance thresholding
SELECTED_FEATURES = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
]


def load_cicids2017(data_dir: str | Path, use_families: bool = True) -> pd.DataFrame:
    """
    Load all CICIDS2017 CSV files from a directory and merge them.
    Maps granular attack labels to attack families for cleaner multi-class problem.
    """
    data_dir = Path(data_dir)
    csv_files = list(data_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    logger.info(f"Loading {len(csv_files)} CSV files from {data_dir}")
    frames = []

    for f in csv_files:
        logger.info(f"  Reading {f.name} ...")
        df = pd.read_csv(f, encoding="utf-8", low_memory=False)
        df.columns = df.columns.str.strip()
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    logger.info(f"Raw dataset shape: {data.shape}")

    if use_families:
        data[LABEL_COLUMN] = data[LABEL_COLUMN].str.strip().map(ATTACK_FAMILY)
        unknown = data[LABEL_COLUMN].isna().sum()
        if unknown > 0:
            logger.warning(f"{unknown} rows with unmapped labels — dropping them")
        data = data.dropna(subset=[LABEL_COLUMN])

    return data


def clean(data: pd.DataFrame) -> pd.DataFrame:
    """Remove infinities, NaNs, and obvious outliers from CICIDS2017."""
    df = data.copy()

    # Replace inf/-inf with NaN then drop
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    before = len(df)
    df.dropna(inplace=True)
    after = len(df)
    logger.info(f"Dropped {before - after} rows with NaN/Inf ({(before-after)/before*100:.1f}%)")

    # Drop duplicate rows (CICIDS2017 has ~1k exact duplicates)
    df.drop_duplicates(inplace=True)
    logger.info(f"After dedup: {len(df)} rows")

    return df


def select_features(data: pd.DataFrame) -> pd.DataFrame:
    """Keep only the pre-selected feature columns + label."""
    available = [c for c in SELECTED_FEATURES if c in data.columns]
    missing = set(SELECTED_FEATURES) - set(available)
    if missing:
        logger.warning(f"Missing features (will be skipped): {missing}")
    return data[available + [LABEL_COLUMN]]


def encode_labels(series: pd.Series, encoder: LabelEncoder | None = None):
    """Fit or apply a LabelEncoder. Returns (encoded array, fitted encoder)."""
    if encoder is None:
        encoder = LabelEncoder()
        encoded = encoder.fit_transform(series)
    else:
        encoded = encoder.transform(series)
    return encoded, encoder


def build_preprocessor(X_train: pd.DataFrame):
    """Fit a StandardScaler on training data."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def save_artifacts(scaler: StandardScaler, encoder: LabelEncoder, out_dir: str | Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, out_dir / "scaler.pkl")
    joblib.dump(encoder, out_dir / "label_encoder.pkl")
    logger.info(f"Artifacts saved to {out_dir}")


def load_artifacts(model_dir: str | Path):
    model_dir = Path(model_dir)
    scaler = joblib.load(model_dir / "scaler.pkl")
    encoder = joblib.load(model_dir / "label_encoder.pkl")
    return scaler, encoder
