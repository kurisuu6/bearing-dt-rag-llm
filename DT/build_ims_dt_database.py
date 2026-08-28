import argparse
import csv
import json
import math
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


DATASET_CONFIG = {
    "1st_test": {
        "channels": 8,
        "bearings": {
            "bearing_1": [0, 1],
            "bearing_2": [2, 3],
            "bearing_3": [4, 5],
            "bearing_4": [6, 7],
        },
        "failure_description": "inner race defect in bearing 3 and roller element defect in bearing 4",
    },
    "2nd_test": {
        "channels": 4,
        "bearings": {
            "bearing_1": [0],
            "bearing_2": [1],
            "bearing_3": [2],
            "bearing_4": [3],
        },
        "failure_description": "outer race failure in bearing 1",
    },
    "3rd_test": {
        "channels": 4,
        "bearings": {
            "bearing_1": [0],
            "bearing_2": [1],
            "bearing_3": [2],
            "bearing_4": [3],
        },
        "failure_description": "outer race failure in bearing 3",
    },
}

FEATURE_COLUMNS = [
    "mean",
    "std",
    "rms",
    "abs_mean",
    "peak",
    "peak_to_peak",
    "skewness",
    "kurtosis",
    "crest_factor",
    "shape_factor",
    "impulse_factor",
    "clearance_factor",
    "energy",
    "zero_crossing_rate",
    "dominant_frequency",
    "spectral_centroid",
    "spectral_entropy",
    "low_band_energy_ratio",
    "mid_band_energy_ratio",
    "high_band_energy_ratio",
]

DEGRADATION_FEATURES = [
    "rms",
    "kurtosis",
    "crest_factor",
    "high_band_energy_ratio",
    "spectral_entropy",
]

WEIGHTS = {
    "rms": 0.35,
    "kurtosis": 0.25,
    "crest_factor": 0.15,
    "high_band_energy_ratio": 0.15,
    "spectral_entropy": 0.10,
}


@dataclass
class SignalFile:
    experiment: str
    path: Path
    timestamp: Optional[datetime]
    sequence_index: int


def parse_timestamp(path: Path) -> Optional[datetime]:
    text = path.name
    match = re.search(r"(\d{4})[._-](\d{2})[._-](\d{2})[._-](\d{2})[._-](\d{2})[._-](\d{2})", text)
    if not match:
        return None
    year, month, day, hour, minute, second = map(int, match.groups())
    return datetime(year, month, day, hour, minute, second)


def discover_experiment_dirs(data_root: Path) -> dict[str, Path]:
    dirs = {}
    for name in DATASET_CONFIG:
        matches = [p for p in data_root.rglob("*") if p.is_dir() and p.name.lower() == name.lower()]
        if matches:
            dirs[name] = matches[0]
    return dirs


def is_signal_file(path: Path) -> bool:
    if path.name.startswith(".") or path.name.startswith("._"):
        return False
    if path.is_dir():
        return False
    if path.suffix.lower() in {".pdf", ".rar", ".zip", ".csv", ".json", ".db"}:
        return False
    return True


def find_signal_files_in_experiment_dir(directory: Path) -> list[Path]:
    direct_files = [p for p in directory.iterdir() if is_signal_file(p)]
    if direct_files:
        return direct_files

    candidate_files = []
    for child in directory.iterdir():
        if not child.is_dir() or child.name.startswith('.'):
            continue
        files = [p for p in child.iterdir() if is_signal_file(p)]
        timestamped = [p for p in files if parse_timestamp(p) is not None]
        if timestamped:
            candidate_files.append((len(timestamped), child, timestamped))

    if not candidate_files:
        return []

    candidate_files.sort(key=lambda item: item[0], reverse=True)
    _, chosen_dir, files = candidate_files[0]
    print(f"[INFO] Using nested IMS data directory {chosen_dir}", flush=True)
    return files


def discover_signal_files(data_root: Path, experiments: Optional[set[str]] = None) -> list[SignalFile]:
    experiment_dirs = discover_experiment_dirs(data_root)
    signal_files = []
    for experiment, directory in experiment_dirs.items():
        if experiments and experiment not in experiments:
            continue
        files = find_signal_files_in_experiment_dir(directory)
        files.sort(key=lambda p: (parse_timestamp(p) or datetime.min, p.name))
        for idx, path in enumerate(files):
            signal_files.append(
                SignalFile(
                    experiment=experiment,
                    path=path,
                    timestamp=parse_timestamp(path),
                    sequence_index=idx,
                )
            )
    return signal_files


def read_signal_matrix(path: Path, expected_channels: int) -> np.ndarray:
    try:
        data = np.loadtxt(path)
    except ValueError:
        data = np.loadtxt(path, delimiter="\t")
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if data.shape[1] < expected_channels:
        raise ValueError(f"{path} has {data.shape[1]} columns, expected {expected_channels}")
    return data[:, :expected_channels].astype(float, copy=False)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0 or not np.isfinite(denominator):
        return default
    return float(numerator / denominator)


def extract_features(signal: np.ndarray, sampling_rate: float) -> dict[str, float]:
    x = np.asarray(signal, dtype=float)
    n = len(x)
    if n == 0:
        raise ValueError("empty signal")

    mean = float(np.mean(x))
    centered = x - mean
    std = float(np.std(x))
    rms = float(np.sqrt(np.mean(x * x)))
    abs_mean = float(np.mean(np.abs(x)))
    peak = float(np.max(np.abs(x)))
    peak_to_peak = float(np.ptp(x))
    fourth = float(np.mean(centered ** 4))
    third = float(np.mean(centered ** 3))
    skewness = safe_divide(third, std ** 3)
    kurtosis = safe_divide(fourth, std ** 4, default=0.0)
    sqrt_abs_mean = float(np.mean(np.sqrt(np.abs(x))))

    zero_crossing_rate = float(np.mean(np.diff(np.signbit(centered)) != 0)) if n > 1 else 0.0
    spectrum = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate)
    power = spectrum ** 2
    total_power = float(np.sum(power))
    if total_power <= 0:
        dominant_frequency = 0.0
        spectral_centroid = 0.0
        spectral_entropy = 0.0
        low_ratio = mid_ratio = high_ratio = 0.0
    else:
        dominant_frequency = float(freqs[int(np.argmax(power))])
        spectral_centroid = float(np.sum(freqs * power) / total_power)
        probabilities = power / total_power
        spectral_entropy = float(-np.sum(probabilities * np.log2(probabilities + 1e-12)) / math.log2(len(probabilities)))
        low_ratio = float(np.sum(power[freqs < 1000]) / total_power)
        mid_ratio = float(np.sum(power[(freqs >= 1000) & (freqs < 5000)]) / total_power)
        high_ratio = float(np.sum(power[freqs >= 5000]) / total_power)

    return {
        "mean": mean,
        "std": std,
        "rms": rms,
        "abs_mean": abs_mean,
        "peak": peak,
        "peak_to_peak": peak_to_peak,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "crest_factor": safe_divide(peak, rms),
        "shape_factor": safe_divide(rms, abs_mean),
        "impulse_factor": safe_divide(peak, abs_mean),
        "clearance_factor": safe_divide(peak, sqrt_abs_mean ** 2),
        "energy": float(np.sum(x * x)),
        "zero_crossing_rate": zero_crossing_rate,
        "dominant_frequency": dominant_frequency,
        "spectral_centroid": spectral_centroid,
        "spectral_entropy": spectral_entropy,
        "low_band_energy_ratio": low_ratio,
        "mid_band_energy_ratio": mid_ratio,
        "high_band_energy_ratio": high_ratio,
    }


def build_feature_records(signal_files: list[SignalFile], sampling_rate: float, max_files_per_experiment: Optional[int]) -> pd.DataFrame:
    records = []
    seen_counts = {name: 0 for name in DATASET_CONFIG}
    for sf in signal_files:
        if max_files_per_experiment is not None and seen_counts[sf.experiment] >= max_files_per_experiment:
            continue
        config = DATASET_CONFIG[sf.experiment]
        matrix = read_signal_matrix(sf.path, config["channels"])
        seen_counts[sf.experiment] += 1
        timestamp_text = sf.timestamp.isoformat(sep=" ") if sf.timestamp else None
        for bearing_id, channel_indices in config["bearings"].items():
            for channel_index in channel_indices:
                features = extract_features(matrix[:, channel_index], sampling_rate)
                record = {
                    "experiment": sf.experiment,
                    "bearing_id": bearing_id,
                    "channel": channel_index + 1,
                    "timestamp": timestamp_text,
                    "sequence_index": sf.sequence_index,
                    "file_name": sf.path.name,
                    "file_path": str(sf.path),
                    "sampling_rate": sampling_rate,
                    **features,
                }
                records.append(record)
        if seen_counts[sf.experiment] % 100 == 0:
            print(f"[INFO] {sf.experiment}: processed {seen_counts[sf.experiment]} files", flush=True)
    return pd.DataFrame(records)


def add_health_index(df: pd.DataFrame, baseline_fraction: float, z_scale: float) -> pd.DataFrame:
    frames = []
    for (experiment, bearing_id, channel), group in df.groupby(["experiment", "bearing_id", "channel"], sort=False):
        group = group.sort_values("sequence_index").copy()
        baseline_count = max(5, int(len(group) * baseline_fraction))
        baseline = group.iloc[:baseline_count]
        score = np.zeros(len(group), dtype=float)
        evidence_columns = []
        for feature in DEGRADATION_FEATURES:
            mu = float(baseline[feature].mean())
            sigma = float(baseline[feature].std(ddof=0)) or 1e-12
            z = ((group[feature] - mu) / sigma).clip(lower=0.0)
            group[f"{feature}_baseline_mean"] = mu
            group[f"{feature}_baseline_std"] = sigma
            group[f"{feature}_z"] = z
            score += WEIGHTS[feature] * z.to_numpy()
            evidence_columns.append(f"{feature}_z")
        group["health_index"] = np.clip(score / z_scale, 0.0, 1.0)
        group["health_state"] = group["health_index"].map(classify_health_state)
        group["evidence"] = group.apply(make_evidence, axis=1)
        frames.append(group)
    return pd.concat(frames, ignore_index=True)


def classify_health_state(health_index: float) -> str:
    if health_index < 0.20:
        return "normal"
    if health_index < 0.40:
        return "early_degradation"
    if health_index < 0.70:
        return "degradation"
    if health_index < 0.90:
        return "severe_degradation"
    return "failure_near"


def make_evidence(row: pd.Series) -> str:
    evidence = []
    labels = [
        ("rms", "RMS"),
        ("kurtosis", "kurtosis"),
        ("crest_factor", "crest factor"),
        ("high_band_energy_ratio", "high-frequency energy ratio"),
        ("spectral_entropy", "spectral entropy"),
    ]
    for key, label in labels:
        z = float(row.get(f"{key}_z", 0.0))
        if z >= 3.0:
            evidence.append(f"{label} z-score {z:.2f}")
    return "; ".join(evidence)


def build_snapshots(df: pd.DataFrame) -> pd.DataFrame:
    agg = {
        "health_index": "max",
        "mean": "mean",
        "std": "mean",
        "rms": "mean",
        "peak": "mean",
        "peak_to_peak": "mean",
        "skewness": "mean",
        "kurtosis": "mean",
        "crest_factor": "mean",
        "shape_factor": "mean",
        "impulse_factor": "mean",
        "clearance_factor": "mean",
        "energy": "mean",
        "zero_crossing_rate": "mean",
        "dominant_frequency": "mean",
        "spectral_centroid": "mean",
        "spectral_entropy": "mean",
        "low_band_energy_ratio": "mean",
        "mid_band_energy_ratio": "mean",
        "high_band_energy_ratio": "mean",
        "evidence": lambda values: "; ".join(sorted({v for v in values if v})),
    }
    snapshots = df.groupby(["experiment", "bearing_id", "timestamp", "sequence_index"], dropna=False).agg(agg).reset_index()
    snapshots["health_state"] = snapshots["health_index"].map(classify_health_state)
    return snapshots


def write_sqlite(db_path: Path, features: pd.DataFrame, snapshots: pd.DataFrame):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        features.to_sql("feature_records", conn, if_exists="replace", index=False)
        snapshots.to_sql("bearing_snapshots", conn, if_exists="replace", index=False)
        experiments = pd.DataFrame(
            [
                {
                    "experiment": name,
                    "channels": config["channels"],
                    "failure_description": config["failure_description"],
                }
                for name, config in DATASET_CONFIG.items()
            ]
        )
        experiments.to_sql("experiments", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_features_exp_bearing_seq ON feature_records(experiment, bearing_id, sequence_index)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_exp_bearing_seq ON bearing_snapshots(experiment, bearing_id, sequence_index)")


def write_summary(output_dir: Path, features: pd.DataFrame, snapshots: pd.DataFrame):
    output_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_dir / "ims_feature_records.csv", index=False)
    snapshots.to_csv(output_dir / "ims_bearing_snapshots.csv", index=False)
    latest = snapshots.sort_values("sequence_index").groupby(["experiment", "bearing_id"], as_index=False).tail(1)
    latest.to_csv(output_dir / "ims_latest_bearing_states.csv", index=False)
    summary = {
        "feature_records": int(len(features)),
        "bearing_snapshots": int(len(snapshots)),
        "experiments": sorted(features["experiment"].dropna().unique().tolist()) if not features.empty else [],
        "latest_states": latest[["experiment", "bearing_id", "health_index", "health_state", "evidence"]].to_dict(orient="records"),
    }
    (output_dir / "ims_dt_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Build a lightweight bearing DT database from extracted IMS bearing data.")
    parser.add_argument("--data-root", default="IMS", help="Directory containing extracted 1st_test/2nd_test/3rd_test folders.")
    parser.add_argument("--output-dir", default="DT/outputs")
    parser.add_argument("--db", default="DT/outputs/ims_dt.db")
    parser.add_argument("--sampling-rate", type=float, default=20000.0)
    parser.add_argument("--baseline-fraction", type=float, default=0.10)
    parser.add_argument("--z-scale", type=float, default=8.0)
    parser.add_argument("--experiments", default="", help="Optional comma list, e.g. 1st_test,2nd_test")
    parser.add_argument("--max-files-per-experiment", type=int, default=None, help="Optional limit for quick tests.")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(
            f"IMS data root not found: {data_root}. Extract IMS.zip and the test .rar files first."
        )
    selected = {item.strip() for item in args.experiments.split(",") if item.strip()} or None
    signal_files = discover_signal_files(data_root, selected)
    if not signal_files:
        raise RuntimeError(f"No IMS signal files found under {data_root}")

    print(f"[INFO] Discovered {len(signal_files)} signal files", flush=True)
    features = build_feature_records(signal_files, args.sampling_rate, args.max_files_per_experiment)
    print(f"[INFO] Extracted {len(features)} feature records", flush=True)
    features = add_health_index(features, args.baseline_fraction, args.z_scale)
    snapshots = build_snapshots(features)
    print(f"[INFO] Built {len(snapshots)} bearing snapshots", flush=True)

    output_dir = Path(args.output_dir)
    write_summary(output_dir, features, snapshots)
    write_sqlite(Path(args.db), features, snapshots)
    print(f"[OK] CSV/JSON outputs -> {output_dir}", flush=True)
    print(f"[OK] SQLite DT database -> {args.db}", flush=True)


if __name__ == "__main__":
    main()
