"""Time- and frequency-domain features for sensor windows."""

import numpy as np
import pandas as pd
from scipy import signal, stats

from config import STRIDE_SEC, TARGET_FS, WINDOW_SEC


def axis_features(values, prefix, fs=TARGET_FS):
    sample_count = len(values)
    features = {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_rms": float(np.sqrt(np.mean(values ** 2))),
        f"{prefix}_range": float(np.max(values) - np.min(values)),
        f"{prefix}_skew": float(stats.skew(values)) if sample_count > 2 else 0.0,
        f"{prefix}_kurt": float(stats.kurtosis(values)) if sample_count > 3 else 0.0,
        f"{prefix}_zcr": float(np.mean(np.diff(np.sign(values)) != 0)),
        f"{prefix}_iqr": float(
            np.percentile(values, 75) - np.percentile(values, 25)
        ),
        f"{prefix}_energy": float(np.sum(values ** 2)),
    }

    if sample_count >= 8:
        frequencies, power = signal.welch(
            values, fs=fs, nperseg=min(sample_count, 32)
        )
        total_power = float(np.sum(power))
        if total_power > 0:
            normalized_power = power / total_power
            features[f"{prefix}_dom_freq"] = float(
                frequencies[np.argmax(power)]
            )
            features[f"{prefix}_spec_centroid"] = float(
                np.sum(frequencies * power) / total_power
            )
            features[f"{prefix}_spec_entropy"] = float(
                -np.sum(normalized_power * np.log2(normalized_power + 1e-12))
            )
        else:
            features[f"{prefix}_dom_freq"] = 0.0
            features[f"{prefix}_spec_centroid"] = 0.0
            features[f"{prefix}_spec_entropy"] = 0.0
    else:
        features[f"{prefix}_dom_freq"] = 0.0
        features[f"{prefix}_spec_centroid"] = 0.0
        features[f"{prefix}_spec_entropy"] = 0.0
    return features


def extract_window_features(window_df, accelerometer, gyroscope):
    features = {}
    for column in accelerometer:
        features.update(
            axis_features(window_df[column].values, prefix=f"acc_{column}")
        )
    if len(accelerometer) == 3:
        magnitude = np.sqrt(
            sum(window_df[column].values ** 2 for column in accelerometer)
        )
        features.update(axis_features(magnitude, prefix="acc_mag"))
    for column in gyroscope:
        features.update(
            axis_features(window_df[column].values, prefix=f"gyro_{column}")
        )
    if len(gyroscope) == 3:
        magnitude = np.sqrt(
            sum(window_df[column].values ** 2 for column in gyroscope)
        )
        features.update(axis_features(magnitude, prefix="gyro_mag"))
    return features


def window_features(
    data_df,
    accelerometer,
    gyroscope,
    fs=TARGET_FS,
    win_sec=WINDOW_SEC,
    stride_sec=STRIDE_SEC,
):
    window_size = int(win_sec * fs)
    stride = int(stride_sec * fs)
    rows = []
    for start in range(0, len(data_df) - window_size + 1, stride):
        window = data_df.iloc[start:start + window_size]
        rows.append(extract_window_features(window, accelerometer, gyroscope))
    return pd.DataFrame(rows)
