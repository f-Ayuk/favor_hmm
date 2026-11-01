"""Read, validate, resample, and organize sensor recordings."""

import glob
import os

import numpy as np
import pandas as pd

from config import ACTIVITIES, DATA_ROOT, RESULTS_DIR, TARGET_FS


def find_time_column(df):
    for column in df.columns:
        normalized = column.lower()
        if any(key in normalized for key in ("time", "stamp", "sec")):
            return column
    return df.columns[0]


def parse_time_column(series):
    if pd.api.types.is_datetime64_any_dtype(series):
        elapsed = (series - series.iloc[0]) / pd.Timedelta(seconds=1)
        return elapsed.values.astype(float)
    if series.dtype == object:
        try:
            parsed = pd.to_datetime(series, errors="raise")
            return ((parsed - parsed.iloc[0]) / pd.Timedelta(seconds=1)).values
        except Exception:
            pass

    values = series.values.astype(float)
    difference = np.median(np.diff(values[:min(100, len(values))]))
    if not np.isfinite(difference) or difference == 0:
        difference = 1.0
    if difference > 1.0:
        values = values / 1000.0
    return values - values[0]


def find_sensor_columns(df, time_column):
    columns = [
        column for column in df.columns
        if column != time_column and pd.api.types.is_numeric_dtype(df[column])
    ]
    accelerometer, gyroscope = [], []
    for column in columns:
        normalized = column.lower().replace("_", "").replace(" ", "").replace("-", "")
        if (
            "gyro" in normalized
            or normalized in ("gx", "gy", "gz")
            or normalized.startswith("gx")
            or normalized.startswith("gyro")
        ):
            gyroscope.append(column)
        elif (
            "acc" in normalized
            or normalized in ("ax", "ay", "az", "x", "y", "z")
        ):
            accelerometer.append(column)

    if not accelerometer and not gyroscope:
        if len(columns) >= 3:
            accelerometer = columns[:3]
        if len(columns) >= 6:
            gyroscope = columns[3:6]
    return accelerometer[:3], gyroscope[:3]


def load_recording(path):
    df = pd.read_csv(path)
    time_column = find_time_column(df)
    times = parse_time_column(df[time_column])
    accelerometer, gyroscope = find_sensor_columns(df, time_column)
    return times, df, accelerometer, gyroscope


def estimate_fs(times):
    if len(times) < 2:
        return 0.0
    intervals = np.diff(times)
    intervals = intervals[np.isfinite(intervals) & (intervals > 0)]
    if len(intervals) == 0:
        return 0.0
    return 1.0 / np.median(intervals)


def resample_recording(times, df, sensor_columns, target_fs=TARGET_FS):
    finite_mask = np.isfinite(times)
    times = times[finite_mask]
    values = df.loc[finite_mask, sensor_columns].values.astype(float)
    monotonic_mask = np.concatenate([[True], np.diff(times) > 0])
    times = times[monotonic_mask]
    values = values[monotonic_mask]
    if len(times) < 2:
        return None, None

    duration = times[-1] - times[0]
    sample_count = int(np.floor(duration * target_fs)) + 1
    new_times = np.linspace(
        times[0], times[0] + (sample_count - 1) / target_fs, sample_count
    )
    resampled = np.zeros((sample_count, len(sensor_columns)))
    for index in range(len(sensor_columns)):
        resampled[:, index] = np.interp(new_times, times, values[:, index])
    return new_times, pd.DataFrame(resampled, columns=sensor_columns)


def discover_recordings(data_root=DATA_ROOT):
    recordings = {}
    for activity in ACTIVITIES:
        folder = os.path.join(data_root, activity)
        if not os.path.isdir(folder):
            print(f"[warn] Missing folder: {folder}")
            recordings[activity] = []
            continue
        recordings[activity] = sorted(glob.glob(os.path.join(folder, "*.csv")))
        print(f"{activity}: {len(recordings[activity])} file(s)")
    return recordings


def split_recordings(recordings):
    train_files = {activity: [] for activity in ACTIVITIES}
    test_files = {activity: [] for activity in ACTIVITIES}
    for activity, files in recordings.items():
        if not files:
            continue
        if len(files) >= 2:
            test_files[activity] = [files[-1]]
            train_files[activity] = files[:-1]
        else:
            print(f"[warn] Only 1 file for {activity}; using it for train and test")
            train_files[activity] = files
            test_files[activity] = files

    split_rows = []
    for activity in ACTIVITIES:
        split_rows.extend(
            {"activity": activity, "file": os.path.basename(path), "split": "train"}
            for path in train_files[activity]
        )
        split_rows.extend(
            {"activity": activity, "file": os.path.basename(path), "split": "test"}
            for path in test_files[activity]
        )
    pd.DataFrame(split_rows).to_csv(
        os.path.join(RESULTS_DIR, "recording_split.csv"), index=False
    )
    return train_files, test_files


def save_sampling_rate_report(recordings):
    rows = []
    for activity, files in recordings.items():
        for path in files:
            times, df, _, gyroscope = load_recording(path)
            rows.append({
                "activity": activity,
                "file": os.path.basename(path),
                "n_samples": len(times),
                "duration_s": float(times[-1] - times[0]) if len(times) > 1 else 0.0,
                "original_fs_hz": estimate_fs(times),
                "has_gyro": len(gyroscope) > 0,
            })
    pd.DataFrame(rows).to_csv(
        os.path.join(RESULTS_DIR, "sampling_rate_report.csv"), index=False
    )
    print("\nSampling rate report saved.")
