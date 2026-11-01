"""Coordinate dataset preparation, HMM training, evaluation, and plots."""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import ACTIVITIES, DATA_ROOT, RESULTS_DIR, TARGET_FS
from data_processing import (
    discover_recordings,
    load_recording,
    resample_recording,
    save_sampling_rate_report,
    split_recordings,
)
from evaluation import compute_metrics, save_predictions
from feature_extraction import window_features
from hmm_model import save_model, train_hmm
from visualization import create_visualizations


warnings.filterwarnings("ignore")


def _extract_dataset(train_files, test_files):
    feature_rows = []
    sequences = []
    duration_summary = {
        activity: {
            "n_recs": 0,
            "total_duration_s": 0.0,
            "n_windows": 0,
        }
        for activity in ACTIVITIES
    }

    for split, file_map in (("train", train_files), ("test", test_files)):
        for activity in ACTIVITIES:
            for path in file_map[activity]:
                try:
                    times, df, accelerometer, gyroscope = load_recording(path)
                except Exception as error:
                    print(f"[skip] {path}: {error}")
                    continue
                if not accelerometer:
                    print(f"[skip] {path}: no accelerometer columns")
                    continue

                sensor_columns = accelerometer + gyroscope
                new_times, resampled = resample_recording(
                    times, df, sensor_columns, TARGET_FS
                )
                if new_times is None:
                    continue

                features = window_features(
                    resampled, accelerometer, gyroscope
                )
                if features.empty:
                    continue

                filename = os.path.basename(path)
                for window_index in range(len(features)):
                    row = features.iloc[window_index].to_dict()
                    row.update({
                        "activity": activity,
                        "file": filename,
                        "split": split,
                        "window_idx": window_index,
                    })
                    feature_rows.append(row)

                duration = float(new_times[-1] - new_times[0])
                sequences.append({
                    "activity": activity,
                    "file": filename,
                    "features": features.values.astype(float),
                    "feature_names": list(features.columns),
                    "split": split,
                    "duration": duration,
                })
                duration_summary[activity]["n_recs"] += 1
                duration_summary[activity]["total_duration_s"] += duration
                duration_summary[activity]["n_windows"] += len(features)

    return feature_rows, sequences, duration_summary


def _save_dataset_summary(duration_summary):
    rows = []
    for activity in ACTIVITIES:
        summary = duration_summary[activity]
        rows.append({
            "activity": activity,
            "n_recordings": summary["n_recs"],
            "total_duration_s": summary["total_duration_s"],
            "n_windows": summary["n_windows"],
            "meets_90s": summary["total_duration_s"] >= 90.0,
        })
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(
        os.path.join(RESULTS_DIR, "dataset_summary.csv"), index=False
    )
    print("\nDataset summary:")
    print(summary_df.to_string(index=False))


def _scale_features(features_df, sequences):
    metadata_columns = {"activity", "file", "split", "window_idx"}
    feature_columns = [
        column for column in features_df.columns
        if column not in metadata_columns
    ]
    values = features_df[feature_columns].values.astype(float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    train_mask = features_df["split"].values == "train"
    scaler = StandardScaler().fit(values[train_mask])
    scaled_values = scaler.transform(values)
    scaled_values = np.nan_to_num(
        scaled_values, nan=0.0, posinf=0.0, neginf=0.0
    )

    train_sequences = []
    train_lengths = []
    train_labels = []
    test_sequences = []
    test_lengths = []
    test_labels = []
    row_index = 0
    for sequence in sequences:
        window_count = len(sequence["features"])
        scaled_sequence = scaled_values[row_index:row_index + window_count]
        row_index += window_count
        if sequence["split"] == "train":
            train_sequences.append(scaled_sequence)
            train_lengths.append(window_count)
            train_labels.append([sequence["activity"]] * window_count)
        else:
            test_sequences.append(scaled_sequence)
            test_lengths.append(window_count)
            test_labels.append([sequence["activity"]] * window_count)

    return (
        scaler,
        feature_columns,
        train_sequences,
        train_lengths,
        train_labels,
        test_sequences,
        test_lengths,
        test_labels,
    )


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 60)
    print("Activity Recognition with Gaussian HMM")
    print("=" * 60)

    recordings = discover_recordings(DATA_ROOT)
    save_sampling_rate_report(recordings)
    train_files, test_files = split_recordings(recordings)
    print("\nExtracting features...")

    feature_rows, sequences, duration_summary = _extract_dataset(
        train_files, test_files
    )
    if not feature_rows:
        print("[error] No features extracted. Aborting.")
        return

    features_df = pd.DataFrame(feature_rows)
    features_df.to_csv(
        os.path.join(RESULTS_DIR, "all_extracted_features.csv"), index=False
    )
    _save_dataset_summary(duration_summary)

    (
        scaler,
        feature_columns,
        train_sequences,
        train_lengths,
        train_labels,
        test_sequences,
        test_lengths,
        test_labels,
    ) = _scale_features(features_df, sequences)
    with open(os.path.join(RESULTS_DIR, "feature_scaler.pkl"), "wb") as file:
        pickle.dump(scaler, file)

    if not train_sequences:
        print("[error] No training sequences. Aborting.")
        return

    X_train = np.vstack(train_sequences)
    lengths_train = np.array(train_lengths)
    y_train = np.concatenate(train_labels)
    print(
        f"\nTraining sequences: {len(train_sequences)} "
        f"(total windows: {len(X_train)})"
    )
    print(f"Test sequences: {len(test_sequences)}")

    model, state_to_activity = train_hmm(X_train, lengths_train, y_train)
    save_model(model, feature_columns)

    if test_sequences:
        X_test = np.vstack(test_sequences)
        lengths_test = np.array(test_lengths)
        y_test = np.concatenate(test_labels)
        state_predictions = model.predict(X_test, lengths_test)
        predicted_labels = np.array([
            state_to_activity[state] for state in state_predictions
        ])
    else:
        X_test = np.empty((0, X_train.shape[1]))
        y_test = np.array([])
        state_predictions = np.array([], dtype=int)
        predicted_labels = np.array([])

    metrics_df = compute_metrics(y_test, predicted_labels, ACTIVITIES)
    metrics_df.to_csv(
        os.path.join(RESULTS_DIR, "evaluation_metrics.csv"), index=False
    )
    print("\nEvaluation metrics:")
    print(metrics_df.to_string(index=False))

    save_predictions(sequences, predicted_labels, state_predictions)
    create_visualizations(
        model=model,
        y_true=y_test,
        y_pred=predicted_labels,
        features_df=features_df,
        feature_columns=feature_columns,
        state_to_activity=state_to_activity,
        train_files=train_files,
        test_files=test_files,
        sequences=sequences,
        state_predictions=state_predictions,
    )

    overall = metrics_df[metrics_df["activity"] == "OVERALL"]
    if not overall.empty:
        accuracy = overall["accuracy"].values[0]
        print(f"\nOverall accuracy on unseen test windows: {accuracy:.3f}")
    print(f"\nAll outputs saved in: {RESULTS_DIR}")
