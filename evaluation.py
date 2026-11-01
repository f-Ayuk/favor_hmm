"""Classification metrics and prediction exports."""

import os

import numpy as np
import pandas as pd

from config import ACTIVITIES, RESULTS_DIR


def compute_metrics(y_true, y_pred, labels=ACTIVITIES):
    results = []
    accuracy = float(np.mean(y_true == y_pred)) if len(y_true) else 0.0
    results.append({
        "activity": "OVERALL",
        "sensitivity": accuracy,
        "specificity": accuracy,
        "accuracy": accuracy,
        "n_windows": int(len(y_true)),
    })
    for activity in labels:
        true_positive = int(np.sum((y_pred == activity) & (y_true == activity)))
        false_negative = int(np.sum((y_pred != activity) & (y_true == activity)))
        false_positive = int(np.sum((y_pred == activity) & (y_true != activity)))
        true_negative = int(np.sum((y_pred != activity) & (y_true != activity)))
        sensitivity = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative else 0.0
        )
        specificity = (
            true_negative / (true_negative + false_positive)
            if true_negative + false_positive else 0.0
        )
        activity_accuracy = (
            (true_positive + true_negative)
            / (true_positive + true_negative + false_positive + false_negative)
            if true_positive + true_negative + false_positive + false_negative
            else 0.0
        )
        results.append({
            "activity": activity,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "accuracy": activity_accuracy,
            "n_windows": int(np.sum(y_true == activity)),
        })
    return pd.DataFrame(results)


def save_predictions(sequences, predicted_labels, state_predictions):
    if not len(state_predictions):
        return

    rows = []
    prediction_index = 0
    for sequence in sequences:
        if sequence["split"] != "test":
            continue
        for window_index in range(len(sequence["features"])):
            rows.append({
                "file": sequence["file"],
                "activity_true": sequence["activity"],
                "window_idx": window_index,
                "state": int(state_predictions[prediction_index + window_index]),
                "activity_pred": predicted_labels[
                    prediction_index + window_index
                ],
            })
        prediction_index += len(sequence["features"])

    pd.DataFrame(rows).to_csv(
        os.path.join(RESULTS_DIR, "unseen_predictions.csv"), index=False
    )
