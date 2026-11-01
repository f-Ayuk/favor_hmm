"""Generate plots for model evaluation and sensor recordings."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from config import ACTIVITIES, RESULTS_DIR, STRIDE_SEC, TARGET_FS
from data_processing import load_recording, resample_recording


def _save_transition_matrix(model):
    transition_df = pd.DataFrame(
        model.transmat_, index=ACTIVITIES, columns=ACTIVITIES
    )
    transition_df.to_csv(os.path.join(RESULTS_DIR, "transition_matrix.csv"))
    plt.figure(figsize=(6, 5))
    sns.heatmap(transition_df, annot=True, fmt=".3f", cmap="Blues", cbar=True)
    plt.title("HMM Transition Matrix")
    plt.ylabel("From")
    plt.xlabel("To")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "transition_matrix.png"), dpi=120)
    plt.close()


def _save_confusion_matrix(y_true, y_pred):
    matrix = (
        confusion_matrix(y_true, y_pred, labels=ACTIVITIES)
        if len(y_true)
        else np.zeros((len(ACTIVITIES), len(ACTIVITIES)), dtype=int)
    )
    matrix_df = pd.DataFrame(matrix, index=ACTIVITIES, columns=ACTIVITIES)
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix_df, annot=True, fmt="d", cmap="Oranges", cbar=True)
    plt.title("Confusion Matrix (Unseen Test)")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=120)
    plt.close()


def _save_feature_distributions(features_df, feature_columns):
    candidates = [
        column for column in feature_columns
        if "std" in column or "rms" in column or "mean" in column
    ]
    selected = candidates[:6] if len(candidates) >= 6 else feature_columns[:6]
    if not selected:
        return

    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for index, feature in enumerate(selected[:6]):
        data = [
            features_df.loc[features_df["activity"] == activity, feature].values
            for activity in ACTIVITIES
        ]
        axes[index].boxplot(data, tick_labels=ACTIVITIES)
        axes[index].set_title(feature)
        axes[index].tick_params(axis="x", rotation=30)
    plt.suptitle("Feature Distributions by Activity")
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "feature_distributions.png"), dpi=120
    )
    plt.close()


def _save_emission_plots(model, state_to_activity):
    plt.figure(figsize=(14, 6))
    plt.imshow(model.means_, aspect="auto", cmap="viridis")
    plt.colorbar(label="value")
    plt.title("Emission Means per State")
    plt.xlabel("Feature index")
    plt.ylabel("State")
    plt.yticks(
        range(len(ACTIVITIES)),
        [f"s{index} ({state_to_activity.get(index, '?')})"
         for index in range(len(ACTIVITIES))],
    )
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "emission_means.png"), dpi=120)
    plt.close()

    variances = np.diagonal(
        np.asarray(model.covars_), axis1=1, axis2=2
    )
    plt.figure(figsize=(14, 6))
    plt.imshow(variances, aspect="auto", cmap="magma")
    plt.colorbar(label="variance")
    plt.title("Emission Variances per State")
    plt.xlabel("Feature index")
    plt.ylabel("State")
    plt.yticks(
        range(len(ACTIVITIES)),
        [f"s{index} ({state_to_activity.get(index, '?')})"
         for index in range(len(ACTIVITIES))],
    )
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "emission_variances.png"), dpi=120)
    plt.close()


def _save_convergence_plot(model):
    history = getattr(model.monitor_, "history", []) or []
    if len(history) > 1:
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(history) + 1), history, marker="o")
        plt.title("Baum-Welch Convergence")
        plt.xlabel("Iteration")
        plt.ylabel("Log-likelihood")
        plt.grid(True)
    else:
        plt.figure()
        plt.text(
            0.5, 0.5, "Convergence history unavailable",
            ha="center", va="center",
        )
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "baum_welch_convergence.png"), dpi=120
    )
    plt.close()


def _save_raw_signal_plots(train_files, test_files):
    for activity in ACTIVITIES:
        paths = train_files[activity] or test_files[activity]
        if not paths:
            continue
        path = paths[0]
        try:
            times, df, accelerometer, gyroscope = load_recording(path)
            new_times, resampled = resample_recording(
                times, df, accelerometer + gyroscope, TARGET_FS
            )
        except Exception:
            continue
        if new_times is None:
            continue

        figure, axes = plt.subplots(
            len(accelerometer), 1,
            figsize=(12, 2.5 * len(accelerometer)),
            sharex=True,
        )
        if len(accelerometer) == 1:
            axes = [axes]
        for index, column in enumerate(accelerometer):
            axes[index].plot(new_times, resampled[column].values)
            axes[index].set_ylabel(column)
        axes[-1].set_xlabel("Time (s)")
        plt.suptitle(f"Raw signal - {activity} - {os.path.basename(path)}")
        plt.tight_layout()
        plt.savefig(
            os.path.join(RESULTS_DIR, f"raw_{activity}_1.png"), dpi=120
        )
        plt.close()


def _save_decoded_sequences(sequences, state_predictions, state_to_activity):
    prediction_index = 0
    for sequence_index, sequence in enumerate(sequences):
        if sequence["split"] != "test":
            continue
        window_count = len(sequence["features"])
        true_sequence = [ACTIVITIES.index(sequence["activity"])] * window_count
        predicted_sequence = [
            ACTIVITIES.index(state_to_activity[int(state)])
            for state in state_predictions[
                prediction_index:prediction_index + window_count
            ]
        ]
        prediction_index += window_count
        window_times = np.arange(window_count) * STRIDE_SEC
        plt.figure(figsize=(12, 3))
        plt.step(
            window_times, true_sequence, where="post",
            label="True", color="green",
        )
        plt.step(
            window_times, predicted_sequence, where="post",
            label="Predicted", color="red", alpha=0.7,
        )
        plt.yticks(range(len(ACTIVITIES)), ACTIVITIES)
        plt.xlabel("Window start time (s)")
        plt.title(
            f"Decoded sequence - {sequence['activity']} - {sequence['file']}"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                RESULTS_DIR,
                f"decoded_sequence_{sequence['activity']}_{sequence_index}.png",
            ),
            dpi=120,
        )
        plt.close()


def create_visualizations(
    model,
    y_true,
    y_pred,
    features_df,
    feature_columns,
    state_to_activity,
    train_files,
    test_files,
    sequences,
    state_predictions,
):
    _save_transition_matrix(model)
    _save_confusion_matrix(y_true, y_pred)
    _save_feature_distributions(features_df, feature_columns)
    _save_emission_plots(model, state_to_activity)
    _save_convergence_plot(model)
    _save_raw_signal_plots(train_files, test_files)
    _save_decoded_sequences(sequences, state_predictions, state_to_activity)
