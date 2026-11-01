"""Gaussian HMM fitting, state labeling, and model serialization."""

import json
import os
import pickle

import numpy as np
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import linear_sum_assignment

from config import ACTIVITIES, N_STATES, RESULTS_DIR


def train_hmm(X_train, lengths_train, y_train_windows):
    activity_means = np.zeros((N_STATES, X_train.shape[1]))
    activity_covariances = np.zeros((N_STATES, X_train.shape[1]))
    for index, activity in enumerate(ACTIVITIES):
        mask = y_train_windows == activity
        if mask.sum() == 0:
            activity_means[index] = np.zeros(X_train.shape[1])
            activity_covariances[index] = np.ones(X_train.shape[1])
        else:
            activity_means[index] = X_train[mask].mean(axis=0)
            activity_covariances[index] = X_train[mask].var(axis=0) + 1e-3

    transition_init = np.full(
        (N_STATES, N_STATES), 0.05 / (N_STATES - 1)
    )
    np.fill_diagonal(transition_init, 0.95)
    start_probability_init = np.full(N_STATES, 1.0 / N_STATES)

    model = GaussianHMM(
        n_components=N_STATES,
        covariance_type="diag",
        n_iter=200,
        tol=1e-4,
        random_state=42,
        init_params="",
    )
    model.startprob_ = start_probability_init
    model.transmat_ = transition_init
    model.means_ = activity_means.copy()
    model.covars_ = activity_covariances.copy()

    print("\nTraining HMM with Baum-Welch...")
    try:
        model.fit(X_train, lengths_train)
    except Exception as error:
        print(f"[warn] fit error: {error}; trying with init_params='stmc'")
        model = GaussianHMM(
            n_components=N_STATES,
            covariance_type="diag",
            n_iter=200,
            tol=1e-4,
            random_state=42,
        )
        model.fit(X_train, lengths_train)

    cost = np.zeros((N_STATES, N_STATES))
    for state in range(N_STATES):
        for activity_index in range(N_STATES):
            cost[state, activity_index] = np.linalg.norm(
                model.means_[state] - activity_means[activity_index]
            )
    row_indices, column_indices = linear_sum_assignment(cost)
    state_to_activity = {
        int(state): ACTIVITIES[int(activity_index)]
        for state, activity_index in zip(row_indices, column_indices)
    }

    print("\nState -> Activity mapping:")
    for state, activity in state_to_activity.items():
        print(f"  state {state} -> {activity}")
    return model, state_to_activity


def save_model(model, feature_columns):
    with open(os.path.join(RESULTS_DIR, "trained_activity_hmm.pkl"), "wb") as file:
        pickle.dump(model, file)

    parameters = {
        "n_states": int(model.n_components),
        "covariance_type": model.covariance_type,
        "activities": ACTIVITIES,
        "feature_names": feature_columns,
        "means": model.means_.tolist(),
        "covars": np.asarray(model.covars_).tolist(),
        "transmat": model.transmat_.tolist(),
        "startprob": model.startprob_.tolist(),
        "log_likelihood_history": [
            float(value)
            for value in (getattr(model.monitor_, "history", []) or [])
        ],
        "converged": bool(getattr(model.monitor_, "converged", False)),
    }
    with open(os.path.join(RESULTS_DIR, "hmm_parameters.json"), "w") as file:
        json.dump(parameters, file, indent=2)
