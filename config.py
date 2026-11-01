"""Shared configuration for the activity-recognition pipeline."""

ACTIVITIES = ["still", "standing", "walking", "jumping"]
DATA_ROOT = "data"
RESULTS_DIR = "results"
TARGET_FS = 50.0
WINDOW_SEC = 2.0
STRIDE_SEC = 1.0
N_STATES = len(ACTIVITIES)
