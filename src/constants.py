"""Shared experiment defaults for the CHAT benchmark runner."""

# HNSW search space.
M_MIN, M_MAX = 4, 64
EFS_MIN, EFS_MAX = 10, 1024
EFC_MIN, EFC_MAX = 8, 1024

# Default experiment target. Scripts may override these values per task.
IMPL = "hnswlib"
DATASET = "nytimes-256-angular"
RECALL_MIN = 0.95
QPS_MIN = None

# Reproducibility and budget defaults.
MAX_SAMPLING_COUNT = 10
TOLERANCE = 0.005
SEED = 42
TUNING_BUDGET = 3600 * 4
