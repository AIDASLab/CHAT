import os

DATA_DIR = os.environ.get("DATA_DIR", "./data")
RESULTS_DIR = os.environ.get("RESULTS_DIR", "./results")
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# EXPERIMENT_CONFIGS
MAX_THREADS = 64 - 8
ITERS = 10
DEFAULT_M = list(range(4, 64+1))
DEFAULT_EFC = list(range(16, 1024+1, 1))
DEFAULT_PARAMS = []
DEFAULT_EFS = list(range(10, 1024+1, 1))

INTERP_KIND="linear"