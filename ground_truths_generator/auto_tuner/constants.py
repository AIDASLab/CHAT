import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(PACKAGE_DIR)

# Environment variables override these defaults for large shared datasets.
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(ROOT_DIR, "data"))
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(ROOT_DIR, "results"))

# EXPERIMENT_CONFIGS
MAX_THREADS = 64
ITERS = 10
DEFAULT_M = list(range(4, 64+1))
DEFAULT_EFC = list(range(8, 1024+1, 1))
DEFAULT_PARAMS = []
DEFAULT_EFS = list(range(10, 1024+1, 1))

INTERP_KIND="linear"
