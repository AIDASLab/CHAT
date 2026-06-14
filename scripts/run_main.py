import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import itertools

from scripts import run_experiments
from scripts import IMPLS, DATASETS, SOLUTIONS, RECALL_MINS, QPS_MINS

from src.constants import MAX_SAMPLING_COUNT
from src.solutions.brute_force.run import run as brute_force
from src.solutions.random_search.run import run as random_search
from src.solutions.vd_tuner.run import run as vd_tuner
from src.solutions.our_solution.run import run as our_solution
# from src.solutions.our_solution.run_old import run as test_solution
from src.solutions.grid_search.run import run as grid_search
from src.solutions.optuna.run import run as optuna
from src.solutions.nsga.run import run as nsga
from src.solutions.eci.run import run as eci
from data.ground_truths.get_qps_dataset import get_qps_metrics_dataset

if __name__ == "__main__":
    IMPLS = [
        "hnswlib",
        "faiss",
    ]
    DATASETS = [
        "nytimes-256-angular",
        "glove-100-angular",
        "sift-128-euclidean",
        "deep1M-256-angular",
        "youtube-1024-angular",
    ]
    SOLUTIONS = [
        (brute_force, "brute_force"),
        (grid_search, "grid_search"),
        (random_search, "random_search"),
        (optuna, "optuna"),
        (nsga, "nsga"),
        (vd_tuner, "vd_tuner"),
        (our_solution, "our_solution"),
        (eci, "eci"),
    ]
    RECALL_MINS = [
        0.95,
    ]
    SAMPLING_COUNT = [
        10,
    ]

    tasks = []
    # Case 1: when recall_min is active
    for impl, dataset, (solution_func, solution_name), sampling_count, recall_min in itertools.product(
        IMPLS, DATASETS, SOLUTIONS, SAMPLING_COUNT, RECALL_MINS
    ):
        tasks.append((impl, dataset, solution_func, solution_name, recall_min, None, sampling_count))
    # Case 2: when qps_min is active
    for impl, dataset, (solution_func, solution_name), sampling_count in itertools.product(
        IMPLS, DATASETS, SOLUTIONS, SAMPLING_COUNT
    ):
        for qps_min in get_qps_metrics_dataset(impl, dataset):
            tasks.append((impl, dataset, solution_func, solution_name, None, qps_min, sampling_count))
    run_experiments(tasks=tasks)
