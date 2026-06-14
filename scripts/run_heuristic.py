import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import itertools

from scripts import run_experiments
from scripts import IMPLS, DATASETS, SOLUTIONS, RECALL_MINS, TUNING_BUDGET, SEED

from src.solutions.our_solution.stats import Stats
# from src.solutions.our_solution.run import run as run 
from src.solutions.our_solution.run_heuristic3 import run
from data.ground_truths.get_qps_dataset import get_qps_metrics_dataset
from src.solutions import postprocess_results
from src.utils import is_already_saved
    
# SOLUTIONS = [
#     (run, "test_solution"),
#     (run_heuristic2, "test_solution2"),
# ]

def worker_function(
        impl, dataset, recall_min=None, qps_min=None, sampling_count=10    
    ):

    # run.py
    _, search_stats, efC_stats = run(
        impl=impl, dataset=dataset, recall_min=recall_min, qps_min=qps_min,
        sampling_count=sampling_count, stats=True
    )
    return search_stats.stats["exploration_ratio"], efC_stats

if __name__ == "__main__":
    IMPLS = [
        "hnswlib",
        "faiss",
    ]
    DATASETS = [
        "nytimes-256-angular",
        "glove-100-angular",
        "sift-128-euclidean",
        "youtube-1024-angular",
        "deep1M-256-angular",
    ]
    RECALL_MIN = 0.95
    QPS_MIN = "q75"
    output = ""
    for dataset in DATASETS:
        ratio_candidates_list = []
        tuning_time_list = []
        _ratio_candidates, _tuning_time = worker_function(
            "hnswlib", dataset, RECALL_MIN, None
        )
        ratio_candidates_list.append(_ratio_candidates)
        tuning_time_list.append(_tuning_time)
        _ratio_candidates, _tuning_time = worker_function(
            "faiss", dataset, RECALL_MIN, None
        )
        ratio_candidates_list.append(_ratio_candidates)
        tuning_time_list.append(_tuning_time)
        _ratio_candidates, _tuning_time = worker_function(
            "hnswlib", dataset, None, 
            get_qps_metrics_dataset(impl="hnswlib", dataset=dataset, ret_dict=True)[QPS_MIN], 
        )
        ratio_candidates_list.append(_ratio_candidates)
        tuning_time_list.append(_tuning_time)
        _ratio_candidates, _tuning_time = worker_function(
            "faiss", dataset, None, 
            get_qps_metrics_dataset(impl="faiss", dataset=dataset, ret_dict=True)[QPS_MIN], 
        )
        ratio_candidates_list.append(_ratio_candidates)
        tuning_time_list.append(_tuning_time)
        ####
        output += f"{dataset},{sum(ratio_candidates_list)/len(ratio_candidates_list)},{sum(tuning_time_list)/len(tuning_time_list)}\n"
    print(output)