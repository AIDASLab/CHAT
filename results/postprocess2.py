import itertools
import multiprocessing
from functools import partial
import numpy as np
from scipy.stats import spearmanr

from src.constants import TUNING_BUDGET, MAX_SAMPLING_COUNT, SEED
from src.utils import (get_local_optimal_hyperparameter, get_optimal_hyperparameter, 
                       plot_efS_3d, plot_multi_accumulated_timestamp, 
                       plot_searched_points_3d, plot_timestamp, 
                       save_search_results, load_search_results)
from data.ground_truths.get_qps_dataset import get_qps_metrics_dataset

def postprocess_results(solution, impl, dataset,
                        recall_min=None, qps_min=None, sampling_count=MAX_SAMPLING_COUNT, seed=SEED, tuning_budget=TUNING_BUDGET):
    results = load_search_results(solution, filename=f"{solution}_{impl}_{dataset}_{recall_min}r_{qps_min}q.csv", 
                                  seed=seed, sampling_count=sampling_count)
    plot_timestamp(
        results, 
        solution=solution, filename=f"{solution}_{impl}_{dataset}_{recall_min}r_{qps_min}q.png", 
        recall_min=recall_min, qps_min=qps_min, tuning_budget=tuning_budget, seed=seed, sampling_count=sampling_count
    )
    plot_multi_accumulated_timestamp(
        {solution: results},
        dirname=solution,
        filename=f"{solution}_multi_accumulated_timestamp_{impl}_{dataset}_{recall_min}r_{qps_min}q.png", 
        recall_min=recall_min, qps_min=qps_min, tuning_budget=tuning_budget, seed=seed, sampling_count=sampling_count
    )
    plot_searched_points_3d(
        results, 
        solution=solution, 
        filename=f"{solution}_searched_points_3d_{impl}_{dataset}_{recall_min}r_{qps_min}q.png", 
        recall_min=recall_min, qps_min=qps_min, tuning_budget=tuning_budget, seed=seed, sampling_count=sampling_count
    )
    plot_efS_3d(
        results, 
        solution=solution, 
        filename=f"{solution}_searched_points_3d_{impl}_{dataset}_{recall_min}r_{qps_min}q.png", 
        recall_min=recall_min, qps_min=qps_min, tuning_budget=tuning_budget, seed=seed, sampling_count=sampling_count
    )

def main():
    SOLUTIONS = [
        "brute_force",
        "our_solution",
        "grid_search",
        "random_search",
        "vd_tuner",
        "optuna",
    ]
    IMPLS = [
        "faiss",
        "hnswlib",
        "milvus",
    ]
    DATASETS = [
        "nytimes-256-angular",
        "glove-100-angular",
        "sift-128-euclidean",
        "youtube-1024-angular",
        "deep1M-256-angular",
    ]
    SAMPLING_COUNT = [
        10,
    ]
    RECALL_MINS = [
        0.95, 
    ]
    SEED = "42"
    
    tasks = []

    all_iters = itertools.product(SOLUTIONS, IMPLS, DATASETS, SAMPLING_COUNT)
    print("--- Preparing tasks for parallel execution ---")
    for solution, impl, dataset, sampling_count in all_iters:
        for recall_min in RECALL_MINS:
            task_args = (solution, impl, dataset, recall_min, None, sampling_count, SEED)
            tasks.append(task_args)
            print(f"  - Queued task: {impl}, {dataset}, recall_min={recall_min}, sampling={sampling_count}")

        for qps_min in get_qps_metrics_dataset(impl, dataset):
            task_args = (solution, impl, dataset, None, qps_min, sampling_count, SEED)
            tasks.append(task_args)
            print(f"  - Queued task: {impl}, {dataset}, qps_min={qps_min}, sampling={sampling_count}")

    print("\n--- All tasks prepared. Starting parallel processing. ---")

    try:
        num_processes = multiprocessing.cpu_count() - 1 if multiprocessing.cpu_count() > 1 else 1
        print(f"Creating a pool of {num_processes} worker processes for {len(tasks)} tasks.")
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.starmap(postprocess_results, tasks)
    except Exception as e:
        print(f"An error occurred during multiprocessing: {e}")

    print("\n--- Parallel processing finished. ---")


if __name__ == "__main__":
    main()