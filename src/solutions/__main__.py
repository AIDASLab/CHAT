import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.constants import TUNING_BUDGET
from src.solutions import postprocess_results, print_optimal_hyperparameters
from src.solutions.brute_force.run import run as brute_force
from src.solutions.random_search.run import run as random_search
from src.solutions.our_solution.run import run as our_solution
from src.solutions.vd_tuner.run import run as vd_tuner

IMPLS = [
    "hnswlib", 
    "faiss",
]
DATASETS = [
    "nytimes-256-angular",
    "glove-100-angular",
    "sift-128-euclidean",
    "youtube-1024-angular",
    "msmarco-384-angular",
    "dbpediaentity-768-angular",
]
RECALL_MINS = [
    0.90,
    0.95,
]
SOLUTIONS = [
    (brute_force, "brute_force"),
    (random_search, "random_search"),
    (vd_tuner, "vd_tuner"),
    (our_solution, "our_solution"),
]

def run_single_combination(impl, dataset, recall_min, solution_func, solution_name):
    print(f"\n[START RUNNING] {impl} | {dataset}")
    results = solution_func(impl=impl, dataset=dataset, recall_min=recall_min)
    print_optimal_hyperparameters(results, recall_min=recall_min)
    postprocess_results(
        results, 
        solution=solution_name,
        impl=impl, 
        dataset=dataset, 
        recall_min=recall_min, 
        tuning_budget=TUNING_BUDGET if solution_name != "brute_force" else float("inf")
    )

if __name__ == "__main__":
    combos = list(itertools.product(RECALL_MINS, IMPLS, DATASETS, SOLUTIONS))
    with ProcessPoolExecutor(max_workers=32) as executor:
        futures = [
            executor.submit(
                run_single_combination, impl, dataset, recall_min, solution_func, solution_name
            )
            for (recall_min, impl, dataset, (solution_func, solution_name)) in combos
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error occurred: {e}")