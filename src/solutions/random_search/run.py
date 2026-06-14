from src.solutions import postprocess_results, print_optimal_hyperparameters
from data.ground_truths import GroundTruth
from src.constants import IMPL, DATASET, SEED, TUNING_BUDGET, RECALL_MIN, M_MAX, M_MIN, EFC_MAX, EFC_MIN, EFS_MAX, EFS_MIN
import random
from joblib import Memory

# memory = Memory("/tmp/random_search_cache", verbose=0)
# @memory.cache
def run(impl=IMPL, dataset=DATASET, recall_min=None, qps_min=None, tuning_budget=TUNING_BUDGET, sampling_count=None, env=(TUNING_BUDGET, SEED)):
    assert (recall_min is None) != (qps_min is None), "Only one of recall_min or qps_min should be set."
    gd = GroundTruth(impl=impl, dataset=dataset, sampling_count=sampling_count)
    random.seed(SEED)
    results = []
    while True:
        M = random.randint(M_MIN, M_MAX)
        efC = random.randint(EFC_MIN, EFC_MAX)
        efS = random.randint(EFS_MIN, EFS_MAX)
        recall, qps, total_time, build_time, index_size = gd.get(M=M, efC=efC, efS=efS)
        if gd.tuning_time > tuning_budget:
            print(f"Tuning time out")
            break
        results.append(((M, efC, efS), (gd.tuning_time, recall, qps, total_time, build_time, index_size)))
    return results

