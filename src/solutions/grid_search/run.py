from src.solutions import postprocess_results, print_optimal_hyperparameters
from data.ground_truths import GroundTruth
from src.constants import EFC_MAX, EFC_MIN, EFS_MAX, EFS_MIN, M_MAX, M_MIN 
from src.constants import IMPL, DATASET, SEED, TUNING_BUDGET, RECALL_MIN
import random
from tqdm import tqdm
from joblib import Memory

STEP_M = 4
STEP_EFC = 64
STEP_EFS = 128

# memory = Memory("/tmp/grid_search_cache", verbose=0)
# @memory.cache
def run(impl=IMPL, dataset=DATASET, recall_min=None, qps_min=None, tuning_budget=TUNING_BUDGET, sampling_count=None, env=(TUNING_BUDGET, SEED)):
    assert (recall_min is None) != (qps_min is None), "Only one of recall_min or qps_min should be set."
    gd = GroundTruth(impl=impl, dataset=dataset, sampling_count=sampling_count)
    random.seed(SEED)
    results = []
    candidates = [
        (M, efC, efS)
        for M in range(M_MIN, M_MAX + 1, STEP_M)
        for efC in range(EFC_MIN, EFC_MAX + 1, STEP_EFC)
        for efS in range(EFS_MIN, EFS_MAX + 1, STEP_EFS)
        if M <= efC
    ]
    # random.shuffle(candidates)  # Shuffle candidates to ensure randomness in the search order
    for M, efC, efS in tqdm(candidates, desc=f"GridSearch[{impl}|{dataset}]", unit="config"):
        recall, qps, total_time, build_time, index_size = gd.get(M=M, efC=efC, efS=efS)
        if gd.tuning_time > tuning_budget:
            print(f"Tuning time out at {gd.tuning_time:.2f}s")
            break
        results.append(((M, efC, efS), (gd.tuning_time, recall, qps, total_time, build_time, index_size)))
    return results
