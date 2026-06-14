from src.constants import DATASET, IMPL, TUNING_BUDGET, RECALL_MIN, M_MIN, M_MAX, EFC_MIN, EFC_MAX, EFS_MIN, EFS_MAX, SEED
from src.solutions import postprocess_results, print_optimal_hyperparameters
from data.ground_truths import GroundTruth
from joblib import Memory

def run(impl=IMPL, dataset=DATASET, recall_min=None , qps_min=None, tuning_budget=TUNING_BUDGET, sampling_count=None, env=(TUNING_BUDGET, SEED)):
    assert (recall_min is None) != (qps_min is None), "Only one of recall_min or qps_min should be set."
    gd = GroundTruth(impl, dataset)
    results = []
    for M in range(M_MIN, M_MAX+1, 1):
        for efC in range(EFC_MIN, EFC_MAX+1, 1):
            efS = gd.get_efS(M, efC, target_recall=recall_min, target_qps=qps_min, method="binary")
            recall, qps, total_time, build_time, index_size = gd.get(M=M, efC=efC, efS=efS, tracking_time=False)
            gd.tuning_time += 100.0
            results.append(((M, efC, efS), (gd.tuning_time, recall, qps, total_time, build_time, index_size)))
    return results
