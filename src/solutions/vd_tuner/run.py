import os

from src.solutions import postprocess_results, print_optimal_hyperparameters

from src.solutions.vd_tuner.optimizer_pobo_sa import PollingBayesianOptimization
from src.solutions.vd_tuner.utils import MockEnv
from src.constants import IMPL, DATASET, SEED, TUNING_BUDGET, RECALL_MIN
from data.ground_truths import GroundTruth

from joblib import Memory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USE_EFS = False

# memory = Memory("/tmp/vd_tuner_cache", verbose=0)
# @memory.cache
def run(impl=IMPL, dataset=DATASET, recall_min=None, qps_min=None, tuning_budget=TUNING_BUDGET, search_efS=True, sampling_count=None, env=(TUNING_BUDGET, SEED)):
    if not recall_min and not qps_min:
        raise ValueError("Either recall_min or qps_min must be specified.")
    if recall_min and qps_min:
        raise ValueError("Only one of recall_min or qps_min should be specified.")
    gd = GroundTruth(impl=impl, dataset=dataset, sampling_count=sampling_count)
    knob_path = os.path.join(BASE_DIR, "params/whole_param.json")
    env = MockEnv(model=gd, knob_path=knob_path, tuning_budget=tuning_budget, search_efS=search_efS)
    model = PollingBayesianOptimization(env, seed=SEED)
    model.init_sample()
    try:
        while True:
            model.step()
    except TimeoutError as e:
        print(f"Tuning time out")
    except KeyboardInterrupt as e:
        print(f"Tuning interrupted by Ctrl+C")
    ## END OF TUNING ##
    results = [] 
    for i in range(len(env.X_record)):
        M, efC, efS = env.X_record[i]
        recall, qps = env.Y_record[i]
        _recall, _qps, total_time, build_time, index_size = gd.get(M, efC, efS, tracking_time=False)
        if abs(recall - _recall) > 1e-5 or abs(qps - _qps) > 1e-5:
            raise ValueError(f"Recall or QPS mismatch: {recall}/{_recall}, {qps}/{_qps}")
        results.append(((M, efC, efS), (env.T_record[i], recall, qps, total_time, build_time, index_size)))
    return results
