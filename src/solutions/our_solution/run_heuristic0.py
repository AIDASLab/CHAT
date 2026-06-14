# hyperparameter_tuner.py

import math
import random
from typing import List, Tuple, Dict, Any

from src.constants import DATASET, IMPL, EFS_MIN, EFS_MAX, SEED, TUNING_BUDGET, RECALL_MIN, EFC_MIN, EFC_MAX, M_MIN, M_MAX
from src.solutions import postprocess_results, print_optimal_hyperparameters
from src.solutions.our_solution.utils import EfSGetterV3, EfSGetterV2
from src.solutions.our_solution.utils import EfCGetter, EfCGetterBase
from src.solutions.our_solution.stats import Stats
from data.ground_truths.ground_truth import GroundTruth

# --- Constants for the tuning algorithm ---
TERNARY_SEARCH_BASE = 2.5
TERNARY_SEARCH_FACTOR = 3
QPS_PERF_PENALTY = 0.95

class HyperparameterTuner:
    """
    Encapsulates the entire hyperparameter tuning process to avoid global state
    and improve code structure.
    """
    def __init__(self, ground_truth: GroundTruth, recall_min: float = None, qps_min: float = None, tuning_budget: float = TUNING_BUDGET):
        assert (recall_min is None) != (qps_min is None), "Only one of recall_min or qps_min should be set."
        
        self.ground_truth = ground_truth
        self.recall_min = recall_min
        self.qps_min = qps_min
        self.tuning_budget = tuning_budget
        
        # --- State variables ---
        self.results: List[Tuple[Tuple, Tuple]] = []
        self.stats: Stats = Stats(tuning_budget=tuning_budget, recall_min=recall_min, qps_min=qps_min)
        self.m_to_perf: List[Tuple[int, float]] = []
        self.searched_hp: set = set()    #* set of (M, efC, efS) tuples
        self.efS_getter = EfSGetterV3(mode="base")
        # self.efC_getter = EfCGetter()
        self.efC_getter = EfCGetterBase()

    def _get_perf(self, perf: Tuple[float, float]) -> float:
        """Returns the relevant performance metric (recall or QPS) based on the optimization goal."""
        # perf is a tuple of (recall, qps)
        return perf[0] if self.qps_min is not None else perf[1]

    def run_tuning(self) -> List[Tuple[Tuple, Tuple]]:
        """Executes the full tuning process, including exploration and exploitation phases."""
        print("--- Starting Exploration Phase ---")
        self._exploration_phase()
        self.stats.exploration_phase(self.results)
        remaining_budget = self.tuning_budget - self.ground_truth.tuning_time
        if remaining_budget > 0:
            print(f"\n--- Starting Exploitation Phase (Remaining Budget: {remaining_budget:.2f}s) ---")
            # self._exploitation_phase()
        self.stats.exploitation_phase(self.results)
        # print_optimal_hyperparameters(self.results, recall_min=self.recall_min, qps_min=self.qps_min)
        print("\n--- Tuning is Done! ---")
        return self.results

    def _exploration_phase(self):
        """
        Broadly searches the 'M' parameter space using a ternary search-like approach
        to identify promising regions.
        """
        m_left, m_right = M_MIN, M_MAX
        processed_m = set()

        try:
            # Ternary search for M
            while (m_right - m_left) > 3:
                if self.ground_truth.tuning_time > self.tuning_budget:
                    raise TimeoutError("Tuning budget exceeded during M exploration.")
                
                m_mid1 = m_left + (m_right - m_left) // 3
                m_mid2 = m_right - (m_right - m_left) // 3

                # Process mid1 if not already done
                if m_mid1 not in processed_m:
                    perf_mid1 = self._find_best_efc_for_m(m_mid1)
                    self.m_to_perf.append((m_mid1, perf_mid1))
                    processed_m.add(m_mid1)
                
                # Process mid2 if not already done
                if m_mid2 not in processed_m:
                    perf_mid2 = self._find_best_efc_for_m(m_mid2)
                    self.m_to_perf.append((m_mid2, perf_mid2))
                    processed_m.add(m_mid2)
                
                # Find the performances from the list as they might have been calculated in a previous step
                perf_mid1 = next((p for m, p in self.m_to_perf if m == m_mid1), 0.0)
                perf_mid2 = next((p for m, p in self.m_to_perf if m == m_mid2), 0.0)

                print(f"\n[M Exploration] Range [{m_left}, {m_right}]: M1({m_mid1}) -> {perf_mid1:.4f}, M2({m_mid2}) -> {perf_mid2:.4f}")

                # Heuristic to shrink the search space for M
                if perf_mid1 == perf_mid2:
                    if self.recall_min is not None:
                        m_left = m_mid1
                    else:
                        m_right = m_mid2
                else:
                    penalized_perf1 = perf_mid1 * QPS_PERF_PENALTY if self.recall_min is not None else perf_mid1
                    if penalized_perf1 <= perf_mid2:
                        m_left = m_mid1
                    else:
                        m_right = m_mid2
            
            # Exhaustive search in the final small range of M
            for m_val in range(m_left, m_right + 1):
                if m_val not in processed_m:
                    perf = self._find_best_efc_for_m(m_val)
                    self.m_to_perf.append((m_val, perf))
                    processed_m.add(m_val)

        except TimeoutError as e:
            print(f"Timeout: {e}")

    def _exploitation_phase(self):
        """
        Focuses on the most promising 'M' values found during the exploration phase
        and performs a more detailed search for efC.
        """
        if not self.m_to_perf:
            print("Warning: No M configurations found to exploit.")
            return

        # Sort M by performance in descending order
        sorted_m_configs = sorted(self.m_to_perf, key=lambda x: x[1], reverse=True)
        
        try:
            for m_val, _ in sorted_m_configs:
                if self.ground_truth.tuning_time > self.tuning_budget:
                    raise TimeoutError("Tuning budget exceeded during exploitation.")
                print(f"\n[Exploitation] Refining M={m_val}...")
                self._find_best_efc_for_m(m_val, is_exploitation=True)
        except TimeoutError as e:
            print(f"Timeout: {e}")


    def _find_best_efc_for_m(self, m: int, is_exploitation: bool = False) -> float:
        """
        Finds the best efC for a given M by performing a search over the efC space.
        """
        efc_left, efc_right = self.efC_getter.get(m)
        
        # if m not in self.efS_getters:
        #     self.efS_getters[m] = EfSGetter()
        # efs_getter = self.efS_getters[m]
        
        efc_iter_limit = (math.ceil(math.log(max(EFC_MAX - EFC_MIN, 1), TERNARY_SEARCH_BASE)) // TERNARY_SEARCH_FACTOR 
                          if not is_exploitation else EFC_MAX)
        
        max_perf_of_m = 0.0
        efc_count = 0
        
        # Ternary search for efC
        while (efc_right - efc_left) > 3 and efc_count < efc_iter_limit:
            if self.ground_truth.tuning_time > self.tuning_budget:
                break
            
            efc_count += 1
            efc_mid1 = efc_left + (efc_right - efc_left) // 3
            efc_mid2 = efc_right - (efc_right - efc_left) // 3
            perf_mid1 = self._evaluate_hp(m, efc_mid1)
            perf_mid2 = self._evaluate_hp(m, efc_mid2)
            
            # Update search range based on performance
            if perf_mid1 == perf_mid2 and self.qps_min is not None:
                efc_right = efc_mid2
                efc_count -= 1
            elif perf_mid1 <= perf_mid2:
                efc_left = efc_mid1
            else:
                efc_right = efc_mid2
            if perf_mid1 != perf_mid2: 
                self.efC_getter.put(m, efc_left, efc_right)
            
            max_perf_of_m = max(max_perf_of_m, perf_mid1, perf_mid2)
            print(f"\t[efC Search for M={m}] Range [{efc_left}, {efc_right}]: efC1({efc_mid1}) -> {perf_mid1:.4f}, efC2({efc_mid2}) -> {perf_mid2:.4f}")

        # Exhaustive search in the final small range of efC
        for efc in range(efc_left, efc_right + 1):
            if efc_count >= efc_iter_limit or self.ground_truth.tuning_time > self.tuning_budget:
                break
            
            if (m, efc) in self.efS_getter:
                continue
                
            efc_count += 1
            perf = self._evaluate_hp(m, efc)
            max_perf_of_m = max(max_perf_of_m, perf)

        print(f"\t=> Max performance for M={m}: {max_perf_of_m:.4f}")
        return max_perf_of_m

    def _evaluate_hp(self, m: int, efc: int) -> float:
        """
        Evaluates a single hyperparameter configuration (M, efC, efS), stores the result,
        and returns its performance.
        """
        if self.ground_truth.tuning_time > self.tuning_budget:
            return 0.0

        efs_min, efs_max = self.efS_getter.get(m, efc)
        efs = self.ground_truth.get_efS(m, efc, self.recall_min, self.qps_min, efS_min=efs_min, efS_max=efs_max)
        
        self.efS_getter.put(m, efc, efs)
        
        hp = (m, efc, efs)
        if hp in self.searched_hp:
            # Find existing result if already searched
            existing_result = next(res for h, res in self.results if h == hp)
            return self._get_perf(existing_result[1:])
        
        # Get performance from ground truth
        perf_tuple = self.ground_truth.get(m, efc, efs)
        
        # Store results
        self.searched_hp.add(hp)
        self.results.append((hp, (self.ground_truth.tuning_time, *perf_tuple)))
        
        return self._get_perf(perf_tuple)

def run(impl=IMPL, dataset=DATASET, recall_min=None, qps_min=None, tuning_budget=TUNING_BUDGET, sampling_count=None, env=(TUNING_BUDGET, SEED), stats=False):
    random.seed(SEED)
    ground_truth = GroundTruth(impl=impl, dataset=dataset, sampling_count=sampling_count)
    tuner = HyperparameterTuner(ground_truth, recall_min, qps_min, tuning_budget)
    results = tuner.run_tuning()
    if stats:
        return results, tuner.stats, tuner.efC_getter.stats()
    return results

def run_recall_min_experiments():    
    for RECALL_MIN in [0.99]:
    # for RECALL_MIN in [0.90, 0.925, 0.95, 0.975, 0.99]:
        for IMPL in ["faiss", "hnswlib"]:
        # for IMPL in ["milvus"]:
            # for DATASET in ["nytimes-256-angular", "sift-128-euclidean", "glove-100-angular", 
            #                 "dbpediaentity-768-angular", "msmarco-384-angular", "youtube-1024-angular"]:
            for DATASET in ["nytimes-256-angular", "sift-128-euclidean", "glove-100-angular", "youtube-1024-angular"]:
            # for DATASET in ["glove-100-angular"]:
                print(f"Running for {IMPL} on {DATASET} with RECALL_MIN={RECALL_MIN}")
                results, stat, efS_stat = run(IMPL, DATASET, recall_min=RECALL_MIN, qps_min=None, tuning_budget=TUNING_BUDGET, stats=True)
                opt, _ = print_optimal_hyperparameters(results, recall_min=RECALL_MIN)
                print(opt)
                postprocess_results(
                    results, solution="test_solution", impl=IMPL, dataset=DATASET, 
                    recall_min=RECALL_MIN, tuning_budget=TUNING_BUDGET, lite=True)

def run_qps_min_experiments():
    for QPS_MIN in [29225]:
        for IMPL in ["faiss"]:
            for DATASET in ["nytimes-256-angular",]:
            # for DATASET in ["dbpediaentity-768-angular"]:
                print(f"Running for {IMPL} on {DATASET} with QPS_MIN={QPS_MIN}")
                results = run(IMPL, DATASET, recall_min=None, qps_min=QPS_MIN, tuning_budget=TUNING_BUDGET)
                opt, _ = print_optimal_hyperparameters(results, recall_min=None, qps_min=QPS_MIN)
                print(opt)
                postprocess_results(
                    results, solution="test_solution", impl=IMPL, dataset=DATASET, 
                    qps_min=QPS_MIN, tuning_budget=TUNING_BUDGET, lite=True)

# The rest of your main script (run_recall_min_experiments, run_qps_min_experiments, etc.) remains the same.
if __name__ == "__main__":
    # run_recall_min_experiments()
    run_qps_min_experiments()