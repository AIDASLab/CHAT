# nsga/run.py

import random
from typing import List, Tuple, Optional

import optuna
from optuna.samplers import NSGAIISampler

from src.solutions import postprocess_results, print_optimal_hyperparameters
from data.ground_truths import GroundTruth
from src.constants import (
    EFC_MAX,
    EFC_MIN,
    EFS_MAX,
    EFS_MIN,
    M_MAX,
    M_MIN,
    IMPL,
    DATASET,
    SEED,
    TUNING_BUDGET,
    MAX_SAMPLING_COUNT,
)

# =======================
# NSGA-II config (Optuna)
# =======================

POP_SIZE = 32          # population size
STEP_M = 1
STEP_EFC = 8
STEP_EFS = 16


def snapped_efc_min(M: int) -> int:
    """Return the smallest efC on the grid {EFC_MIN + k*STEP_EFC} such that efC >= max(M, EFC_MIN)."""
    raw_min = max(M, EFC_MIN)
    offset = (raw_min - EFC_MIN + STEP_EFC - 1) // STEP_EFC
    efc_min_bound = EFC_MIN + offset * STEP_EFC
    return efc_min_bound


def run(
    impl: str = IMPL,
    dataset: str = DATASET,
    recall_min: Optional[float] = None,
    qps_min: Optional[float] = None,
    tuning_budget: float = TUNING_BUDGET,
    sampling_count: Optional[int] = MAX_SAMPLING_COUNT,
    env=(TUNING_BUDGET, SEED),
):
    """
    Optuna-NSGAII baseline for HNSW hyperparameter tuning.

    Decision variables:
        - M, efC, efS (same discrete grid as grid_search via STEP_*)

    Modes (exactly one must be set):
        - recall_min != None: maximize QPS s.t. Recall >= recall_min
        - qps_min   != None: maximize Recall s.t. QPS >= qps_min

    Objective shaping for NSGA-II:
        - recall_min mode: (violation, -qps) is minimized
        - qps_min   mode: (violation, -recall) is minimized
        where violation = max(0, threshold - metric).

    Return:
        List of ((M, efC, efS),
                 (tuning_time, recall, qps, total_time, build_time, index_size))
    """
    assert (recall_min is None) != (qps_min is None), \
        "Only one of recall_min or qps_min should be set."

    random.seed(SEED)

    # GroundTruth instance (tuning_time accumulates inside)
    gd = GroundTruth(impl=impl, dataset=dataset, sampling_count=sampling_count)

    # Results buffer compatible with other baselines
    results: List[
        Tuple[
            Tuple[int, int, int],
            Tuple[float, float, float, float, float, float],
        ]
    ] = []

    # Multi-objective directions:
    # Both objectives are minimized.
    # recall_min mode : (violation, -qps)
    # qps_min   mode : (violation, -recall)
    study = optuna.create_study(
        directions=["minimize", "minimize"],
        sampler=NSGAIISampler(
            population_size=POP_SIZE,
            seed=SEED,
        ),
    )

    def objective(trial: optuna.Trial):
        # Sample M on the grid
        M = trial.suggest_int("M", M_MIN, M_MAX, step=STEP_M)

        # Compute efC lower bound on the same efC grid and respecting M <= efC
        efc_min_bound = snapped_efc_min(M)
        if efc_min_bound > EFC_MAX:
            # No feasible efC on the grid; give a clearly bad objective
            return 1.0, 0.0

        efC = trial.suggest_int(
            "efConstruction",
            efc_min_bound,
            EFC_MAX,
            step=STEP_EFC,
        )
        efS = trial.suggest_int("efSearch", EFS_MIN, EFS_MAX, step=STEP_EFS)

        # Evaluate HNSW performance
        recall, qps, total_time, build_time, index_size = gd.get(
            M=M,
            efC=efC,
            efS=efS,
        )

        # Global tuning budget check
        if gd.tuning_time > tuning_budget:
            trial.study.stop()
            # Return some dummy but finite values
            return 1.0, 0.0

        # Record result for later analysis
        results.append(
            (
                (M, efC, efS),
                (gd.tuning_time, recall, qps, total_time, build_time, index_size),
            )
        )

        # Constraint-aware multi-objective shaping
        if recall_min is not None:
            # Constraint: recall >= recall_min
            violation = max(0.0, recall_min - recall)
            # Second objective: maximize QPS -> minimize -QPS
            return violation, -qps

        # qps_min mode
        assert qps_min is not None
        violation = max(0.0, qps_min - qps)
        # Second objective: maximize Recall -> minimize -Recall
        return violation, -recall

    # Run until budget-based stop inside objective
    study.optimize(
        objective,
        n_trials=None,
        timeout=None,
        catch=(Exception,),  # optional: catch unexpected errors per trial
    )

    return results

