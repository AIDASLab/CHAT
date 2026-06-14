import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set

import numpy as np
import torch

from botorch.acquisition.analytic import ConstrainedExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

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

STEP_M = 1
STEP_EFC = 8
STEP_EFS = 16

# BO knobs
N_WARMUP = 16
MAX_ITERS = 512
CANDIDATE_POOL_SIZE = 6000

DEVICE = torch.device("cpu")
DTYPE = torch.double


@dataclass(frozen=True)
class Config:
    M: int
    efC: int
    efS: int


def _snapped_efc_min(M: int) -> int:
    """Return smallest efC on grid such that efC >= max(M, EFC_MIN)."""
    raw_min = max(M, EFC_MIN)
    offset = (raw_min - EFC_MIN + STEP_EFC - 1) // STEP_EFC
    return EFC_MIN + offset * STEP_EFC


def _build_candidate_grid() -> List[Config]:
    """Build discrete grid with efC >= M constraint and step sizes."""
    out: List[Config] = []
    for M in range(M_MIN, M_MAX + 1, STEP_M):
        efc_min_bound = _snapped_efc_min(M)
        if efc_min_bound > EFC_MAX:
            continue
        for efC in range(efc_min_bound, EFC_MAX + 1, STEP_EFC):
            for efS in range(EFS_MIN, EFS_MAX + 1, STEP_EFS):
                out.append(Config(M=M, efC=efC, efS=efS))
    return out


def _to_unit_x(cfg: Config) -> torch.Tensor:
    """Map config to [0,1]^3 for GP input."""
    xM = (cfg.M - M_MIN) / max(1, (M_MAX - M_MIN))
    xC = (cfg.efC - EFC_MIN) / max(1, (EFC_MAX - EFC_MIN))
    xS = (cfg.efS - EFS_MIN) / max(1, (EFS_MAX - EFS_MIN))
    return torch.tensor([xM, xC, xS], device=DEVICE, dtype=DTYPE)


def _fit_multi_output_gp(train_X: torch.Tensor, train_Y: torch.Tensor) -> SingleTaskGP:
    """
    Fit a multi-output SingleTaskGP.
    train_Y must be shape [n, 2]:
      - output 0: objective (to maximize)
      - output 1: constraint value g(x) (feasible if <= 0)
    """
    model = SingleTaskGP(
        train_X,
        train_Y,
        input_transform=Normalize(d=train_X.shape[-1]),
        outcome_transform=Standardize(m=train_Y.shape[-1]),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def run(
    impl: str = IMPL,
    dataset: str = DATASET,
    recall_min: Optional[float] = None,
    qps_min: Optional[float] = None,
    tuning_budget: float = TUNING_BUDGET,
    sampling_count: Optional[int] = MAX_SAMPLING_COUNT,
    env=(TUNING_BUDGET, SEED),
):
    assert (recall_min is None) != (qps_min is None), "Set exactly one of recall_min or qps_min."

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    gd = GroundTruth(impl=impl, dataset=dataset, sampling_count=sampling_count)

    results: List[
        Tuple[
            Tuple[int, int, int],
            Tuple[float, float, float, float, float, float],
        ]
    ] = []

    all_candidates = _build_candidate_grid()
    if not all_candidates:
        return results

    evaluated: Set[Config] = set()

    X_list: List[torch.Tensor] = []
    Y_list: List[torch.Tensor] = []  # each is shape [2]: [objective, constraint_value]

    def evaluate(cfg: Config) -> Dict[str, float]:
        recall, qps, total_time, build_time, index_size = gd.get(M=cfg.M, efC=cfg.efC, efS=cfg.efS)
        return {
            "recall": float(recall),
            "qps": float(qps),
            "total_time": float(total_time),
            "build_time": float(build_time),
            "index_size": float(index_size),
        }

    def append_result(cfg: Config, metrics: Dict[str, float]) -> None:
        results.append(
            (
                (cfg.M, cfg.efC, cfg.efS),
                (
                    float(gd.tuning_time),
                    metrics["recall"],
                    metrics["qps"],
                    metrics["total_time"],
                    metrics["build_time"],
                    metrics["index_size"],
                ),
            )
        )

    def obj_and_con(metrics: Dict[str, float]) -> Tuple[float, float]:
        """
        Objective is always "maximize".
        Constraint is g(x) <= 0, with g = threshold - achieved_metric.
        """
        if recall_min is not None:
            obj = metrics["qps"]
            con = recall_min - metrics["recall"]
        else:
            assert qps_min is not None
            obj = metrics["recall"]
            con = qps_min - metrics["qps"]
        return float(obj), float(con)

    # -----------------------------
    # Warm-up random points
    # -----------------------------
    warmup = random.sample(all_candidates, k=min(N_WARMUP, len(all_candidates)))
    for cfg in warmup:
        if gd.tuning_time > tuning_budget:
            break
        evaluated.add(cfg)
        metrics = evaluate(cfg)
        append_result(cfg, metrics)

        obj, con = obj_and_con(metrics)
        X_list.append(_to_unit_x(cfg))
        Y_list.append(torch.tensor([obj, con], device=DEVICE, dtype=DTYPE))

    if not X_list:
        return results

    # -----------------------------
    # BO loop
    # -----------------------------
    for _ in range(MAX_ITERS):
        if gd.tuning_time > tuning_budget:
            break

        train_X = torch.stack(X_list, dim=0)
        train_Y = torch.stack(Y_list, dim=0)  # [n,2]

        try:
            model = _fit_multi_output_gp(train_X, train_Y)
        except Exception:
            remaining = [c for c in all_candidates if c not in evaluated]
            if not remaining:
                break
            cfg = random.choice(remaining)
            evaluated.add(cfg)
            metrics = evaluate(cfg)
            append_result(cfg, metrics)
            obj, con = obj_and_con(metrics)
            X_list.append(_to_unit_x(cfg))
            Y_list.append(torch.tensor([obj, con], device=DEVICE, dtype=DTYPE))
            continue

        # Determine best_f among feasible observations
        Y_obj = train_Y[:, 0]
        Y_con = train_Y[:, 1]
        feas = (Y_con <= 0.0)
        if feas.any():
            best_f = Y_obj[feas].max().item()
        else:
            best_f = Y_obj.max().item()  # fallback; ECI will still prioritize feasibility via constraint

        # Candidate pool from remaining discrete grid
        remaining = [c for c in all_candidates if c not in evaluated]
        if not remaining:
            break
        pool = remaining if len(remaining) <= CANDIDATE_POOL_SIZE else random.sample(remaining, k=CANDIDATE_POOL_SIZE)
        X_pool = torch.stack([_to_unit_x(c) for c in pool], dim=0)  # [m,3]

        # ECI: objective output 0, constraint output 1 must be <= 0
        # Many BoTorch versions require objective_index explicitly.
        acq = ConstrainedExpectedImprovement(
            model=model,
            best_f=best_f,
            objective_index=0,
            constraints={1: (None, 0.0)},
        )

        with torch.no_grad():
            # analytic acquisition expects X shape [batch, 1, d]
            vals = acq(X_pool.unsqueeze(1)).squeeze(-1)
            idx = int(torch.argmax(vals).item())
            next_cfg = pool[idx]

        evaluated.add(next_cfg)
        metrics = evaluate(next_cfg)
        append_result(next_cfg, metrics)

        obj, con = obj_and_con(metrics)
        X_list.append(_to_unit_x(next_cfg))
        Y_list.append(torch.tensor([obj, con], device=DEVICE, dtype=DTYPE))

    return results

