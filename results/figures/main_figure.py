import os
import sys
import csv
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import gridspec
from matplotlib.ticker import ScalarFormatter
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

from src.constants import TUNING_BUDGET
from src.utils import (
    filename_builder,
    get_optimal_hyperparameter,
    load_search_results,
    _feasible_and_objective_factory,
)
from data.ground_truths.get_qps_dataset import get_qps_metrics_dataset


# -----------------------------
# Global configs
# -----------------------------
CURRENT_DIR = "results/figures"
SEED = "42"

SOL_STYLES = {
    "our_solution": {"c": "#d62728", "marker": "o", "ls": "-", "lw": 2.0, "zorder": 10, "label": "CHAT"},
    "vd_tuner": {"c": "#9467bd", "marker": "s", "ls": "--", "lw": 1.2, "zorder": 5, "label": "VDTuner"},
    "eci": {"c": "#1f77b4", "marker": "X", "ls": "-.", "lw": 1.2, "zorder": 6, "label": "ECI (GP)"},
    "optuna": {"c": "#8c564b", "marker": "^", "ls": "-.", "lw": 1.2, "zorder": 4, "label": "Optuna"},
    "nsga": {"c": "#e377c2", "marker": "D", "ls": ":", "lw": 1.2, "zorder": 3, "label": "NSGA-II"},
    "random_search": {"c": "#3a7d44", "marker": "v", "ls": "--", "lw": 1.2, "zorder": 2, "label": "Random"},
    "grid_search": {"c": "#bcbd22", "marker": "P", "ls": ":", "lw": 1.2, "zorder": 1, "label": "Grid"},
}
SOL_ORDER = ["our_solution", "vd_tuner", "eci", "optuna", "nsga", "random_search", "grid_search"]

# Failure jitter as a fraction of tuning budget (stable across different budgets)
FAIL_JITTER_FRAC = {
    "our_solution": -0.06,
    "vd_tuner": -0.05,
    "optuna": -0.08,
    "eci": -0.02,
    "nsga": -0.01,
    "random_search": 0,
    "grid_search": 0.01,
}

ALPHAS_BASE = [0.7, 0.8, 0.9, 0.95]
ALPHAS_CALC = [0.001, 0.25, 0.5] + ALPHAS_BASE
TAIL_START = 0.95

# Broken-axis (piecewise linear) mapping
BOUNDARY_REAL = 0.7
BOUNDARY_VISUAL = 0.30


def squash_y(y: float) -> float:
    """Map real ratio y in [0,1] to visual y in [0,1] with piecewise linear squash."""
    y = float(y)
    if y <= BOUNDARY_REAL:
        return y * (BOUNDARY_VISUAL / BOUNDARY_REAL)
    slope = (1.0 - BOUNDARY_VISUAL) / (1.0 - BOUNDARY_REAL)
    return BOUNDARY_VISUAL + (y - BOUNDARY_REAL) * slope


def draw_wavy_y_break(
    ax,
    y_data: float,
    x_center: float = -0.02,
    width: float = 0.06,
    amp: float = 0.012,
    gap: float = 0.028,
    cycles: float = 1.0,
    lw: float = 0.5,
    color: str = "gray",
    alpha: float = 0.5,
):
    """Draw a 'wavy break' symbol (two sine-like strokes) near the left y-axis at y_data."""
    trans = blended_transform_factory(ax.transAxes, ax.transData)

    xs = np.linspace(x_center - width / 2.0, x_center + width / 2.0, 200)
    phase = 2.0 * np.pi * cycles * (xs - xs.min()) / (xs.max() - xs.min())

    y_top = y_data + gap / 2.0 + amp * np.sin(phase)
    y_bot = y_data - gap / 2.0 + amp * np.sin(phase)

    ax.plot(
        xs,
        y_top,
        transform=trans,
        color=color,
        lw=lw,
        alpha=alpha,
        clip_on=False,
        solid_capstyle="round",
        zorder=50,
    )
    ax.plot(
        xs,
        y_bot,
        transform=trans,
        color=color,
        lw=lw,
        alpha=alpha,
        clip_on=False,
        solid_capstyle="round",
        zorder=50,
    )


def short_ds(name: str) -> str:
    return name.split("-")[0]


def get_results(impl, dataset, solutions, recall_min=None, qps_min=None, sampling_count=None, tuning_time=TUNING_BUDGET):
    assert (recall_min is not None) != (qps_min is not None)

    results_combi = {}
    oracle_best_metric = 0.0

    for solution in solutions:
        filename = filename_builder(solution, impl, dataset, recall_min, qps_min)
        results = load_search_results(solution, filename, seed=SEED, sampling_count=sampling_count)

        if solution == "brute_force":
            optimal_hp = get_optimal_hyperparameter(results, recall_min=recall_min, qps_min=qps_min)
            hp = optimal_hp[0]
            _tt, recall, qps, total_time, build_time, index_size = optimal_hp[1]
            perf = (0.0, recall, qps, total_time, build_time, index_size)
            results = [(hp, perf)]

            metric = qps if recall_min is not None else recall
            oracle_best_metric = max(oracle_best_metric, float(metric))
        else:
            results = [r for r in results if float(r[1][0]) <= tuning_time]

        results_combi[solution] = results

    return {
        "impl": impl,
        "dataset": dataset,
        "recall_min": recall_min,
        "qps_min": qps_min,
        "results": results_combi,
        "oracle_best_metric": oracle_best_metric,
    }


def build_attainment_points(
    data,
    is_feasible,
    objective,
    oracle_best,
    budget,
    alphas_calc=ALPHAS_CALC,
    tail_start=TAIL_START,
):
    """Return attainment points:
    - Keep alpha markers (0.001, 0.25, 0.5, 0.7, 0.8, 0.9, 0.95) but avoid stacked points at same time.
    - For the first attainment, draw a vertical jump at time t1: (t1,0)->(t1,y1).
    - After that, connect points diagonally/straight as usual.
    - If no attainment within budget, return a single failure point at (budget, 0%).
    """
    if (not data) or oracle_best <= 1e-12:
        return [], []

    valid = []
    for _, perf in data:
        if is_feasible(perf):
            t = float(perf[0])
            if t <= budget:
                val = float(objective(perf))
                valid.append((t, val))
    valid.sort(key=lambda x: x[0])

    if not valid:
        return [float(budget)], [0.0]

    # Build monotone best-so-far envelope
    times, bests = [], []
    cur = -np.inf
    for t, v in valid:
        if v > cur:
            cur = v
            times.append(float(t))
            bests.append(float(cur))

    # Alpha-threshold points: keep only maximum y per hit time
    hit_to_y = {}
    for alpha in alphas_calc:
        target = alpha * oracle_best
        hit = None
        for t, b in zip(times, bests):
            if b >= target:
                hit = t
                break
        if hit is not None:
            prev = hit_to_y.get(hit, -1.0)
            if alpha > prev:
                hit_to_y[hit] = alpha

    # Tail points after tail_start: also merged by time (keep max y)
    tail_target = tail_start * oracle_best
    start_idx = None
    for i, b in enumerate(bests):
        if b >= tail_target:
            start_idx = i
            break

    if start_idx is not None:
        for t, b in zip(times[start_idx:], bests[start_idx:]):
            ratio = min(float(b / oracle_best), 1.0)
            prev = hit_to_y.get(t, -1.0)
            if ratio > prev:
                hit_to_y[t] = ratio

    if not hit_to_y:
        return [float(budget)], [0.0]

    plot_points = sorted(hit_to_y.items(), key=lambda x: x[0])
    t1, y1 = float(plot_points[0][0]), float(plot_points[0][1])

    xs = [t1, t1]
    ys = [0.0, y1]

    for t, y in plot_points[1:]:
        xs.append(float(t))
        ys.append(float(y))

    return xs, ys


def plot_attainment_on_ax(
    ax,
    results_dict,
    recall_min=None,
    qps_min=None,
    tuning_budget=TUNING_BUDGET,
    max_perf=None,
    show_xlabel=False,
    show_ylabel=False,
):
    is_feasible, objective, _ = _feasible_and_objective_factory(recall_min, qps_min, tuning_budget, max_perf)

    if not max_perf or max_perf <= 0:
        ax.text(0.5, 0.5, "No Oracle", ha="center", va="center", fontsize=8)
        return

    for sol in SOL_ORDER:
        data = results_dict.get(sol, [])
        style = SOL_STYLES[sol]

        xs, ys_real = build_attainment_points(
            data=data,
            is_feasible=is_feasible,
            objective=objective,
            oracle_best=max_perf,
            budget=tuning_budget,
        )

        if not xs:
            continue

        ys_vis = [squash_y(y) for y in ys_real]

        # Failure case: only one point at 0% within budget -> plot a single jittered marker
        if len(xs) == 1 and abs(ys_real[0]) < 1e-12:
            jitter = FAIL_JITTER_FRAC.get(sol, 0.0) * float(tuning_budget)
            xj = float(tuning_budget) + float(jitter)
            xj = max(0.0, xj)

            ax.plot(
                [xj],
                [ys_vis[0]],
                marker=style["marker"],
                color=style["c"],
                linestyle="None",
                markersize=6,
                zorder=style["zorder"],
                label=style["label"],
                alpha=0.9,
            )
        else:
            ax.plot(
                xs,
                ys_vis,
                marker=style["marker"],
                color=style["c"],
                linestyle=style["ls"],
                linewidth=style["lw"],
                markersize=5,
                zorder=style["zorder"],
                label=style["label"],
                alpha=0.9,
            )

    # X axis
    ax.set_xlim(-500, tuning_budget * 1.05)
    ax.set_xticks([0, tuning_budget / 4, tuning_budget / 2, tuning_budget * 3 / 4, tuning_budget])
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(formatter)
    if show_xlabel:
        ax.tick_params(axis="x", labelsize=12)
    else:
        ax.set_xticklabels([])

    # Y axis (custom ticks in visual space, labels in real %)
    display_yticks_real = [0.0] + ALPHAS_BASE + [1.0]
    display_yticks_vis = [squash_y(y) for y in display_yticks_real]
    ax.set_ylim(0.0, 1.05)
    ax.set_yticks(display_yticks_vis)
    if show_ylabel:
        ax.set_yticklabels([f"{int(y * 100)}%" for y in display_yticks_real], fontsize=12)
    else:
        ax.set_yticklabels([])

    # Grid
    ax.grid(True, which="major", linestyle=":", linewidth=0.7, color="gray", alpha=0.8)

    # Reference lines
    break_y_vis = squash_y(BOUNDARY_REAL)
    ax.axhline(break_y_vis, linestyle=":", linewidth=0.7, color="gray", alpha=0.8, zorder=0)

    top_y_vis = squash_y(1.0)
    ax.axhline(top_y_vis, linestyle="--", linewidth=1.0, color="blue", alpha=0.9, zorder=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axvline(tuning_budget, linestyle=":", color="gray", linewidth=0.7, alpha=0.8, zorder=0)

    # Wavy compression mark
    y_mid_real = BOUNDARY_REAL / 2.0
    y_mid_vis = squash_y(y_mid_real)
    if show_ylabel:
        draw_wavy_y_break(
            ax,
            y_data=y_mid_vis,
            x_center=-0.005,
            width=0.05,
            amp=0.015,
            gap=0.028,
            cycles=1.0,
            lw=0.9,
            color="black",
            alpha=0.8,
        )


def setup_fonts():
    """Load Libertine fonts if present."""
    font_path_r = f"{CURRENT_DIR}/LinLibertine_R.ttf"
    if os.path.exists(font_path_r):
        fm.fontManager.addfont(font_path_r)

    font_path_b = f"{CURRENT_DIR}/LinLibertine_B.ttf"
    if os.path.exists(font_path_b):
        fm.fontManager.addfont(font_path_b)

    if os.path.exists(font_path_r):
        font_prop = fm.FontProperties(fname=font_path_r)
        plt.rcParams["font.family"] = font_prop.get_name()
    plt.rcParams["axes.unicode_minus"] = False


def build_global_legend(fig):
    """Global legend with line+marker handles (method identity)."""
    handles = []
    labels = []

    for sol in SOL_ORDER:
        s = SOL_STYLES[sol]

        # CHAT 먼저
        if sol == "our_solution":
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=s["c"],
                    marker=s["marker"],
                    linestyle=s["ls"],
                    linewidth=s["lw"],
                )
            )
            labels.append("CHAT (Our solution)")

            # --- Oracle reference 바로 뒤에 삽입 ---
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color="blue",
                    linestyle="--",
                    linewidth=1.5,
                )
            )
            labels.append("Oracle")

        else:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=s["c"],
                    marker=s["marker"],
                    linestyle=s["ls"],
                    linewidth=s["lw"],
                )
            )
            labels.append(s["label"])

    leg = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.485, 0.975),
        ncol=len(labels),
        fontsize=19,
        frameon=False,
    )

    for t in leg.get_texts():
        if t.get_text().startswith("CHAT"):
            t.set_fontweight("bold")



def main():
    setup_fonts()

    SOLUTIONS = ["brute_force", "our_solution", "grid_search", "random_search", "vd_tuner", "eci", "optuna", "nsga"]
    IMPLS = ["hnswlib", "faiss"]
    DATASETS = ["nytimes-256-angular", "glove-100-angular", "sift-128-euclidean", "youtube-1024-angular", "deep1M-256-angular"]
    SAMPLING_COUNT = 10
    RECALL_MIN = 0.95
    QPS_MIN_KEY = "q75"

    # Load data: 2 impl * 5 datasets * (QPS/Recall rows) = 20 subplots
    results_list = []
    for impl in IMPLS:
        for dataset in DATASETS:
            results_list.append(get_results(impl, dataset, SOLUTIONS, recall_min=RECALL_MIN, qps_min=None, sampling_count=SAMPLING_COUNT))
        for dataset in DATASETS:
            qps_min = get_qps_metrics_dataset(impl, dataset, ret_dict=True)[QPS_MIN_KEY]
            results_list.append(get_results(impl, dataset, SOLUTIONS, recall_min=None, qps_min=qps_min, sampling_count=SAMPLING_COUNT))

    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(
        nrows=4,
        ncols=5,
        figure=fig,
        left=0.06,
        right=0.98,
        top=0.86,
        bottom=0.10,
        wspace=0.16,
        hspace=0.18,
    )

    axes = [[fig.add_subplot(gs[r, c]) for c in range(5)] for r in range(4)]

    for idx, rdict in enumerate(results_list):
        r = idx // 5
        c = idx % 5
        ax = axes[r][c]

        plot_attainment_on_ax(
            ax,
            results_dict=rdict["results"],
            recall_min=rdict["recall_min"],
            qps_min=rdict["qps_min"],
            tuning_budget=TUNING_BUDGET,
            max_perf=rdict["oracle_best_metric"],
            show_xlabel=True,
            show_ylabel=True,
        )

        if r == 0:
            ax.set_title(short_ds(rdict["dataset"]), fontsize=19, pad=4, weight="bold")
        if r == 3:
            ax.set_xlabel("Time (seconds)", fontsize=18, weight="bold")

    build_global_legend(fig)

    # Row/column labels
    fig.text(0.015, 0.71, "Hnswlib", va="center", ha="center", rotation="vertical", fontsize=24, weight="bold")
    fig.text(0.015, 0.30, "Faiss", va="center", ha="center", rotation="vertical", fontsize=24, weight="bold")

    label_opts = {"va": "center", "ha": "center", "rotation": "vertical", "fontsize": 18, "weight": "bold"}
    fig.text(0.035, 0.78, "QPS", **label_opts)
    fig.text(0.035, 0.59, "Recall", **label_opts)
    fig.text(0.035, 0.38, "QPS", **label_opts)
    fig.text(0.035, 0.185, "Recall", **label_opts)

    output_filename = "main_figure_attainment_broken_axis.pdf"
    fig.savefig(output_filename, bbox_inches="tight")
    print(f"Saved broken axis plot to {output_filename}")


if __name__ == "__main__":
    main()
