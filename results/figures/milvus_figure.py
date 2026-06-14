import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import gridspec
from matplotlib.ticker import ScalarFormatter
from matplotlib.lines import Line2D
import numpy as np
import os

from src.constants import TUNING_BUDGET
from src.utils import (
    filename_builder,
    get_optimal_hyperparameter,
    load_search_results,
    _feasible_and_objective_factory,
)
from data.ground_truths.get_qps_dataset import get_qps_metrics_dataset

current_dir = "results/figures"
SEED = "42"

# --- Styles per solution ---
SOL_STYLES = {
    "our_solution":  {"c": "#d62728", "marker": "o", "ls": "-",  "lw": 3, "zorder": 10, "label": "CHAT"},
    "vd_tuner":      {"c": "#9467bd", "marker": "s", "ls": "--", "lw": 2.5, "zorder": 6,  "label": "VDTuner"},
    "eci":           {"c": "#1f77b4", "marker": "X", "ls": "-.", "lw": 2.5, "zorder": 5,  "label": "ECI"},
    "optuna":        {"c": "#8c564b", "marker": "^", "ls": "-.", "lw": 2.5, "zorder": 4,  "label": "Optuna"},
    "nsga":          {"c": "#e377c2", "marker": "D", "ls": ":",  "lw": 2.5, "zorder": 3,  "label": "NSGA"},
    "random_search": {"c": "#3a7d44", "marker": "v", "ls": "--", "lw": 2.5, "zorder": 2,  "label": "Rand"},
    "grid_search":   {"c": "#bcbd22", "marker": "P", "ls": ":",  "lw": 2.5, "zorder": 1,  "label": "Grid"},
}

# Plot order in the figure and legend
SOL_ORDER = ["our_solution", "vd_tuner", "eci", "optuna", "nsga", "random_search", "grid_search"]

# Base target levels
ALPHAS_BASE = [0.7, 0.8, 0.9, 0.925, 0.95]
TAIL_START = 0.95


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
            if metric > oracle_best_metric:
                oracle_best_metric = metric
        else:
            results = [r for r in results if r[1][0] <= tuning_time]

        results_combi[solution] = results

    return {
        "impl": impl,
        "dataset": dataset,
        "recall_min": recall_min,
        "qps_min": qps_min,
        "results": results_combi,
        "oracle_best_metric": oracle_best_metric,
    }


def build_attainment_points_linear(
    data,
    is_feasible,
    objective,
    oracle_best,
    budget,
    alphas_base=ALPHAS_BASE,
    tail_start=TAIL_START,
):
    if (not data) or oracle_best <= 1e-12:
        return [], []

    valid_points = []
    for _, perf in data:
        if is_feasible(perf):
            t = float(perf[0])
            if t <= budget:
                val = float(objective(perf))
                valid_points.append((t, val))
    valid_points.sort(key=lambda x: x[0])

    if not valid_points:
        return [], []

    times = []
    best_perfs = []
    current_max = -np.inf
    for t, val in valid_points:
        if val > current_max:
            current_max = val
            times.append(t)
            best_perfs.append(current_max)

    plot_xs = []
    plot_ys = []

    for alpha in alphas_base:
        target = alpha * oracle_best
        found_t = None
        for t, b in zip(times, best_perfs):
            if b >= target:
                found_t = t
                break
        if found_t is not None:
            plot_xs.append(found_t)
            plot_ys.append(alpha)

    tail_target = tail_start * oracle_best
    start_idx = None
    for i, b in enumerate(best_perfs):
        if b >= tail_target:
            start_idx = i
            break

    if start_idx is not None:
        last_base_t = plot_xs[-1] if plot_xs else None
        last_base_y = plot_ys[-1] if plot_ys else None

        for t, b in zip(times[start_idx:], best_perfs[start_idx:]):
            ratio = min(float(b / oracle_best), 1.0)
            if last_base_t is not None and last_base_y is not None:
                if abs(t - last_base_t) < 1e-12 and abs(ratio - last_base_y) < 1e-12:
                    continue
            plot_xs.append(float(t))
            plot_ys.append(ratio)

    return plot_xs, plot_ys


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

    if max_perf is None or max_perf <= 0:
        ax.text(0.5, 0.5, "No Oracle", ha="center", va="center", fontsize=8)
        return

    for sol in SOL_ORDER:
        data = results_dict.get(sol, [])

        if sol not in SOL_STYLES:
            continue

        style = SOL_STYLES[sol]
        plot_xs, plot_ys = build_attainment_points_linear(
            data=data,
            is_feasible=is_feasible,
            objective=objective,
            oracle_best=max_perf,
            budget=tuning_budget,
            alphas_base=ALPHAS_BASE,
            tail_start=TAIL_START,
        )

        if plot_xs:
            ax.plot(
                plot_xs,
                plot_ys,
                marker=style["marker"],
                color=style["c"],
                linestyle=style["ls"],
                linewidth=style["lw"],
                markersize=10,
                zorder=style["zorder"],
                label=style["label"],
                alpha=0.9,
            )

    ax.set_xlim(0.0, tuning_budget * 1.05)
    major_ticks = [0, tuning_budget / 4, tuning_budget / 2, tuning_budget * 3 / 4, tuning_budget]
    ax.set_xticks(major_ticks)

    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    ax.get_xaxis().set_major_formatter(formatter)

    if show_xlabel:
        ax.tick_params(axis="x", labelsize=24)
        ax.set_xlabel("Time (seconds)", fontsize=36, weight="bold")
    else:
        ax.set_xticklabels([])

    yticks = ALPHAS_BASE + [1.0]
    ax.set_ylim(0.7, 1.02)
    ax.set_yticks(yticks)

    if show_ylabel:
        ax.set_yticklabels([f"{int(a*100)}%" for a in yticks], fontsize=24)
    else:
        ax.set_yticklabels([])

    ax.grid(True, which="major", linestyle=":", linewidth=1.2, color="black", alpha=0.6)
    ax.axhline(1.0, linestyle="--", linewidth=1.5, color="blue", alpha=0.9, zorder=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axvline(tuning_budget, linestyle=":", color="black", linewidth=1.0, alpha=0.5)


def main():
    # -------------------------------------------------------------------------
    # Font setup
    # -------------------------------------------------------------------------
    font_path_r = f"{current_dir}/LinLibertine_R.ttf"
    fm.fontManager.addfont(font_path_r)

    font_path_b = f"{current_dir}/LinLibertine_B.ttf"
    if os.path.exists(font_path_b):
        fm.fontManager.addfont(font_path_b)
        print(f"Bold font loaded: {font_path_b}")
    else:
        print("Warning: Bold font not found. Text may not appear bold.")

    font_prop = fm.FontProperties(fname=font_path_r)
    font_name = font_prop.get_name()
    plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False
    # -------------------------------------------------------------------------

    SOLUTIONS = ["brute_force", "our_solution", "grid_search", "random_search", "vd_tuner", "eci", "optuna", "nsga"]

    TARGET_IMPL = "milvus"
    TARGET_DATASET = "nytimes-256-angular"
    SAMPLING_COUNT = 10
    RECALL_MIN = 0.95
    QPS_MIN_KEY = "q75"

    results_recall = get_results(
        TARGET_IMPL, TARGET_DATASET, SOLUTIONS,
        recall_min=RECALL_MIN, qps_min=None, sampling_count=SAMPLING_COUNT
    )

    qps_target_val = get_qps_metrics_dataset(TARGET_IMPL, TARGET_DATASET, ret_dict=True)[QPS_MIN_KEY]
    results_qps = get_results(
        TARGET_IMPL, TARGET_DATASET, SOLUTIONS,
        recall_min=None, qps_min=qps_target_val, sampling_count=SAMPLING_COUNT
    )

    # ---- Layout ----
    fig = plt.figure(figsize=(15, 6))
    gs = gridspec.GridSpec(
        nrows=1,
        ncols=2,
        figure=fig,
        left=0.08,
        right=0.98,
        top=0.78,
        bottom=0.12,
        wspace=0.20,
    )

    # ---- Plotting ----
    ax_left = fig.add_subplot(gs[0, 0])
    plot_attainment_on_ax(
        ax_left,
        results_dict=results_recall["results"],
        recall_min=RECALL_MIN,
        qps_min=None,
        tuning_budget=TUNING_BUDGET,
        max_perf=results_recall["oracle_best_metric"],
        show_xlabel=True,
        show_ylabel=True,
    )
    ax_left.set_ylabel("QPS", fontsize=36, weight="bold")

    ax_right = fig.add_subplot(gs[0, 1])
    plot_attainment_on_ax(
        ax_right,
        results_dict=results_qps["results"],
        recall_min=None,
        qps_min=qps_target_val,
        tuning_budget=TUNING_BUDGET,
        max_perf=results_qps["oracle_best_metric"],
        show_xlabel=True,
        show_ylabel=True,
    )
    ax_right.set_ylabel("Recall", fontsize=36, weight="bold")

    ax_left.yaxis.set_label_coords(-0.1, 0.5)
    ax_right.yaxis.set_label_coords(-0.1, 0.5)

    # ---- Global Legend (top): CHAT then Oracle (blue dashed line), then baselines ----
    legend_handles = []
    legend_labels = []

    for sol in SOL_ORDER:
        s = SOL_STYLES[sol]

        if sol == "our_solution":
            # CHAT
            legend_handles.append(
                Line2D([0], [0],
                       color=s["c"],
                       marker=s["marker"],
                       linestyle=s["ls"],
                       linewidth=s["lw"],
                       markersize=10)
            )
            legend_labels.append(s["label"])

            # Oracle reference line (no marker)
            legend_handles.append(
                Line2D([0], [0],
                       color="blue",
                       linestyle="--",
                       linewidth=1.5)
            )
            legend_labels.append("Oracle")

        else:
            legend_handles.append(
                Line2D([0], [0],
                       color=s["c"],
                       marker=s["marker"],
                       linestyle=s["ls"],
                       linewidth=s["lw"],
                       markersize=10)
            )
            legend_labels.append(s["label"])

    leg = fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=len(legend_labels),
        fontsize=30,
        frameon=False,
        columnspacing=0.35,
        handletextpad=0.25,
        handlelength=1.2,
    )

    for text in leg.get_texts():
        if text.get_text().startswith("CHAT"):
            text.set_fontweight("bold")

    output_filename = "milvus_attainment_linear.pdf"
    fig.savefig(output_filename, bbox_inches="tight")
    print(f"Saved plot to {output_filename}")


if __name__ == "__main__":
    main()
