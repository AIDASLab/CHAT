import os
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.font_manager as fm
from matplotlib import colors

from src.constants import EFC_MAX, EFC_MIN, M_MAX, M_MIN, RECALL_MIN, SEED, TOLERANCE, TUNING_BUDGET, MAX_SAMPLING_COUNT

font_path = './results/figures/LinLibertine_R.ttf'

if font_path not in [f.fname for f in fm.fontManager.ttflist]:
    fm.fontManager.addfont(font_path)

font_name = fm.FontProperties(fname=font_path).get_name()
plt.rcParams['font.family'] = font_name

plt.rcParams['axes.unicode_minus'] = False

def filename_builder(solution, impl, dataset, recall_min, qps_min, tuning_budget=None):
    if tuning_budget:
        return f"{solution}_{impl}_{dataset}_{recall_min}r_{qps_min}q_{tuning_budget}.csv"
    else:
        return f"{solution}_{impl}_{dataset}_{recall_min}r_{qps_min}q.csv"

def _save_path(output_type, solution, filename, seed, sampling_count):
    path = os.path.join("results", output_type, str(seed), str(sampling_count), solution)
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, filename)

def is_already_saved(solution, filename, seed=SEED, sampling_count=MAX_SAMPLING_COUNT):
    save_path = _save_path("result", solution, filename, seed, sampling_count)
    return os.path.exists(save_path)

def save_search_results(results, solution, filename, seed=SEED, sampling_count=MAX_SAMPLING_COUNT):
    save_path = _save_path("result", solution, filename, seed, sampling_count)
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["M", "efC", "efS", "T_record", "recall", "qps", "total_time", "build_time", "index_size"])
        for (M, efC, efS), (T_record, recall, qps, total_time, build_time, index_size) in results:
            writer.writerow([M, efC, efS, T_record, recall, qps, total_time, build_time, index_size])

def load_search_results(solution, filename, seed=SEED, sampling_count=MAX_SAMPLING_COUNT):
    load_path = _save_path("result", solution, filename, seed, sampling_count)
    results = []
    if not os.path.exists(load_path):
        print(f"File {load_path} does not exist.")
        return results
    with open(load_path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            M, efC, efS = map(int, row[:3])
            T_record, recall, qps, total_time, build_time = map(float, row[3:8])
            index_size = int(float(row[8]))
            results.append(((M, efC, efS), (T_record, recall, qps, total_time, build_time, index_size)))
    return results

def get_optimal_hyperparameter(results, recall_min=None, qps_min=None):
    assert (recall_min is None) != (qps_min is None), "Only one of recall_min or qps_min should be set."
    
    optimal_hyperparameters = ((None, None, None), (0.0, 0.0, 0.0, 0.0, 0.0, 0))

    if recall_min is not None:
        best_qps = 0.0
        for (M, efC, efS), (tuning_time, recall, qps, total_time, build_time, index_size) in results:
            if recall >= recall_min and qps > best_qps:
                best_qps = qps
                optimal_hyperparameters = ((M, efC, efS), (
                    round(float(tuning_time), 2), round(float(recall), 3), round(float(qps), 2),
                    round(float(total_time), 2), round(float(build_time), 2), int(index_size)
                ))

    elif qps_min is not None:
        best_recall = 0.0
        for (M, efC, efS), (tuning_time, recall, qps, total_time, build_time, index_size) in results:
            if qps >= qps_min and recall > best_recall:
                best_recall = recall
                optimal_hyperparameters = ((M, efC, efS), (
                    round(float(tuning_time), 2), round(float(recall), 3), round(float(qps), 2),
                    round(float(total_time), 2), round(float(build_time), 2), int(index_size)
                ))
    return optimal_hyperparameters

def get_local_optimal_hyperparameter(results, recall_min=None, qps_min=None):
    assert (recall_min is None) != (qps_min is None), "Only one of recall_min or qps_min should be set."

    Ms = list(set([M for (M, efC, efS), _ in results]))
    Ms.sort()
    get_local_optimal_hyperparameters = []

    for cur_M in Ms:
        best_metric = 0.0
        local_opt = None
        for (M, efC, efS), (tuning_time, recall, qps, total_time, build_time, index_size) in results:
            if M != cur_M:
                continue
            if recall_min is not None:
                if recall >= (recall_min - TOLERANCE - 1e-5) and qps > best_metric:
                    best_metric = qps
                else:
                    continue
            elif qps_min is not None:
                if qps >= (qps_min - TOLERANCE - 1e-5) and recall > best_metric:
                    best_metric = recall
                else:
                    continue
            local_opt = (
                (M, efC, efS),
                (
                    round(tuning_time.item(), 2) if hasattr(tuning_time, "item") else round(tuning_time, 2),
                    round(recall.item(), 3) if hasattr(recall, "item") else round(recall, 3),
                    round(qps.item(), 2) if hasattr(qps, "item") else round(qps, 2),
                    round(total_time.item(), 2) if hasattr(total_time, "item") else round(total_time, 2),
                    round(build_time.item(), 2) if hasattr(build_time, "item") else round(build_time, 2),
                    int(index_size)
                )
            )
        if local_opt:
            get_local_optimal_hyperparameters.append(local_opt)
    return get_local_optimal_hyperparameters

def save_optimal_hyperparameters(impl, dataset, optimal_combi, recall_min=None, qps_min=None, seed=SEED, sampling_count = MAX_SAMPLING_COUNT):
    assert (recall_min is None) != (qps_min is None), "Only one of recall_min or qps_min should be set."
    save_path = _save_path("optimal_hyperparameters", "_".join(optimal_combi.keys()), f"optimal_hyperparameters_{impl}_{dataset}_{recall_min}r_{qps_min}q.csv", seed, sampling_count)
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["solution", "M", "efC", "efS", "T_record", "recall", "qps", "total_time", "build_time", "index_size"])
        for solution, (hp, perf) in optimal_combi.items():
            M, efC, efS = hp
            T_record, recall, qps, total_time, build_time, index_size = perf
            writer.writerow([solution, M, efC, efS, T_record, recall, qps, total_time, build_time, index_size])

def optimal_hyperparameters_for_times(results, recall_min=None, qps_min=None):
    assert (recall_min is None) != (qps_min is None), "Only one of recall_min or qps_min should be set."
    get_perf = lambda perf : perf[2] if recall_min else perf[1]
    times = [t for t in range(3 * 3600, TUNING_BUDGET + 1, 3 * 3600)]
    optimal_hyperparameters = [] 
    for t in times:
        filtered_results = [0.0, 0.0]
        filtered_results += [
            get_perf(perf) for hp, perf in results
            if perf[0] <= t and (perf[1] >= recall_min if recall_min else perf[2] >= qps_min)
        ]
        sorted_results = sorted(filtered_results, reverse=True)
        optimal_hyperparameters.append(sorted_results[0] if sorted_results else 0.0)
    return optimal_hyperparameters

def plot_accumulated_timestamp_on_ax(
    ax,
    results,
    recall_min=None,
    qps_min=None,
    max_perf = None,
    tuning_budget=TUNING_BUDGET,
):
    assert (recall_min is None) != (qps_min is None), "Only one of recall_min or qps_min should be set."

    markers = {
        "brute_force": 'o',
        "our_solution": 's',
        "random_search": '^',
        "grid_search": 'D',
        "optuna": "*",
        "nsga": "x",
        "random_search_heuristic": '^',
        "grid_search_heuristic": 'D',
        "vd_tuner": 'p',
        "10_tests": 's',
        "5_tests": '^',
        "3_tests": 'D',
        "1_tests": 'p',
    }
    colors = {
        "brute_force": cm.get_cmap('tab10')(0),
        "our_solution": cm.get_cmap('tab10')(1),
        "grid_search": cm.get_cmap('tab10')(2),
        "random_search": cm.get_cmap('tab10')(3),
        "grid_search_heuristic": cm.get_cmap('tab10')(2),
        "random_search_heuristic": cm.get_cmap('tab10')(3),
        "vd_tuner": cm.get_cmap('tab10')(4),
        "optuna": cm.get_cmap('tab10')(5),
        "nsga": cm.get_cmap('tab10')(6),
        "10_tests": cm.get_cmap('tab10')(1),
        "5_tests": cm.get_cmap('tab10')(2),
        "3_tests": cm.get_cmap('tab10')(3),
        "1_tests": cm.get_cmap('tab10')(4),
    }

    for solution, result in results.items():
        if recall_min:
            filtered = [
                (value[0], value[2] if max_perf >= value[2] else max_perf)
                for _, value in result
                if value[1] >= recall_min
            ]
        elif qps_min:
            filtered = [
                (value[0], value[1] if max_perf >= value[1] else max_perf)
                for _, value in result
                if value[2] >= qps_min
            ]
        if not filtered:
            continue

        filtered.insert(0, (0.0, 0.0))
        filtered.sort(key=lambda x: x[0])
        accumulated = [(0.0, 0.0)]
        max_perf = 0.0
        for t, perf in filtered:
            if perf > max_perf:
                accumulated.append((t, max_perf))
                max_perf = perf
                accumulated.append((t, perf))
        accumulated.append((tuning_budget, max_perf))

        timestamps, perf_values = zip(*accumulated)

        if solution == "brute_force":
            _label = "oracle_solution"
        elif solution == "our_solution":
            _label = "CHAT"
        else:
            _label = solution

        ax.plot(
            timestamps,
            perf_values,
            marker=markers[solution],
            color=colors[solution],
            label=_label,
            linewidth=1.0,
            markersize=3 if solution == "grid_search" else 4
        )

    ax.set_xlabel("Time (seconds)", fontsize=14)
    ax.set_ylabel("QPS" if recall_min else "Recall", fontsize=14)
    ax.set_xlim(0, tuning_budget)
    ax.grid(True)

def plot_multi_accumulated_timestamp(
    results,
    dirname,
    filename,
    recall_min=None,
    qps_min=None,
    tuning_budget=TUNING_BUDGET,
    seed=SEED,
    sampling_count=MAX_SAMPLING_COUNT,
):
    assert (recall_min is None) != (qps_min is None), "Only one of recall_min or qps_min should be set."

    save_path = _save_path("accumulated_timestamp", dirname, filename, seed, sampling_count)

    markers = {
        "brute_force": "o",
        "our_solution": "s",
        "random_search": "^",
        "grid_search": "D",
        "random_search_heuristic": "^",
        "grid_search_heuristic": "D",
        "vd_tuner": "p",
        "optuna": "*",
        "nsga": "x",
        "eci": "X",
        "test_solution": "x",
        "test_solution2": "*",
        "10_tests": "s",
        "5_tests": "^",
        "3_tests": "D",
        "1_tests": "p",
    }

    colors = {
        "brute_force": cm.get_cmap("tab10")(0),
        "our_solution": cm.get_cmap("tab10")(1),
        "grid_search": cm.get_cmap("tab10")(2),
        "random_search": cm.get_cmap("tab10")(3),
        "grid_search_heuristic": cm.get_cmap("tab10")(2),
        "random_search_heuristic": cm.get_cmap("tab10")(3),
        "vd_tuner": cm.get_cmap("tab10")(4),
        "optuna": cm.get_cmap("tab10")(5),
        "nsga": cm.get_cmap("tab10")(6),
        "eci": cm.get_cmap("tab10")(7),
        "test_solution": cm.get_cmap("tab10")(5),
        "test_solution2": cm.get_cmap("tab10")(6),
        "10_tests": cm.get_cmap("tab10")(1),
        "5_tests": cm.get_cmap("tab10")(2),
        "3_tests": cm.get_cmap("tab10")(3),
        "1_tests": cm.get_cmap("tab10")(4),
    }

    def pretty_label(solution: str) -> str:
        if solution == "brute_force":
            return "oracle_solution"
        if solution == "eci":
            return "ECI (GP)"
        if solution == "vd_tuner":
            return "VDTuner"
        if solution == "our_solution":
            return "CHAT"
        return solution

    def get_marker(solution: str) -> str:
        return markers.get(solution, "v")

    def get_color(solution: str):
        if solution in colors:
            return colors[solution]
        return cm.get_cmap("tab10")(abs(hash(solution)) % 10)

    plt.figure(figsize=(10, 6))

    for solution, result in results.items():
        if recall_min is not None:
            filtered = [
                (value[0], value[2])
                for _, value in result
                if value[1] >= recall_min
            ]
            y_label = "QPS"
        else:
            assert qps_min is not None
            filtered = [
                (value[0], value[1])
                for _, value in result
                if value[2] >= qps_min
            ]
            y_label = "Recall"

        if not filtered:
            if recall_min is not None:
                print(f"[{solution}] No results with recall >= {recall_min}")
            else:
                print(f"[{solution}] No results with qps >= {qps_min}")
            continue

        filtered.insert(0, (0.0, 0.0))
        filtered.sort(key=lambda x: x[0])

        accumulated = [(0.0, 0.0)]
        best = 0.0
        for t, perf in filtered:
            if perf > best:
                accumulated.append((t, best))
                best = perf
                accumulated.append((t, perf))

        accumulated.append((tuning_budget, best))
        timestamps, perf_values = zip(*accumulated)

        plt.plot(
            timestamps,
            perf_values,
            marker=get_marker(solution),
            color=get_color(solution),
            label=pretty_label(solution),
        )

    plt.title("Accumulated TimeStamp Plot")
    plt.xlabel("Time (seconds)")
    plt.ylabel(y_label)
    plt.xlim(0, tuning_budget)
    plt.grid(True)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_timestamp_2(results, solution, filename,
                   recall_min=None, qps_min=None,
                   tuning_budget=TUNING_BUDGET,
                   seed=SEED, sampling_count=MAX_SAMPLING_COUNT):

    assert (recall_min is None) != (qps_min is None), "Only one of recall_min or qps_min should be set."
    save_path = _save_path("timestamp", solution, filename, seed, sampling_count)

    if recall_min:
        filtered_results = [
            (value[0], value[2])
            for _, value in results
            if value[1] >= recall_min and value[0] <= tuning_budget
        ]
    if qps_min:
        filtered_results = [
            (value[0], value[1])
            for _, value in results
            if value[2] >= qps_min and value[0] <= tuning_budget
        ]

    if not filtered_results:
        print(f"No results meet the constraints.")
        return

    filtered_results.append(
        (tuning_budget, filtered_results[-1][1])
    )

    print(f"Filtered Results: {filtered_results}")

    timestamps, y_values = zip(*filtered_results)
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, y_values, marker='o')

    y_max = max(y_values)
    y_min = min(y_values)
    plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
    if len(filtered_results) >= 2:
        x_vline = filtered_results[-2][0]
        y_vline = filtered_results[-2][1]
        plt.vlines(
            x=x_vline,
            ymin=0,
            ymax=y_vline,
            colors='red',
            linestyles='--',
            label=f'Early Stop at {x_vline:.0f}s'
        )
    plt.xlabel("Time (seconds)", fontsize=22)
    plt.ylabel("QPS" if recall_min else "Recall", fontsize=24)
    plt.xlim(0, tuning_budget)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.legend(
        loc="lower center",
        bbox_to_anchor=(0.425, 1.02),
        ncol=2,
        borderaxespad=0.0,
        frameon=False,
        fontsize=22,
    )

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot at: {save_path}")

def plot_timestamp(results, solution, filename, recall_min=None, qps_min=None, tuning_budget=TUNING_BUDGET, seed=SEED, sampling_count=MAX_SAMPLING_COUNT):
    assert (recall_min is None) != (qps_min is None), "Only one of recall_min or qps_min should be set."
    save_path = _save_path("timestamp", solution, filename, seed, sampling_count)
    if recall_min:
        filtered_results = [
            (value[0], value[2])
            for _, value in results
                if value[1] >= recall_min and value[0] <= tuning_budget
        ]
    if qps_min:
        filtered_results = [
            (value[0], value[1])
            for _, value in results
                if value[2] >= qps_min and value[0] <= tuning_budget
        ]
    if not filtered_results:
        print(f"No results with recall >= {recall_min}")
        return
    timestamps, qps_values = zip(*filtered_results)
    plt.plot(timestamps, qps_values, marker='o', label=solution)
    plt.title("TimeStamp Plot")
    plt.xlabel("Time")
    plt.ylabel("QPS" if recall_min else "Recall")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_searched_points_3d(results, solution, filename, recall_min=None, qps_min=None, tuning_budget=TUNING_BUDGET, seed=SEED, sampling_count=MAX_SAMPLING_COUNT, surface=False):
    assert (recall_min is None) != (qps_min is None), "Only one of recall_min or qps_min should be set."
    
    if surface:
        name, ext = os.path.splitext(filename)
        filename = f"{name}_surface{ext}"
        
    save_path = _save_path("searched_points", solution, filename, seed, sampling_count)
    
    M_vals = []
    efC_vals = []
    perf_vals = []
    
    if recall_min:
        filtered_points = [
            (hp, perf) for hp, perf in results
            if perf[1] >= recall_min and hp[1] % 8 == 0
        ]
        perf_idx_to_plot = 2
    if qps_min:
        filtered_points = [
            (hp, perf) for hp, perf in results
            if perf[2] >= qps_min and hp[1] % 8 == 0
        ]
        perf_idx_to_plot = 1
        
    if not filtered_points:
        print(f"No valid points to plot for {filename}. Skipping.")
        return
    print(len(filtered_points), "points to plot for", filename) 
    for hp, perf in filtered_points:
        M_vals.append(hp[0])
        efC_vals.append(hp[1])
        perf_vals.append(perf[perf_idx_to_plot])

    M_vals = np.array(M_vals)
    efC_vals = np.array(efC_vals)
    perf_vals = np.array(perf_vals)

    fig = plt.figure(figsize=(20, 20))
    views = [(30, -120), (30, -210), (30, -300), (30, 30), (45, 180), (45, 0), (30, 270), (225, 60), (270, 90)]
    
    global_opt = get_optimal_hyperparameter(results, recall_min=recall_min, qps_min=qps_min)
    local_opts = get_local_optimal_hyperparameter(results, recall_min=recall_min, qps_min=qps_min)

    for i, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(3, 3, i + 1, projection='3d')
        
        if surface:
            ax.plot_trisurf(efC_vals, M_vals, perf_vals, cmap=cm.viridis, alpha=0.8, edgecolor='none')
        else:
            ax.scatter(efC_vals, M_vals, perf_vals, c=perf_vals, cmap=cm.viridis, s=20, alpha=0.7)
            for x, y, z in zip(efC_vals, M_vals, perf_vals):
                ax.plot([x, x], [y, y], [0, z], color='gray', alpha=0.3, linewidth=1)
                ax.scatter(x, y, 0, color='gray', alpha=0.3, s=5)
        
        if local_opts:
            for hp, perf in local_opts:
                M, efC, efS = hp
                if global_opt and (M, efC) == global_opt[0][:2]:
                    continue
                z_val = perf[perf_idx_to_plot]
                ax.scatter(efC, M, z_val, color='red', s=60, edgecolor='black', depthshade=True, label="Local Optima")
            
        if global_opt:
            global_hp, global_perf = global_opt
            M_g, efC_g, _ = global_hp
            z_val_g = global_perf[perf_idx_to_plot]
            ax.scatter(efC_g, M_g, z_val_g, color='blue', marker='*', s=350, edgecolor='black', depthshade=True, label="Global Optimum")

        ax.set_title(f'View {i + 1} (elev={elev}, azim={azim})')
        ax.set_xlabel('efConstruction')
        ax.set_ylabel('M')
        ax.set_zlabel('QPS' if recall_min else 'Recall')
        
        ax.set_xlim(EFC_MIN, EFC_MAX)
        ax.set_ylim(M_MIN, M_MAX)
        
        ax.view_init(elev=elev, azim=azim)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"3D plot saved to: {save_path}")

def plot_efS_3d(results, solution, filename, recall_min=None, qps_min=None, tuning_budget=TUNING_BUDGET, seed=SEED, sampling_count=MAX_SAMPLING_COUNT):
    assert (recall_min is None) != (qps_min is None), "Only one of recall_min or qps_min should be set."
    save_path = _save_path("efS_3d", solution, filename, seed, sampling_count)
    
    M_vals = []
    efC_vals = []
    efS_vals = []
    if recall_min:
        filtered_results = [
            hp
            for hp, perf  in results
            if perf[1] >= recall_min and hp[0] % 2 == 0 and hp[1] % 16 == 0
        ]
    if qps_min:
        filtered_results = [
            hp
            for hp, perf in results
            if perf[2] >= qps_min and hp[0] % 2 == 0 and hp[1] % 16 == 0
        ]
    for M, efC, efS in filtered_results:
        M_vals.append(M)
        efC_vals.append(efC)
        efS_vals.append(efS)

    M_vals = np.array(M_vals)
    efC_vals = np.array(efC_vals)
    efS_vals = np.array(efS_vals)

    fig = plt.figure(figsize=(20, 20))
    views = [(30, -120), (30, -210), (30, -300), (30, 30), (45, 180), (45, 0), (30, 270), (225, 60), (270, 90)]

    for i, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(3, 3, i+1, projection='3d')
        ax.scatter(efC_vals, M_vals, efS_vals, c=efS_vals, cmap=cm.viridis, s=20, alpha=0.7)
        for x, y, z in zip(efC_vals, M_vals, efS_vals):
            ax.plot([x, x], [y, y], [0, z], color='gray', alpha=0.3, linewidth=1)
            if z > 0:   ax.scatter(x, y, 0, color='gray', alpha=0.3, s=5)
            else:       ax.scatter(x, y, 0, color="black", alpha=0.95, s=25)
        ax.set_title(f'View {i+1}')
        ax.set_xlabel('efC')
        ax.set_ylabel('M')
        ax.set_zlabel('efS')
        ax.set_xlim(EFC_MIN, EFC_MAX)
        ax.set_ylim(M_MIN, M_MAX)
        ax.view_init(elev=elev, azim=azim)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def _feasible_and_objective_factory(recall_min, qps_min, tuning_budget, max_perf):
    assert (recall_min is None) != (qps_min is None), "Only one of recall_min or qps_min should be set."

    if recall_min is not None:
        y_label = "QPS"

        def is_feasible(perf):
            return (perf[0] <= tuning_budget) and (perf[1] >= (recall_min - TOLERANCE - 1e-5))

        def objective(perf):
            return float(perf[2] if max_perf >= perf[2] else max_perf)

    else:
        y_label = "Recall"

        def is_feasible(perf):
            return (perf[0] <= tuning_budget) and (perf[2] >= (qps_min - TOLERANCE - 1e-5))

        def objective(perf):
            return float(perf[1] if max_perf >= perf[1] else max_perf)

    return is_feasible, objective, y_label


def _oracle_best(results_dict, is_feasible, objective):
    best = None
    for _, perf in results_dict.get("brute_force", []):
        if is_feasible(perf):
            v = objective(perf)
            if best is None or v > best:
                best = v
    return best


def _time_of_reaching_each_y(sol_list, is_feasible, objective, tuning_budget, y_bins=240):
    pts = []
    for _, perf in sol_list:
        if not is_feasible(perf):
            continue
        t = float(perf[0])
        m = objective(perf)
        pts.append((t, m))

    if not pts:
        return None, 0.0

    pts.sort(key=lambda x: x[0])

    segs = []
    best = 0.0
    for t, m in pts:
        if m > best:
            segs.append((best, m, t))
            best = m

    if best <= 0.0:
        return None, 0.0

    ys = np.linspace(0.0, best, y_bins, endpoint=False)
    t_of_y = np.zeros_like(ys)

    t_of_y[:] = float(tuning_budget)

    for y0, y1, t in segs:
        if y1 <= y0:
            continue
        mask = (ys >= y0) & (ys < y1)
        t_of_y[mask] = float(t)

    return t_of_y, best


def plot_gradient_bar_with_oracle_on_ax(
    ax,
    results_dict,
    recall_min=None,
    qps_min=None,
    tuning_budget=TUNING_BUDGET,
    max_perf=None,
    cmap_name="Greys",
    y_bins=240,
    show_xticklabels=True,
    row=0,
): 
    is_feasible, objective, y_label = _feasible_and_objective_factory(recall_min, qps_min, tuning_budget, max_perf)

    order = ["our_solution", "vd_tuner", "optuna", "nsga", "random_search", "grid_search"]
    label_map = {
        "our_solution": "CHAT",
        "vd_tuner": "VDTuner",
        "optuna": "Optuna",
        "nsga": "NSGA-II",
        "random_search": "Random",
        "grid_search": "Grid",
    }

    norm = colors.Normalize(vmin=0.0, vmax=float(tuning_budget))
    cmap = cm.get_cmap(cmap_name)

    bar_w = 0.75
    xs = np.arange(len(order))

    max_y = 0.0

    for i, key in enumerate(order):
        sol_list = results_dict.get(key, [])
        t_of_y, best = _time_of_reaching_each_y(sol_list, is_feasible, objective, tuning_budget, y_bins=y_bins)
        max_y = max(max_y, best)

        if t_of_y is None or best <= 0.0:
            ax.bar(i, 0.0, width=bar_w, edgecolor="0.6", facecolor="none", linewidth=0.8)
            continue

        img = t_of_y.reshape(-1, 1)

        ax.imshow(
            img,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            norm=norm,
            extent=(i - bar_w / 2.0, i + bar_w / 2.0, 0.0, best),
            interpolation="nearest",
            zorder=2,
        )

        ax.plot(
            [i - bar_w / 2.0, i + bar_w / 2.0, i + bar_w / 2.0, i - bar_w / 2.0, i - bar_w / 2.0],
            [0.0, 0.0, best, best, 0.0],
            linewidth=0.6,
            color="0.4",
            zorder=3,
        )

    oracle_val = _oracle_best(results_dict, is_feasible, objective)
    if oracle_val is not None:
        max_y = max(max_y, oracle_val)
        ax.axhline(oracle_val, linestyle="--", linewidth=1.1, color=cm.get_cmap("tab10")(0), label="Oracle", zorder=4)

        if oracle_val > 0:
            custom_ticks = np.linspace(0, oracle_val, 5)
            ax.set_yticks(custom_ticks)
            
            from matplotlib.ticker import FormatStrFormatter
            if oracle_val > 100: ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            else: ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax.set_xlim(-0.6, len(order) - 0.4)

    if max_y > 0:
        ax.set_ylim(0, max_y * 1.10)

    ax.grid(True, axis="y", alpha=0.25, zorder=1)

    ax.set_xticks(xs)
    if show_xticklabels:
        ax.set_xticklabels([label_map.get(k, k) for k in order], rotation=30, ha="right", fontsize=9)
    else:
        ax.set_xticklabels([])

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    return sm

def get_best_perf_at_time(sol_list, is_feasible, objective, time_limit):
    valid_perfs = [
        objective(perf) 
        for _, perf in sol_list 
        if is_feasible(perf) and perf[0] <= time_limit
    ]
    
    if not valid_perfs:
        return 0.0
    return max(valid_perfs)
