from src.constants import SEED, MAX_SAMPLING_COUNT
from src.utils import get_local_optimal_hyperparameter, get_optimal_hyperparameter, plot_efS_3d, \
    plot_multi_accumulated_timestamp, plot_searched_points_3d, plot_timestamp, save_search_results, \
    plot_timestamp_2

def print_optimal_hyperparameters(results, recall_min=None, qps_min=None):
    global_opt_hp = get_optimal_hyperparameter(results, recall_min=recall_min, qps_min=qps_min)
    print("[Optimal Hyperparameter]")
    if global_opt_hp is None:
        print("No optimal hyperparameter found")
        # return
        return None
    print(f"{global_opt_hp[0]} | {global_opt_hp[1]}")
    print("[Local Optimal Hyperparameters]")
    local_opt_hps = get_local_optimal_hyperparameter(results, recall_min=recall_min, qps_min=qps_min)
    for hp, perf in local_opt_hps:
        print(f"{hp} | {perf}")
    print("")
    return global_opt_hp, local_opt_hps

def postprocess_results(results, solution, impl, dataset, tuning_budget, 
                        recall_min=None, qps_min=None, seed=SEED, sampling_count=MAX_SAMPLING_COUNT, lite=False):
    save_search_results(
        results, 
        solution=solution, filename=f"{solution}_{impl}_{dataset}_{recall_min}r_{qps_min}q.csv", 
        seed=seed, sampling_count=sampling_count
    )
    plot_timestamp(
        results, 
        solution=solution, filename=f"{solution}_{impl}_{dataset}_{recall_min}r_{qps_min}q.png", 
        recall_min=recall_min, qps_min=qps_min, tuning_budget=tuning_budget, seed=seed, sampling_count=sampling_count
    )
    # plot_timestamp_2(
    #     results, 
    #     solution=solution, filename=f"{solution}_{impl}_{dataset}_{recall_min}r_{qps_min}q.pdf", 
    #     recall_min=recall_min, qps_min=qps_min, tuning_budget=tuning_budget, seed=seed, sampling_count=sampling_count
    # )
    plot_multi_accumulated_timestamp(
        {solution: results},
        dirname=solution,
        filename=f"{solution}_multi_accumulated_timestamp_{impl}_{dataset}_{recall_min}r_{qps_min}q.png", 
        recall_min=recall_min, qps_min=qps_min, tuning_budget=tuning_budget, seed=seed, sampling_count=sampling_count
    )
    if lite:
        return
    try:
        plot_searched_points_3d(
            results, 
            solution=solution, 
            filename=f"{solution}_searched_points_3d_{impl}_{dataset}_{recall_min}r_{qps_min}q.png", 
            recall_min=recall_min, qps_min=qps_min, tuning_budget=tuning_budget, seed=seed, sampling_count=sampling_count
        )
        plot_efS_3d(
            results, 
            solution=solution, 
            filename=f"{solution}_searched_points_3d_{impl}_{dataset}_{recall_min}r_{qps_min}q.png", 
            recall_min=recall_min, qps_min=qps_min, tuning_budget=tuning_budget, seed=seed, sampling_count=sampling_count
        )
    except Exception as e:
        print(f"Error in plotting 3D graphs: {e}")