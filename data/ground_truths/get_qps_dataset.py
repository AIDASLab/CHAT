from data.ground_truths.ground_truth import GroundTruth
import numpy as np

DEFAULT_QPS_KEYS = ("q75",)


def get_qps_metrics_dataset(impl, dataset, ret_dict=False, keys=DEFAULT_QPS_KEYS):
    """Return representative QPS thresholds for one ground-truth surface.

    By default experiments use q75, matching the paper scripts. Set ret_dict=True
    when callers need all available quantiles.
    """
    gd = GroundTruth(impl=impl, dataset=dataset, sampling_count=10)
    results = gd.load_ground_truths(impl=impl, dataset=dataset)
    QPSs = [qps for _, qps, *__ in results.values()]

    q50 = int(np.quantile(QPSs, 0.50))
    q60 = int(np.quantile(QPSs, 0.60))
    q70 = int(np.quantile(QPSs, 0.70))
    q75 = int(np.quantile(QPSs, 0.75))
    q80 = int(np.quantile(QPSs, 0.80))
    q90 = int(np.quantile(QPSs, 0.90))
    q95 = int(np.quantile(QPSs, 0.95))
    metrics = {
        "q50": q50,
        "q60": q60,
        "q70": q70,
        "q75": q75,
        "q80": q80,
        "q90": q90,
        "q95": q95,
    }
    if ret_dict:
        return metrics
    return tuple(metrics[key] for key in keys)

if __name__ == "__main__":
    for IMPL in ["hnswlib", "faiss"]:
        for DATASET in ["nytimes-256-angular", "sift-128-euclidean", "glove-100-angular",
                        "deep1M-256-angular", "youtube-1024-angular"]:
            print(f"\n**** {DATASET} ****")
            for metric, value in get_qps_metrics_dataset(IMPL, DATASET, ret_dict=True).items():
                print(f"{metric: 7}: {value}")
