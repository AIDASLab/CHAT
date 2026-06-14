import sys
import os
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

from auto_tuner.scripts import summary_from_dir, summary_from_csv
from auto_tuner.models.hnsw_config import HnswConfig

# QPS_MIN = [0, 10000, 15000, 20000, 25000, 30000, 35000, 40000]
# RECALL_MIN = [0.85,0.9, 0.93, 0.95, 0.97, 0.96, 0.98, 0.99, 0.995] 
QPS_MIN = [1500, 2500, 5000, 7500, 10000, 12500,]
RECALL_MIN = [0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995]
TARGET_DIR = "./gd" #! <- Change this to your target directory
# TARGET_FILE = "main_glove_hnswlib_RQ_glove-100-angular_0000_000000"
TARGET_FILE = "main_dbpediaentity_hnswlib_RQ_dbpediaentity-768-angular_0000_000000"
INCLUDE = ["RQ", "hnswlib"]
OPT = sys.argv[1] if len(sys.argv) > 1 else "1"

def run_config(args):
    import gc
    recall_min, qps_min, opt = args
    HnswConfig.recall_min = recall_min
    HnswConfig.qps_min = qps_min

    if opt == "1":
        summary_from_csv(TARGET_FILE, TARGET_DIR)
    elif opt == "2":
        summary_from_dir(TARGET_DIR, include=INCLUDE)

    # Optional: Garbage collect to minimize memory footprint
    gc.collect()

    return f"Done: recall_min={recall_min}, qps_min={qps_min}"

if __name__ == "__main__":
    args_list = [(recall_min, qps_min, OPT) for recall_min, qps_min in product(RECALL_MIN, QPS_MIN)]

    max_workers = max(1, os.cpu_count() - 4)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_config, args) for args in args_list]
        for future in as_completed(futures):
            print(future.result())
