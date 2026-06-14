"""Experiment helpers for generating HNSW ground-truth CSV files.

This module intentionally does not instantiate datasets at import time. The
artifact runner should be able to start even when only one dataset is present.
"""

import os

from auto_tuner.models.hnsw_config import HnswConfig, hnsw_config_mapping, save_results_to_csv
from auto_tuner.postprocess import ResultProcessor

def score_config(config):
    try:
        return config.score(HnswConfig.recall_min, HnswConfig.qps_min)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e

def summary_from_csv(filename:str, dir:str):
    """Load an RQ CSV, recompute summary scores, and emit summary plots."""
    print(f"summary_from_csv: {filename}")
    if ".csv" not in filename:
        filename += ".csv"
    hnsw_configs = HnswConfig.from_csv(filename, dir)
    for config in hnsw_configs:
        score_config(config)
    result_processor = ResultProcessor(hnsw_configs, filename=filename, smoothen=True)
    result_processor.plot_score()
    result_processor.plot_recall()
    result_processor.plot_qps()
    result_processor.plot_recall_qps()
    result_processor.plot_build_time()
    result_processor.plot_index_size()


def summary_from_dir(dir:str, include:list[str]=None):
    """Summarize every CSV in a directory matching all include patterns."""
    print(f"summary_from_dir: {dir}")
    include = list(include or [])
    include.append("RQ")
    target_files = []
    for filename in os.listdir(dir):
        flag = True
        for pattern in include:
            if pattern not in filename:
                flag = False
                break
        if flag:
            target_files.append(filename)
    for filename in target_files:
        summary_from_csv(filename, dir)


def csv_files_in_dir(dir:str, patterns:list[str]=None):
    """Return CSV files under dir whose path contains every requested pattern."""
    patterns = list(patterns or [])
    patterns.append(".csv")
    csv_files = []
    for dirpath, _, filenames in os.walk(dir):
        for filename in filenames:
            if all(pattern in filename for pattern in patterns):
                csv_files.append(os.path.join(dirpath, filename))
    return csv_files

def run_hnsw_config(dataset, impl, M=None, efC=None, params=None, prefix="", warmup=True):
    """Run a grid of HNSW configs and save both raw RQ and score CSV files."""
    M = list(M or [])
    efC = list(efC or [])
    params = list(params or [])
    if warmup:
        print(f"Warmup for {dataset.name} ...")
        hnsw_config_mapping[impl](dataset, 8, 8, batch=True).score(0, 0)

    hnsw_configs = []
    if len(params) != 0:
        for m, efc in params:
            hnsw_config = hnsw_config_mapping[impl](dataset, m, efc, batch=True)
            hnsw_configs.append(hnsw_config)
    else:
        for m in M:
            for ef in efC:
                if ef < m:
                    continue
                hnsw_config = hnsw_config_mapping[impl](dataset, m, ef, batch=True)
                hnsw_configs.append(hnsw_config)
    print(f"Running {len(hnsw_configs)} configurations ...")
    for config in hnsw_configs:
        print(config)
    d, _ , f = save_results_to_csv(hnsw_configs, prefix=prefix)
    summary_from_csv(f, dir=d)
