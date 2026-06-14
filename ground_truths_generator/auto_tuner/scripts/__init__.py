from auto_tuner.dataset import dataset_mapping
from auto_tuner.models.hnsw_config import HnswConfig, save_results_to_csv
from auto_tuner.models.hnsw_config import HnswConfigHnswlib, HnswConfigFaiss, hnsw_config_mapping
from auto_tuner.postprocess import ResultProcessor

def score_config(config):
    try:
        return config.score(HnswConfig.recall_min, HnswConfig.qps_min)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e

def summary_from_csv(filename:str, dir:str):
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


def summary_from_dir(dir:str, include:list[str]=[]):
    print(f"summary_from_dir: {dir}")
    include.append("RQ")
    import os
    target_files = []
    for filename in os.listdir(dir):
        flag = True
        for pattern in include:
            if pattern not in filename:
                flag = False
                break
        if flag:
            target_files.append(filename)
    [summary_from_csv(f, dir) for f in target_files]

######################################

def csv_files_in_dir(dir:str, patterns:list[str]=[]):
    import os
    patterns += [".csv"]
    csv_files = []
    for dirpath, _, filenames in os.walk(dir):
        for filename in filenames:
            if all(pattern in filename for pattern in patterns):
                csv_files.append(os.path.join(dirpath, filename))
    return csv_files

def run_hnsw_config(dataset, impl, M=[], efC=[], params=[], prefix="", warmup=True):
    if warmup:
        print(f"Warmup for {dataset.name} ...")
        hnsw_config_mapping[impl](dataset, 8, 8, batch=True).score(0, 0)
    ####
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


# nytimes-256-angular
dataset_nytimes_hnswlib = dataset_mapping["nytimes-256-angular"](recompute=True, impl="hnswlib")
dataset_nytimes_faiss = dataset_mapping["nytimes-256-angular"](recompute=True, impl="faiss")
# glove-100-angular
dataset_glove_hnswlib = dataset_mapping["glove-100-angular"](recompute=True, impl="hnswlib")
dataset_glove_faiss = dataset_mapping["glove-100-angular"](recompute=True, impl="faiss")
# sift-128-euclidean
dataset_sift_hnswlib = dataset_mapping["sift-128-euclidean"](recompute=True, impl="hnswlib")
dataset_sift_faiss = dataset_mapping["sift-128-euclidean"](recompute=True, impl="faiss")
# youtube-1024-angular
dataset_youtube_hnswlib = dataset_mapping["youtube-1024-angular"](recompute=True, impl="hnswlib")
dataset_youtube_faiss = dataset_mapping["youtube-1024-angular"](recompute=True, impl="faiss")
# msmarco-384-angular
dataset_msmarco_hnswlib = dataset_mapping["msmarco-384-angular"](recompute=True, impl="hnswlib")
dataset_msmarco_faiss = dataset_mapping["msmarco-384-angular"](recompute=True, impl="faiss")
# dbpediaentity-768-angular
dataset_dbpediaentity_hnswlib = dataset_mapping["dbpediaentity-768-angular"](recompute=True, impl="hnswlib")
dataset_dbpediaentity_faiss = dataset_mapping["dbpediaentity-768-angular"](recompute=True, impl="faiss")

## ORIGINAL DATASETS : TODO remove it
dataset_og_nytimes = dataset_mapping["nytimes-256-angular"](recompute=False)
dataset_og_glove = dataset_mapping["glove-100-angular"](recompute=False)
dataset_og_sift = dataset_mapping["sift-128-euclidean"](recompute=False)
dataset_og_youtube = dataset_mapping["youtube-1024-angular"](recompute=False)