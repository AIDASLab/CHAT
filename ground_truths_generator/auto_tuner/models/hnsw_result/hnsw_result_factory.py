import numpy as np
from scipy.interpolate import LinearNDInterpolator
from auto_tuner.models.hnsw_result import HnswResult
from auto_tuner.constants import DEFAULT_EFS

class HnswResultFactory:
    def __init__(self, impl:str, dataset:str, M_min=4, M_max=64, efC_min=16, efC_max=512, efS_min=32, efS_max=1024):
        self.impl = impl
        self.dataset = dataset
        self.M_min = M_min
        self.M_max = M_max
        self.efC_min = efC_min
        self.efC_max = efC_max
        self.efS_min = efS_min
        self.efS_max = efS_max
        self.ground_truth = dict()  # (M, efC, efS) -> HnswResult
    
    def init_ground_truth(self, dir=None, smoothen=True):
        import os
        from auto_tuner.scripts import csv_files_in_dir
        from auto_tuner.models.hnsw_config import HnswConfig
        
        if dir is None:
            dir = os.path.join(os.environ["ROOT_DIR"], "main_results")
        if not os.path.exists(dir):
            return
        files = csv_files_in_dir(dir, patterns=[self.dataset, self.impl, "RQ"])
        for file in files:
            configs = HnswConfig.from_csv(file)
            for config in configs:
                for result in config.results.values():
                    _M, _efC, _efS = result.M, result.efC, result.efS
                    if (_M, _efC, _efS) not in self.ground_truth:
                        self.ground_truth[(_M, _efC, _efS)] = result
                    else:   #* Duplicate
                        self.ground_truth[(_M, _efC, _efS)].build_time \
                            = (self.ground_truth[(_M, _efC, _efS)].build_time + result.build_time) / 2
                        self.ground_truth[(_M, _efC, _efS)].index_size \
                            = (self.ground_truth[(_M, _efC, _efS)].index_size + result.index_size) / 2
                        self.ground_truth[(_M, _efC, _efS)].test_time \
                            = (self.ground_truth[(_M, _efC, _efS)].test_time + result.test_time) / 2
                        self.ground_truth[(_M, _efC, _efS)].search_times += result.search_times
                        self.ground_truth[(_M, _efC, _efS)]._recall_qps += result._recall_qps
        #* Smoothen the all metrics
        if smoothen:
            from scipy.ndimage import gaussian_filter
            sigma = 1
            # metric_names = ["build_time", "test_time", "search_time", "recall", "qps"]
            metric_names = ["qps"]
            efS_set = sorted(set(r.efS for r in self.ground_truth.values()))
            unique_M = sorted(set(r.M for r in self.ground_truth.values()))
            unique_efC = sorted(set(r.efC for r in self.ground_truth.values()))

            def get_metric(r, name):
                if name == "qps":
                    return r.qps
                elif name == "recall":
                    return r.recall
                elif name == "search_time":
                    return r.search_time
                else:
                    return getattr(r, name)

            def set_metric(r, name, value):
                if name == "qps":
                    r._recall_qps = [(r.recall, value)]
                elif name == "recall":
                    r._recall_qps = [(value, r.qps)]
                elif name == "search_time":
                    r.search_times = [value]
                else:
                    setattr(r, name, value)

            for metric in metric_names:
                for efS in efS_set:
                    matrix = np.zeros((len(unique_M), len(unique_efC)))

                    for r in self.ground_truth.values():
                        if r.efS != efS:
                            continue
                        i = unique_M.index(r.M)
                        j = unique_efC.index(r.efC)
                        matrix[i, j] = get_metric(r, metric)

                    smoothed = gaussian_filter(matrix, sigma=sigma)

                    for r in self.ground_truth.values():
                        if r.efS != efS:
                            continue
                        i = unique_M.index(r.M)
                        j = unique_efC.index(r.efC)
                        set_metric(r, metric, float(smoothed[i, j]))
        #* Generate the Interpolator
        results = self.ground_truth.values()
        points = np.array([(r.M, r.efC, r.efS) for r in results])
        
        build_times = np.array([r.build_time for r in results])
        index_sizes = np.array([r.index_size for r in results])
        test_times  = np.array([r.test_time for r in results])
        search_times_vals = np.array([r.search_time for r in results])
        recalls = np.array([r.recall for r in results])
        qps_vals = np.array([r.qps for r in results])
        
        self.interp_build_time  = LinearNDInterpolator(points, build_times)
        self.interp_index_size  = LinearNDInterpolator(points, index_sizes)
        self.interp_test_time   = LinearNDInterpolator(points, test_times)
        self.interp_search_time = LinearNDInterpolator(points, search_times_vals)
        self.interp_recall      = LinearNDInterpolator(points, recalls)
        self.interp_qps         = LinearNDInterpolator(points, qps_vals)
    
    def __safe_interp(self, metric, value):
        if value is None or np.isnan(value):
            raise ValueError("Interpolating is failed. Value is None or NaN")
        if metric == "build_time" and value < 0:
            raise ValueError("Interpolating is failed. Build time is negative")
        if metric == "test_time" and value < 0:
            raise ValueError("Interpolating is failed. Test time is negative")
        if metric == "search_time" and value < 0:
            raise ValueError("Interpolating is failed. Search time is negative")
        if metric == "recall" and not (0.0 <= value and value <= 1.0):
            raise ValueError("Interpolating is failed. Recall is out of range [0, 1]")
        if metric == "qps" and value < 0:
            raise ValueError("Interpolating is failed. QPS is negative")
        return float(value)
    
    def get_result(self, M:int, efC:int, efS:int):
        if (M, efC, efS) in self.ground_truth:
            return self.ground_truth[(M, efC, efS)]
        M_new, efC_new, efS_new = M, efC, efS 
        new_build_time  = self.__safe_interp("build_time", self.interp_build_time(M_new, efC_new, efS_new))
        new_index_size  = self.__safe_interp("index_size", self.interp_index_size(M_new, efC_new, efS_new))
        new_test_time   = self.__safe_interp("test_time", self.interp_test_time(M_new, efC_new, efS_new))
        new_search_time = self.__safe_interp("search_time", self.interp_search_time(M_new, efC_new, efS_new))
        new_recall      = self.__safe_interp("recall", self.interp_recall(M_new, efC_new, efS_new))
        new_qps         = self.__safe_interp("qps", self.interp_qps(M_new, efC_new, efS_new))
        new_result = HnswResult(
            M = M_new,
            efC = efC_new,
            efS = efS_new,
            build_time = new_build_time,
            index_size = new_index_size,
            test_time = new_test_time,
            search_times = [new_search_time],
            recall_qps = [(new_recall, new_qps)]
        )
        self.ground_truth[(M_new, efC_new, efS_new)] = new_result
        return new_result
    
    def get_hnsw_config(self, M:int, efC:int):
        from auto_tuner.models.hnsw_config import hnsw_config_mapping
        from auto_tuner.dataset import dataset_mapping
        results = []
        for efS in DEFAULT_EFS:
            results.append(self.get_result(M, efC, efS))
        dataset = dataset_mapping[self.dataset](impl=self.impl)
        hnsw_config = hnsw_config_mapping[self.impl](dataset, M, efC, efS=DEFAULT_EFS, build_time=results[0].build_time)
        for result in results:
            hnsw_config.results[result.efS] = result
        return hnsw_config
        
        