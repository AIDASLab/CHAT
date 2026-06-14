import ast
import os
from scipy.interpolate import LinearNDInterpolator
from csv import reader
import numpy as np
import random
from src.constants import SEED, MAX_SAMPLING_COUNT
from src.constants import EFS_MIN, EFS_MAX, TOLERANCE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

random.seed(SEED)   # SEED = 42

class GroundTruth:
    def __init__(self, impl: str, dataset: str, sampling_count=None):
        self.impl = impl
        self.dataset = dataset
        self.__sampling_count = MAX_SAMPLING_COUNT if sampling_count == None else sampling_count
        self.data = self.load_ground_truths(impl, dataset)
        self.init()
        self.tuning_time = 0.0
        self.current_hp = (0, 0)
        self.searched_cache = dict()
        self.searched_timestamp = dict()
        self._get_count = 0
        
    def init(self):
        configs = list(self.data.keys())
        points = np.array(configs)
        recalls = np.array([v[0] for v in self.data.values()])
        qpss = np.array([v[1] for v in self.data.values()])
        total_times = np.array([v[2] for v in self.data.values()])
        build_times = np.array([v[3] for v in self.data.values()])
        index_sizes = np.array([v[4] for v in self.data.values()])

        self.interp_recall = LinearNDInterpolator(points, recalls)
        self.interp_qps = LinearNDInterpolator(points, qpss)
        self.interp_total_time = LinearNDInterpolator(points, total_times)
        self.interp_build_time = LinearNDInterpolator(points, build_times)
        self.interp_index_size = LinearNDInterpolator(points, index_sizes)

    def load_ground_truths(self, impl: str="hnswlib", dataset: str="nytimes"):
        results = dict()
        filename = f"{impl}/{dataset}.csv"
        filename = os.path.join(BASE_DIR, filename)
        with open(filename, "r") as f:
            csv_reader = reader(f)
            header = next(csv_reader)
            for row in csv_reader:
                _M = int(row[4])
                _efC = int(row[5])
                _efS = int(row[6])
                _index_size = int(float(row[7]))
                _build_time = float(row[8])
                _total_time = float(row[8]) + float(row[9])
                __recall_raw = ast.literal_eval(row[11])
                __qps_raw = ast.literal_eval(row[12])
                if self.__sampling_count and len(__recall_raw) >= self.__sampling_count:
                    sample_indices = random.sample(range(len(__recall_raw)), self.__sampling_count)
                    __recall = [__recall_raw[i] for i in sample_indices]
                    __qps = [__qps_raw[i] for i in sample_indices]
                else:
                    __recall = __recall_raw
                    __qps = __qps_raw
                _recall = np.array(__recall).mean()
                _qps = np.array(__qps).mean()
                results[(_M, _efC, _efS)] = (_recall, _qps, _total_time, _build_time, _index_size)
        return results

    def __safe_interp(self, value):
        if value is None or np.isnan(value):
            raise ValueError("Interpolation failed due to NaN or None value.")
        return value

    def get(self, M, efC, efS, tracking_time=True):
        if efS < EFS_MIN:
            return 0.0, 0.0, 0.0, 0.0, 0
        if (M, efC, efS) in self.searched_cache:
            return self.searched_cache[(M, efC, efS)]
        self._get_count += 1
        try:
            recall = self.__safe_interp(self.interp_recall(M, efC, efS))
            qps = self.__safe_interp(self.interp_qps(M, efC, efS))
            total_time = self.__safe_interp(self.interp_total_time(M, efC, efS))
            build_time = self.__safe_interp(self.interp_build_time(M, efC, efS))
            index_size = self.__safe_interp(self.interp_index_size(M, efC, efS))
        except ValueError as e:
            return 0.0, 0.0, 0.0, 0.0, 0
        if tracking_time:
            self.tuning_time += total_time
            if (M, efC) == self.current_hp:
                self.tuning_time -= build_time
            self.searched_cache[(M, efC, efS)] = (recall, qps, total_time, build_time, index_size)
            self.searched_timestamp[(M, efC, efS)] = self.tuning_time
        self.current_hp = (M, efC)
        return float(recall), float(qps), float(total_time), float(build_time), int(index_size)
    
    def _get_efS_for_qps(self, M, efC, target_qps, method, efS_min, efS_max, tolerance):
        qps_min = self.get(M, efC, efS_min)[1]
        if qps_min < target_qps:
            return 0
        best_efS = None
        best_diff = float("inf")
        if method == "linear":
            for efS in range(efS_min, efS_max+1):
                recall, qps, *_ = self.get(M, efC, efS)
                if qps >= target_qps:
                    best_efS = efS
                    break
        elif method == "binary":
            left, right = efS_min, efS_max
            while left <= right:
                mid = (left + right) // 2
                recall, qps, *_ = self.get(M, efC, mid)
                if qps == 0.0:
                    right = mid - 1
                    continue
                diff = abs(qps - target_qps)
                if diff < best_diff and qps >= target_qps:
                    best_diff = diff
                    best_efS = mid 
                if qps < target_qps:
                    right = mid - 1
                else:
                    left = mid + 1
        else:
            raise ValueError(f"Unknown method '{method}'")
        return best_efS if best_efS else 0 
    
    def _get_efS_for_recall(self, M, efC, target_recall, method, efS_min, efS_max, tolerance):
        recall_min = self.get(M, efC, efS_min)[0]
        if recall_min >= target_recall:
            return efS_min
        best_efS = None
        best_diff = float("inf")
        if method == "linear":
            for efS in range(efS_min, efS_max+1):
                recall, *_ = self.get(M, efC, efS)
                if recall >= target_recall:
                    best_efS = efS
                    break
        elif method == "binary":
            left, right = efS_min, efS_max
            while left <= right:
                mid = (left + right) // 2
                recall, qps, *_ = self.get(M, efC, mid)
                if recall == 0.0:
                    left = mid + 1
                    continue
                diff = abs(recall - target_recall)
                if diff < best_diff and recall >= target_recall:
                    best_diff = diff
                    best_efS = mid
                if recall < target_recall:
                    left = mid + 1
                else:
                    right = mid - 1
        else:
            raise ValueError(f"Unknown method '{method}'")
        return best_efS if best_efS else 0
    
    def get_efS(self, M, efC, target_recall=None, target_qps=None, method="binary", efS_min=EFS_MIN, efS_max=EFS_MAX, tolerance=TOLERANCE, skip_time=False):
        assert (target_recall is None) != (target_qps is None), "Only one of recall_min or qps_min should be set."
        if target_recall:
            return self._get_efS_for_recall(M, efC, target_recall, method, efS_min, efS_max, tolerance)
        if target_qps:
            return self._get_efS_for_qps(M, efC, target_qps, method, efS_min, efS_max, tolerance)
        raise ValueError("Critical error: this should never happen.") 

    
if __name__ == "__main__":
    gd = GroundTruth("faiss", "deep1M-256-angular")
    print(gd.get(48, 202, 32))
    print(gd.get(48, 202, 100))