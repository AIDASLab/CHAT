import os
from functools import lru_cache
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
import csv
import time
import gc


from auto_tuner.dataset import dataset_mapping, Dataset
from auto_tuner.utils import log_execution

from auto_tuner.models.hnsw_result import HnswResult
from auto_tuner.constants import MAX_THREADS, RESULTS_DIR, DEFAULT_EFS, ITERS, RECALL_MIN, QPS_MIN, INTERP_KIND


class HnswConfig:
    recall_min = RECALL_MIN
    qps_min = QPS_MIN

    def __init__(self, dataset:Dataset, M:int, efC:int, batch=True, efS=DEFAULT_EFS, iters=ITERS, build_time=0.0, index_size=0):
        self.dataset = dataset
        self.M = M
        self.efC = efC
        self.batch = batch
        self.iters = iters
        self.build_time = build_time
        ##
        self.impl = "hnswlib"   # Set the default implementation
        self._index = None
        self._efS = efS
        self.results = dict()           # efS -> HnswResult
        ##
        self.index_size = index_size
        self._interp_recall = None
        self._interp_qps = None
        self._interp_recall_max_qps = -1


    @property
    def test_time(self):
        return sum([result.test_time for result in self.results.values()])


    @property
    def total_time(self):
        return self.build_time + self.test_time


    @property
    def recall(self):
        return [results.recall for results in self.results.values()]


    @property
    def qps(self):
        return [results.qps for results in self.results.values()]

    @lru_cache(maxsize=None)
    def qps_recall(self, recall):
        f_qps = interp1d(np.array(self.recall), np.array(self.qps), kind=INTERP_KIND, bounds_error=False, fill_value="extrapolate")
        qps = f_qps(recall)
        return qps

    @lru_cache(maxsize=None)
    def __get_recall_qps_for_score(self):
        _recall = []; _qps = []
        if self.results:
            _recall = [result.recall for result in self.results.values()]
            _qps = [result.qps for result in self.results.values()]
        else:
            if self._index is None:
                self._build()
            for efS in self._efS:
                __recall, __qps = self._evaluate(efS)
                _recall.append(__recall)
                _qps.append(__qps)
        _recall = np.array(_recall)
        _qps = np.array(_qps)
        if not np.all(np.diff(_recall) > 0):
            _sort_idx = np.argsort(_recall)
            _recall = _recall[_sort_idx]
            _qps = _qps[_sort_idx]
        return _recall, _qps

    @lru_cache(maxsize=None)
    def _f_recall(self, qps):
        _recall, _qps = self.__get_recall_qps_for_score()
        if qps > _qps[0]:
            raise ValueError("Out of QPS")
        if qps < _qps[-1]:
            return _recall[-1]
        if self._interp_recall or self._interp_recall_max_qps >= qps:
            return self._interp_recall(qps).item()
        # raw interpolation
        f_recall_raw = interp1d(_qps[::-1], _recall[::-1], kind=INTERP_KIND, bounds_error=False, fill_value="extrapolate")

        num_points = int(np.ceil(max(_qps[0], qps) - _qps[-1]))
        num_points = max(num_points, 1)
        qps_dense = np.linspace(_qps[-1], max(_qps[0], qps), num_points)

        recall_dense = f_recall_raw(qps_dense)

        qps_norm = qps_dense / (np.max(qps_dense) + 1e-8)
        dy = np.gradient(recall_dense, qps_norm)
        THRESHOLD = 1e-3
        flat_mask = np.abs(dy) < THRESHOLD
        flat_index = np.argmax(flat_mask) if np.any(flat_mask) else len(recall_dense)

        recall_dense[flat_index:] = _recall[-1]

        self._interp_recall_max_qps = qps
        self._interp_recall = interp1d(qps_dense, recall_dense, kind=INTERP_KIND, bounds_error=False, fill_value="extrapolate")
        return self._interp_recall(qps).item()


    @lru_cache(maxsize=None)
    def _f_qps(self, recall):
        _recall, _qps = self.__get_recall_qps_for_score()
        if recall < _recall[0]:
            return _qps[0]
        if self._interp_qps:
            return self._interp_qps(recall).item()
        f_qps_raw = interp1d(_recall, _qps, kind=INTERP_KIND, bounds_error=False, fill_value="extrapolate")

        recall_dense = np.linspace(_recall[0], 1.0,  int((1.0 - _recall[0]) / 1e-5))
        qps_dense = f_qps_raw(recall_dense)

        # Normalize QPS for gradient analysis
        qps_norm = qps_dense / (np.max(qps_dense) + 1e-8)
        dy = np.gradient(qps_norm, recall_dense)

        #! REMARK
        THRESHOLD = 1e-3
        flat_mask = np.abs(dy) < THRESHOLD
        if np.any(flat_mask):
            flat_index = np.argmax(flat_mask)
        else:
            flat_index = len(qps_norm)

        qps_dense[flat_index:] = 0.0

        self._interp_qps = interp1d(recall_dense, qps_dense, kind=INTERP_KIND, bounds_error=False, fill_value="extrapolate")
        return self._interp_qps(recall).item()


    @lru_cache(maxsize=None)
    def __score_harmony(self):
        #* Contraints of the tuning problem
        recall_min = HnswConfig.recall_min
        qps_min = HnswConfig.qps_min
        try:
            recall_qps_min = self._f_recall(qps_min)
        except ValueError:
            print(f"Case0: M: {self.M}, efC: {self.efC}")
            return 0

        a, b = min(recall_min,recall_qps_min), max(recall_min, recall_qps_min)
        score, _ = quad(lambda x: (self._f_qps(x) - qps_min) / max(self.__get_recall_qps_for_score()[1]), a, b)
        if score == 0:
            print(f"a: {a}, b: {b}")
            print(f"Case1: M: {self.M}, efC: {self.efC}")
            return 0
        if score < 0:
            print(f"Case2: M: {self.M}, efC: {self.efC}")
            print(f"qps_min: {qps_min}")
            print(f"recall(qps_min): {self._f_recall(qps_min)}")
            print(f"qps(recall(qps_min)): {self._f_qps(self._f_recall(qps_min))}")
            print(f"a: {a}, b: {b}")
            print(f"{self._f_qps(a) - qps_min}, {self._f_qps(b) - qps_min}")
            print("================")
            return 0
        return max(score, 0)

    def __score_max_recall(self):
        try:
            score = self._f_recall(HnswConfig.qps_min)
        except ValueError:
            score = 0
        return max(score, 0)

    def __score_max_qps(self):
        score = self._f_qps(HnswConfig.recall_min)
        return max(score, 0)

    @lru_cache(maxsize=None)
    def score(self, _recall_min, _qps_min):
        harmony_score = self.__score_harmony()
        max_recall_score = self.__score_max_recall()
        max_qps_score = self.__score_max_qps()
        #* Clean up the Index
        self._clean_up()
        return (harmony_score, max_recall_score, max_qps_score)

    def to_score_csv(self):
        _score = self.score(HnswConfig.recall_min, HnswConfig.qps_min)   # Force the score calculation
        mean_recall = np.mean([result.recall for result in self.results.values()])
        mean_qps = np.mean([result.qps for result in self.results.values()])
        return [self.impl, self.dataset.name, self.dataset.recompute, self.batch, self.M, self.efC, self._efS,\
                self.index_size, self.build_time, self.test_time, [round(s.search_time, 2) for s in self.results.values()],\
                HnswConfig.recall_min, HnswConfig.qps_min,\
                round(_score[0], 4),
                round(_score[1], 4),
                round(_score[2], 4)]

    def to_recall_qps_csv(self):
        _ = self.score(HnswConfig.recall_min, HnswConfig.qps_min)        # Force the score calculation
        return [[self.impl, self.dataset.name, self.dataset.recompute, self.batch, self.M, self.efC, efS, self.index_size, self.build_time, self.results[efS].test_time,
                    [round(s, 6) for s in self.results[efS].search_times],
                    [round(r, 5) for r, q in self.results[efS]._recall_qps],
                    [round(q, 6) for r, q in self.results[efS]._recall_qps]]
                    for efS in self._efS]

    def _clean_up(self):
        if self._index:
            del self._index
            self._index = None
        gc.collect()

    def __str__(self):
        return f"{self.impl}_{self.dataset.name}_{self.M}_{self.efC}"


    @classmethod
    def from_csv(cls, filename:str, dir:str=None):
        import ast
        hnsw_configs = dict()  # (M, efC) -> HnswConfig
        file = f"{dir}/{filename}" if dir else filename
        with open(file, mode='r') as f:
            csv_reader = csv.reader(f)
            header = next(csv_reader)
            for row in csv_reader:
                # START OF PARSING
                impl = str(row[0])
                dataset_name = str(row[1])
                recompute = bool(row[2])
                batch = bool(row[3])
                M = int(row[4])
                efC = int(row[5])
                efS = ast.literal_eval(row[6])
                index_size = float(row[7])
                build_time = float(row[8])
                test_time = float(row[9])
                search_times = ast.literal_eval(row[10])
                recall = ast.literal_eval(row[11])
                qps = ast.literal_eval(row[12])
                recall_qps = list(zip(recall, qps))
                dataset = dataset_mapping[dataset_name](recompute=recompute, impl=impl)
                hnsw_result = HnswResult(M, efC, efS, build_time, index_size, test_time, search_times, recall_qps)
                if (M, efC) not in hnsw_configs:
                    hnsw_configs[(M, efC)] = HnswConfig(dataset, M, efC, batch=batch, efS=[], build_time=build_time, index_size=index_size)
                hnsw_configs[(M, efC)]._efS.append(efS)
                hnsw_configs[(M, efC)].results[efS] = hnsw_result
        return sorted(list(hnsw_configs.values()))


    def __lt__(self, other):
        return self.efC < other.efC or (self.efC == other.efC and self.M < other.M)


    def __str__(self):
        return f"{self.impl}_{self.dataset.name}_{self.M}_{self.efC}_{self._efS}_{self.batch}"


def save_results_to_csv(results, prefix="", only_score=False):
    _suffix = time.strftime("%m%d_%H%M%S")
    dirname = f"{RESULTS_DIR}/results-{time.strftime('%m%d')}"
    filename = results[0].dataset.name
    score_filename = f"{prefix}_SCORE_{filename}_{_suffix}.csv"
    recall_qps_filename = f"{prefix}_RQ_{filename}_{_suffix}.csv"

    os.makedirs(dirname, exist_ok=True)

    with open(f"{dirname}/{score_filename}", mode='w', newline='') as score_f, \
        open(f"{dirname}/{recall_qps_filename}", mode='w', newline='') as recall_qps_f:
        score_writer = csv.writer(score_f)
        recall_qps_writer = csv.writer(recall_qps_f)

        score_writer.writerow(["Implementation", "Dataset", "recompute", "batch", "M", "efC", "efS",\
                               "index_size", "build_time", "test_time", "search_times",\
                                "recall_min", "qps_min", "score_harmony", "score_max_recall", "max_max_qps"])
        recall_qps_writer.writerow(["Implementation", "Dataset", "recompute", "batch", "M", "efC", "efS", "index_size", "build_time", \
            "test_time", "search_times", "Recall", "QPS"])
        score_f.flush(); recall_qps_f.flush()
        for result in results:
            score_writer.writerow(result.to_score_csv())
            for row in result.to_recall_qps_csv():
                recall_qps_writer.writerow(row)
            score_f.flush(); recall_qps_f.flush()
    return dirname, score_filename, recall_qps_filename
