import time

# import devhnswlib as hnswlib
import hnswlib
import psutil
import numpy as np

from auto_tuner.utils import log_execution
from auto_tuner.constants import MAX_THREADS

from .hnsw_config import HnswConfig
from auto_tuner.models.hnsw_result import HnswResult


class HnswConfigHnswlib(HnswConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.impl = "hnswlib"
    
    @log_execution() 
    def _build(self):
        _train_data = self.dataset.get_database()
        self._index = hnswlib.Index(space=self.dataset.metric, dim=self.dataset.dim)
        self._index.init_index(max_elements=len(_train_data), ef_construction=self.efC, M=self.M)
        self._index.set_num_threads(MAX_THREADS)
        mem_before = []
        mem_after = []
        for i in range(5):
            mem_before.append(psutil.Process().memory_info().rss // 1024)
            time.sleep(0.3)
        _start = time.time()
        self._index.add_items(_train_data)
        self.build_time = time.time() - _start
        for i in range(5):
            mem_after.append(psutil.Process().memory_info().rss // 1024)
            time.sleep(0.3)
        # remove min and max value in mem list
        mem_before.sort(); mem_after.sort()
        mem_before = mem_before[1:-1]
        mem_after = mem_after[1:-1]
        self.index_size = int(np.mean(mem_after) - np.mean(mem_before))
        if not self.batch:
            self._index.set_num_threads(1)
        return self.build_time
    
    
    @log_execution()
    def _evaluate(self, efS):
        _test_time = time.time()
        _test_data = self.dataset.get_queries()
        self._index.set_ef(efS)
        recall_qps = []
        search_times = []
        for _ in range(self.iters):
            _search_time = time.time()
            labels, _distances = self._index.knn_query(_test_data, self.dataset.k)
            _search_time = (time.time() - _search_time)
            _qps = len(_test_data) / _search_time
            
            gt = self.dataset.get_groundtruth()
            correct = 0
            for i in range(len(_test_data)):
                for label in labels[i]:
                    for correct_label in gt[i]:
                        if label == correct_label:
                            correct += 1
                            break
            _recall = float(correct) / (self.dataset.k * len(_test_data))
            recall_qps.append((_recall, _qps))
            search_times.append(_search_time)
        _test_time = time.time() - _test_time
        hnsw_result = HnswResult(self.M, self.efC, efS, self.build_time, self.index_size, _test_time, search_times, recall_qps)
        self.results[efS] = hnsw_result
        print(f"efS: {efS}, recall: {hnsw_result.recall}, qps: {hnsw_result.qps}")
        return hnsw_result.recall, hnsw_result.qps
        