import time
import numpy as np
import psutil

from .hnsw_config import HnswConfig
from auto_tuner.models.hnsw_result import HnswResult
from auto_tuner.constants import MAX_THREADS
from auto_tuner.utils import log_execution

import faiss


class HnswConfigFaiss(HnswConfig):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.impl = "faiss"
    
    
    @log_execution() 
    def _build(self):
        _train_data = self.dataset.get_database()
        faiss.omp_set_num_threads(MAX_THREADS)
        if self.dataset.metric == "l2":
            self._index = faiss.IndexHNSWFlat(self.dataset.dim, self.M)
        else:
            self._index = faiss.IndexHNSWFlat(self.dataset.dim, self.M, faiss.METRIC_INNER_PRODUCT)
        self._index.hnsw.efConstruction = self.efC
        self._index.verbose = True
        if _train_data.dtype != np.float32:
            _train_data = _train_data.astype(np.float32)
        if self.dataset.metric == "cosine":
            faiss.normalize_L2(_train_data)
        print("\nSTART OF BUILD ...\n")
        mem_before = []
        mem_after = []
        for i in range(5):
            mem_before.append(psutil.Process().memory_info().rss // 1024)
            time.sleep(0.3)
        _start = time.time()
        self._index.add(_train_data)
        self.build_time = time.time() - _start
        for i in range(5):
            mem_after.append(psutil.Process().memory_info().rss // 1024)
            time.sleep(0.3)
        mem_before.sort(); mem_after.sort()
        mem_before = mem_before[1:-1]
        mem_after = mem_after[1:-1]
        self.index_size = int(np.mean(mem_after) - np.mean(mem_before))
        print("\nEND OF BUILD !!!\n")
        if not self.batch:
            faiss.omp_set_num_threads(1)
        return self.build_time
    
    
    @log_execution()
    def _evaluate(self, efS):
        _test_time = time.time()
        _test_data = self.dataset.get_queries()
        if _test_data.dtype != np.float32:
            _test_data = _test_data.astype(np.float32)
        if self.dataset.metric == "cosine":
            faiss.normalize_L2(_test_data)
        faiss.cvar.hnsw_stats.reset()
        self._index.hnsw.efSearch = efS
        self._index.hnsw.search_bounded_queue = False
        recall_qps = []
        search_times = []
        for _ in range(self.iters):
            print(f"Running iteration {_} ...")
            _search_time = time.time()
            # _test_data = np.random.rand(1, self.dataset.dim).astype(np.float32)
            _D, _labels = self._index.search(_test_data, self.dataset.k)
            _search_time = (time.time() - _search_time)
            _qps = len(_test_data) / _search_time
            labels = []
            for i in range(len(_D)):
                r = []
                for l, d in zip(_labels[i], _D[i]):
                    r.append(-1 if l == -1 else l)
                labels.append(r)
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

