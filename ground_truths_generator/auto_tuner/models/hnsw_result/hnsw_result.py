from typing import List, Tuple
import numpy as np

from auto_tuner.constants import DEFAULT_EFS
# For a efS
class HnswResult:
    
    def __init__(self, M:int, efC:int, efS:int, build_time:float, index_size:int, test_time:float, search_times:List[float], recall_qps:List[Tuple[float, float]],):
        self.M = M
        self.efC = efC
        self.efS = efS
        self.build_time = build_time
        self.index_size = index_size
        self.test_time = test_time
        self.search_times = search_times
        self._recall_qps = recall_qps
        self.smoothened_qps = None
    
    @property
    def search_time(self):
        return np.mean(self.search_times)
    
    @property
    def recall(self):
        return np.mean([r for r, _ in self._recall_qps])

    @property
    def qps(self):
        if self.smoothened_qps:
            return self.smoothened_qps
        return np.mean([q for _, q in self._recall_qps])    
    
    def __lt__(self, other):
        if self.M == other.M:
            if self.efC == other.efC:
                return self.efS < other.efS
            return self.efC < other.efC
        return self.M < other.M
    
    def __eq__(self, other):
        return self.M == other.M and self.efC == other.efC and self.efS == other.efS
    
