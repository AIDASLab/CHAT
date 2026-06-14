from auto_tuner.models.constraints import Constraints
from auto_tuner.models.hnsw_result import HnswResultFactory

class Solution:
    def __init__(self, constraints:Constraints, resultFactory:HnswResultFactory):
        self.constraints = constraints
        self.resultFactory = resultFactory
