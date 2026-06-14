from src.solutions.our_solution.estimators import BuildTimeEstimator, IndexSizeEstimator

class MockBuildTimeEstimator(BuildTimeEstimator):
    def __init__(self, threshold):
        super().__init__(threshold=threshold)

    def binary_classification(self, efC: float, M: float, *, safe_side: str = "below") -> bool:
        return True

class MockIndexSizeEstimator(IndexSizeEstimator):
    def __init__(self,N,d, threshold):
        super().__init__(N=N,d=d, threshold=threshold)
    
    def binary_classification(self, efC: float, M: float, *, safe_side: str = "below") -> bool:
        return True