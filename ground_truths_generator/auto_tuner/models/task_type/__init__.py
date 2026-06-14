from enum import Enum

class TaskType(Enum):
    MAX_HARMONIC = "max_harmonic"
    MAX_RECALL = "max_recall"
    MAX_QPS = "max_qps"