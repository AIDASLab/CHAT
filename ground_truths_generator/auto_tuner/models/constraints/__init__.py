class Constraints:

    def __init__(self, TaskType, recall_min=0.90, qps_min=1000, index_max=1e12, build_time=24 * 3600, tuning_time = 7 * 24 * 3600):
        self.TaskType = TaskType
        self.recall_min = recall_min
        self.qps_min = qps_min
        self.index_max = index_max
        self.build_time = build_time
        self.tuning_time = tuning_time

    def __str__(self):
        return f"TaskType: {self.TaskType}, recall_min: {self.recall_min}, qps_min: {self.qps_min}, index_max: {self.index_max}, build_time: {self.build_time}, tuning_time: {self.tuning_time}"
