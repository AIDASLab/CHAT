import os
import csv
from src.solutions import print_optimal_hyperparameters

class Stats:

    def __init__(self, tuning_budget:float, recall_min:float=None, qps_min:float=None):
        assert tuning_budget > 0, "Tuning budget must be greater than 0"
        assert (recall_min is None) != (qps_min is None), "Only one of recall_min or qps_min should be set."
        self.tuning_budget = tuning_budget
        self.recall_min = recall_min
        self.qps_min = qps_min
        self.get_perf = lambda hp_perf: hp_perf[1][2] if recall_min is not None else hp_perf[1][1]
        self.stats = {}
        self.results = ([], [])
    
    def exploration_phase(self, results:list):
        self.results[0].extend(results)
        self.stats["exploration_time"] = results[-1][1][0]
        self.stats["exploration_ratio"] = results[-1][1][0] / self.tuning_budget
        opt_hp, _ = print_optimal_hyperparameters(results, recall_min=self.recall_min, qps_min=self.qps_min)
        if opt_hp is None:
            self.stats["exploration_performance"] = 0.0
        else:
            self.stats["exploration_performance"] = self.get_perf(opt_hp)

    def exploitation_phase(self, results:list):
        assert self.stats.get("exploration_time") is not None, "Exploration phase must be completed before exploitation phase."
        self.results[1].extend(results)
        self.stats["exploitation_time"] = results[-1][1][0] - self.stats["exploration_time"]
        self.stats["exploitation_ratio"] = results[-1][1][0] / self.tuning_budget
        self.stats["tuning_time"] = self.stats["exploration_time"] + self.stats["exploitation_time"]
        self.stats["tuning_time_ratio"] = self.stats["tuning_time"] / self.tuning_budget
        opt_hp, _ = print_optimal_hyperparameters(results, recall_min=self.recall_min, qps_min=self.qps_min)
        if opt_hp is None:
            self.stats["optimal_performance"] = 0.0
        else:
            self.stats["optimal_performance"] = self.get_perf(opt_hp)
        if self.stats["exploration_performance"] < self.stats["optimal_performance"]:
            self.stats["exploitation_performance"] = self.stats["optimal_performance"]
        else:
            self.stats["exploitation_performance"] = self.stats["exploration_performance"]

    @staticmethod
    def compare_stats(stats:dict, other_stats:dict, heuristic_type:str, impl:str, dataset:str, recall_min:float=None, qps_min:float=None):
        assert (recall_min is None) != (qps_min is None), "Only one of recall_min or qps_min should be set."
        """
        Compare two stats dictionaries and return a summary of differences.
        """
        criteria = ["exploration_time", "exploitation_time", "tuning_time", "exploration_performance", "exploitation_performance", "optimal_performance"]
        results = {}
        for criterion in criteria:
            if stats.get(criterion) == 0.0 and other_stats.get(criterion) == 0.0:
                result = "NULL"
            elif other_stats.get(criterion) == 0.0:
                result = "N/A"
            else:
                result = stats.get(criterion) / other_stats.get(criterion) 
            results[f"{criterion}_ratio"] = result
        # save it as a csv
        filename = f"{impl}_{dataset}_stats_comparison.csv"
        path = os.path.join("results", "stats", heuristic_type)
        os.makedirs(path, exist_ok=True)
        file = os.path.join(path, filename)
        if not os.path.exists(file):
            with open(file, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["impl", "dataset", "recall_min", "qps_min"] + [header for header in results.keys()])
        with open(file, "a") as f:
            writer = csv.writer(f)
            writer.writerow([impl, dataset, recall_min, qps_min] + [value for _, value in results.items()])
