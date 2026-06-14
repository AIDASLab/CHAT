import copy
import os

import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from auto_tuner.constants import RESULTS_DIR
from auto_tuner.models.hnsw_config import HnswConfig
from auto_tuner.models.hnsw_result import HnswResult

class ResultProcessor:

    # def __init__(self, results:list[HnswResult]):
    def __init__(self, results:list[HnswConfig], filename:str=None, smoothen:bool=True):
        self.__results = sorted(results)
        self.__filtered_results = copy.copy(self.__results)
        self.__filename = filename if filename else time.strftime("%m%d_%H%M")
        self.__filter_args = [[], [], []]
        self.smoothend = smoothen
        if self.smoothend:
            self.__smoothen_results(is_static=True)

    def __dir_path(self):
        _suffix = self.__filename.split("_")[-2]
        dir_path = f"{RESULTS_DIR}/summary/summary-{_suffix}/{self.__filename}"
        os.makedirs(dir_path, exist_ok=True)
        return dir_path


    def __save_plot(self, fig, plot_name, filter_args=[[], [], []]):
        for arg_idx in range(3):
            arg = self.__filter_args[arg_idx] + [] if filter_args[arg_idx] is None else filter_args[arg_idx]
            if len(arg) > 0:
                arg = str(arg)
                plot_name = f"{plot_name}_{arg}"
        dir_path = self.__dir_path()
        _filename = f"{dir_path}/{plot_name}.svg"
        fig.savefig(_filename, format='svg')


    def __smoothen_results(self, is_static=True):
        sigma = 1
        _filtered_results = copy.copy(self.__filtered_results)
        unique_M = sorted(np.unique([x.M for x in self.__filtered_results]))
        unique_efC = sorted(np.unique([x.efC for x in self.__filtered_results]))
        for efS in _filtered_results[0]._efS:
            qps_matrix = np.zeros((len(unique_M), len(unique_efC)))
            for hnsw_result in _filtered_results:
                result_efS = hnsw_result.results[efS]
                qps_matrix[unique_M.index(result_efS.M), unique_efC.index(result_efS.efC)] = result_efS.qps
            qps_matrix = gaussian_filter(qps_matrix, sigma=sigma)
            for hnsw_result in _filtered_results:
                hnsw_result.results[efS].smoothened_qps = \
                qps_matrix[unique_M.index(hnsw_result.M), unique_efC.index(hnsw_result.efC)]
        if is_static:
            self.__filtered_results = _filtered_results
        return _filtered_results

    def __plot_3d(self, metric, params=None, Ms=None, efCs=None, is_statis=False):
        _filtered_results = self.filter_results(params, Ms, efCs, is_statis)

        ## metric getter ##
        if metric == "recall_qps_harmony":
            max_qps = max([max(result.qps) for result in _filtered_results])
            min_qps = min([min(result.qps) for result in _filtered_results])
            metric_getter = lambda x: (np.mean(x.recall) * (np.mean(x.qps) / (max_qps - min_qps)))
        elif metric == "recall":
            metric_getter = lambda x: np.mean(x.recall)
        elif "score_harmony" in metric:
            metric_getter = lambda x: (x.score(HnswConfig.recall_min, HnswConfig.qps_min)[0])
        elif "score_recall" in metric:
            metric_getter = lambda x: (x.score(HnswConfig.recall_min, HnswConfig.qps_min)[1])
        elif "score_qps" in metric:
            metric_getter = lambda x: (x.score(HnswConfig.recall_min, HnswConfig.qps_min)[2])
        elif "qps" in metric:
            __recall = float(metric.split("_")[-1]) / 100
            metric_getter = lambda x: x.qps_recall(__recall)
        elif metric == "index_size":
            metric_getter = lambda x: x.index_size
        elif metric == "build_time":
            metric_getter = lambda x: x.build_time
        else:
            raise ValueError(f"Unknown metric: {metric}")

        x = [result.efC for result in _filtered_results]
        y = [result.M for result in _filtered_results]
        z_raw = [metric_getter(result) for result in _filtered_results]
        z = [max(val, 0) for val in z_raw]

        x_unique = np.unique(x)
        y_unique = np.unique(y)

        X, Y = np.meshgrid(x_unique, y_unique)
        Z = np.full_like(X, np.nan, dtype=np.float64)

        for i in range(len(x)):
            xi = np.where(x_unique == x[i])[0][0]
            yi = np.where(y_unique == y[i])[0][0]
            if z[i] > 0:
                Z[yi, xi] = z[i]
            else:
                Z[yi, xi] = np.nan

        if np.isnan(Z).all():
            z_min, z_max = 0, 1
        else:
            z_valid = Z[~np.isnan(Z)]
            z_min = np.min(z_valid)
            z_max = np.max(z_valid)
            if z_min == z_max:
                delta = 1e-3 if z_min == 0 else abs(z_min * 0.05)
                z_min -= delta
                z_max += delta

        # Top-5 높은 z값 인덱스 뽑기
        z_with_idx = [(i, val) for i, val in enumerate(z) if val > z_min]
        z_top5 = sorted(z_with_idx, key=lambda t: t[1], reverse=True)[:5]

        views = [(30, -120), (30, -210), (30, -300), (30, 30)]

        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(20, 16))
        for idx, (elev, azim) in enumerate(views):
            ax = fig.add_subplot(2, 2, idx + 1, projection='3d')
            surf = ax.plot_surface(X, Y, Z, cmap='viridis', antialiased=True, rstride=1, cstride=1)

            for i in range(len(x)):
                zi = z[i]
                xi = x[i]
                yi = y[i]

                if zi > z_min:
                    ax.scatter(xi, yi, z_min, color="b", marker="o", alpha=0.5)
                    ax.plot([xi, xi], [yi, yi], [z_min, zi], "k--", alpha=0.5)
                elif zi == z_min:
                    ax.scatter(xi, yi, z_min, color="red", marker="x", alpha=0.8)

            # top-5 z값 위에 등수 표시

            for rank, (i, _) in enumerate(z_top5):
                ax.scatter(x[i], y[i], z[i], color="black", marker="o", s=50, zorder=10)
                ax.text(x[i], y[i], z[i] + 0.01 * (z_max - z_min), f"{rank+1}",
                        color="black", fontsize=12, ha='center', zorder=11)

            ax.set_zlim(z_min, z_max)
            ax.set_xlabel("efC")
            ax.set_ylabel("M")
            ax.set_zlabel(metric)
            ax.set_title(f"View {idx+1}: elev={elev}, azim={azim}")
            ax.view_init(elev, azim)

        fig.suptitle(f"{metric} per (M, efC) - Multiple Views", fontsize=18)
        self.__save_plot(fig, metric, [params, Ms, efCs])
        plt.close(fig)



    def filter_results(self, params=None, Ms=None, efCs=None, is_static=True):
        _filtered_results = copy.copy(self.__filtered_results)
        if params is not None:
            _filtered_results = filter(lambda x: (x.M, x.efC) in params, _filtered_results)
            if is_static : self.__filter_args[0] += params
        else:
            if Ms is not None:
                _filtered_results = filter(lambda x: x.M in Ms, _filtered_results)
                if is_static : self.__filter_args[1] += Ms
            if efCs is not None:
                _filtered_results = filter(lambda x: x.efC in efCs, _filtered_results)
                if is_static : self.__filter_args[2] += efCs
        if is_static:
            self.__filtered_results = list(_filtered_results)
        return list(_filtered_results)

    def plot_score(self, params=None, Ms=None, efCs=None, is_statis=False, \
                   scores=["harmony", "recall", "qps"]):
        if "harmony" in scores:
            self.__plot_3d(f"score_harmony_{HnswConfig.recall_min}_{HnswConfig.qps_min}", params, Ms, efCs, is_statis)
        if "recall" in scores:
            self.__plot_3d(f"score_recall_{HnswConfig.qps_min}", params, Ms, efCs, is_statis)
        if "qps" in scores:
            self.__plot_3d(f"score_qps_{HnswConfig.recall_min}", params, Ms, efCs, is_statis)
        _filtered_results = self.filter_results(params, Ms, efCs, is_statis)
        # hp_score_1 = sorted([(result.M, result.efC, result.score(HnswConfig.recall_min, HnswConfig.qps_min)[0]) for result in _filtered_results], key=lambda x: x[2], reverse=True)
        # hp_score_2 = sorted([(result.M, result.efC, result.score(HnswConfig.recall_min, HnswConfig.qps_min)[1]) for result in _filtered_results], key=lambda x: x[2], reverse=True)
        # hp_score_3 = sorted([(result.M, result.efC, result.score(HnswConfig.recall_min, HnswConfig.qps_min)[2]) for result in _filtered_results], key=lambda x: x[2], reverse=True)
        # print(f"Harmony Score ====")
        # for result in hp_score_1:
        #     print(f"M: {result[0]}, efC: {result[1]}, score: {result[2]}")
        # print(f"Recall Score ====")
        # for result in hp_score_2:
        #     print(f"M: {result[0]}, efC: {result[1]}, score: {result[2]}")
        # print(f"QPS Score ====")
        # for result in hp_score_3:
        #     print(f"M: {result[0]}, efC: {result[1]}, score: {result[2]}")
        # print(f"===================")
        print("END OF plot_score")

    def plot_recall(self, params=None, Ms=None, efCs=None, is_statis=False):
        return self.__plot_3d("recall", params, Ms, efCs, is_statis)

    def plot_qps(self, params=None, Ms=None, efCs=None, is_statis=False, recalls=[0.80, 0.90, 0.95, 0.99]):
        for recall in recalls:
            self.__plot_3d(f"qps_{int(recall*100)}", params, Ms, efCs, is_statis)

    def plot_build_time(self, params=None, Ms=None, efS=None, is_static=False):
        return self.__plot_3d("build_time", params, Ms, efS, is_static)

    def plot_index_size(self, params=None, Ms=None, efS=None, is_static=False):
        return self.__plot_3d("index_size", params, Ms, efS, is_static)

    def plot_recall_qps(self, params=None, Ms=None, efCs=None, is_static=False):
        _filtered_results = self.filter_results(params, Ms, efCs, is_static)
        fig = plt.figure(figsize=(16, 8))
        colors = ["red", "orange", "yellow", "green", "blue", "navy", "purple"]
        for i, result in enumerate(_filtered_results):
            color = colors[i % len(colors)]
            plt.plot(result.recall, result.qps,
                     label=f"M={result.M}, efC={result.efC}, recall={np.mean(result.recall):.4f}, qps={np.mean(result.qps):.2f}, score={result.score(HnswConfig.recall_min, HnswConfig.qps_min)[0]:.2f}",
                     marker="o", color=color)
        plt.axvline(x=HnswConfig.recall_min, color="gray", linestyle="--", label="Recall Min")
        plt.axhline(y=HnswConfig.qps_min, color="gray", linestyle="--", label="QPS Min")
        plt.xlabel("Recall")
        plt.ylabel("QPS")
        plt.title("Recall vs QPS")
        plt.legend()
        self.__save_plot(fig, "recall_qps", [params, Ms, efCs])
        plt.close(fig)

    def save_results(self, params=None, Ms=None, efCs=None, is_static=False):
        _filtered_results = self.filter_results(params, Ms, efCs, is_static)
        dir_path = self.__dir_path()
        _filename = f"{dir_path}/result_{HnswConfig.recall_min}_{HnswConfig.qps_min}_{self.__filename}.csv"
        with open(_filename, mode='w') as f:
            f.write("M,efC,index_size,build_time,recall_min,qps_min,max_qps,max_recall,max_harmony\n")
            for result in _filtered_results:
                f.write(f"{result.M},{result.efC},{result.index_size},{result.build_time},")
                f.write(f"{HnswConfig.recall_min},{HnswConfig.qps_min},")
                f.write(f"{result.score(HnswConfig.recall_min, HnswConfig.qps_min)[0]},")
                f.write(f"{result.score(HnswConfig.recall_min, HnswConfig.qps_min)[1]},")
                f.write(f"{result.score(HnswConfig.recall_min, HnswConfig.qps_min)[2]}\n")

