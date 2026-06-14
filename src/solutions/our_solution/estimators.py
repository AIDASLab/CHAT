import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

DATA_SPECS = {
    "nytimes": {"N": 290_000, "d": 256, "encoding": "angular"},
    "glove":   {"N": 1_183_514, "d": 100, "encoding": "angular"},
    "sift":    {"N": 1_000_000, "d": 128, "encoding": "euclidean"},
    "youtube": {"N": 990_072, "d": 1024, "encoding": "angular"},
    "deep1M":  {"N": 1_000_000, "d": 256, "encoding": "angular"},
}

def get_data_spec(dataset: str):
    for key in DATA_SPECS:
        if key in dataset:
            spec = DATA_SPECS[key]
            return spec["N"], spec["d"]
    raise ValueError(f"Spec not found for dataset: {dataset}")

class BuildTimeEstimator:
    """Online model of HNSW *build time* as a function of (efC, M).

    After every ``update`` the estimator optionally smooths the raw target column
    and – once ≥5 samples are present – performs a non-linear curve-fit to
    determine parameters (alpha, beta, g₀, g₁, delta).

    A conservative *margin* is subtracted on every prediction so that the model
    errs on the side of **under-estimating** the true build time; this avoids
    discarding promising hyper-parameter points too early.
    """

    def __init__(self, threshold, *, smooth: bool = False, sigma: float = 10.0,
                 margin: float = 0.0) -> None:
        if smooth and sigma <= 0:
            raise ValueError("`sigma` must be positive when smoothing is enabled.")
        self.smooth = smooth
        self.sigma = sigma
        self.margin = margin
        self.threshold = threshold

        self._update_count = 0
        self.df: pd.DataFrame = pd.DataFrame(columns=["efC", "M", "BuildTime"])
        self.params: Optional[np.ndarray] = None  # alpha, beta, g₀, g₁, delta

    # ---------------------------------------------------------------------
    # data ingestion & fitting
    # ---------------------------------------------------------------------
    def update(self, efC: float, M: float, build_time: float) -> None:
        """Append a new measurement and refit if enough data is present."""
        self.df.loc[len(self.df)] = {"efC": efC, "M": M, "BuildTime": build_time}

        # Optional temporal smoothing (simple 1-D Gaussian along append order)
        if self.smooth and len(self.df) > 1:
            self.df["BuildTime"] = gaussian_filter(
                self.df["BuildTime"].to_numpy(dtype=float), sigma=self.sigma
            )
        if len(self.df) == 5 or len(self.df) % 8 == 0:
            if len(self.df) != 0:
                self._fit_model()
        self._update_count += 1
        if build_time > self.threshold:
            print(f"{build_time} > {self.threshold} @@@@@@@@@@@@@@@@@@@@@@")
        return build_time <= self.threshold

    # ------------------------------------------------------------------
    # prediction helpers
    # ------------------------------------------------------------------
    def estimate(self, efC: float, M: float) -> float:
        """Return *conservative* prediction (margin already subtracted)."""
        if self.params is None:
            return float("nan")
        alpha, beta, g0, g1, delta = self.params
        ln_m = np.log2(M)
        pred = alpha + beta * efC + (g0 + g1 * efC) * ln_m + delta * efC * ln_m ** 2
        return float(pred - self.margin)

    def _predict_bulk(self, points: np.ndarray) -> np.ndarray:
        if self.params is None:
            return np.full(points.shape[0], np.nan)
        alpha, beta, g0, g1, delta = self.params
        efC, M = points[:, 0], points[:, 1]
        return (
            self._formula_build_time(efC, M, alpha, beta, g0, g1, delta) - self.margin
        )

    # ------------------------------------------------------------------
    # binary decision
    # ------------------------------------------------------------------
    def binary_classification(self, efC: float, M: float, *, safe_side: str = "below") -> bool:
        """True ⟹ prediction is inside *safe* region w.r.t. ``threshold``."""
        if self._update_count <= 4:    
            return True
        pred = self.estimate(efC, M)
        print(f"pred : {pred} vs thres : {self.threshold}")
        if safe_side == "below":
            return pred <= self.threshold
        elif safe_side == "above":
            return pred >= self.threshold
        raise ValueError("safe_side must be 'below' or 'above'.")

    # ------------------------------------------------------------------
    # internal model
    # ------------------------------------------------------------------
    @staticmethod
    def _formula_build_time(efC: np.ndarray, M: np.ndarray,
                             alpha: float, beta: float, g0: float, g1: float, delta: float) -> np.ndarray:
        ln_m = np.log2(M)
        return alpha + beta * efC + (g0 + g1 * efC) * ln_m + delta * efC * ln_m ** 2

    def _formula_wrapper(self, xdata: Tuple[np.ndarray, np.ndarray],
                         alpha: float, beta: float, g0: float, g1: float, delta: float) -> np.ndarray:
        return self._formula_build_time(xdata[0], xdata[1], alpha, beta, g0, g1, delta)

    def _fit_model(self) -> None:
        X = self.df[["efC", "M"]].to_numpy(dtype=float)
        y = self.df["BuildTime"].to_numpy(dtype=float)
        p0 = np.array([0.0, 1.0, 1.0, 0.0, 1e-3])  # initial guess
        try:
            opt, _ = curve_fit(self._formula_wrapper, (X[:, 0], X[:, 1]), y, p0=p0)
            self.params = opt
        except Exception as exc:
            print("Curve-fit failed:", exc)

    def plot_surface(self, thres: Optional[float] = None,
                     view_angles: Optional[List[Tuple[int, int]]] = None,
                     grid_points: int = 30) -> None:
        if self.df.empty:
            raise RuntimeError("No data to plot.")

        efC_lin = np.linspace(self.df["efC"].min(), self.df["efC"].max(), grid_points)
        M_lin   = np.linspace(self.df["M"].min(),  self.df["M"].max(),  grid_points)
        efC_grid, M_grid = np.meshgrid(efC_lin, M_lin)
        grid = np.column_stack([efC_grid.ravel(), M_grid.ravel()])
        z_pred = (self._predict_bulk(grid).reshape(efC_grid.shape)
                  if self.params is not None else np.full_like(efC_grid, np.nan))

        if view_angles is None:
            view_angles = [(30, 60), (30, 150), (30, 240), (30, 330)]

        fig, axes = plt.subplots(1, len(view_angles), figsize=(5 * len(view_angles), 4),
                                 subplot_kw={"projection": "3d"})
        axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]

        for ax, (elev, azim) in zip(axes, view_angles):
            ax.scatter(self.df["efC"], self.df["M"], self.df["BuildTime"],
                       c="k", s=20, depthshade=True)
            ax.plot_trisurf(efC_grid.ravel(), M_grid.ravel(), z_pred.ravel(),
                            cmap="viridis", alpha=0.5, edgecolor="none")
            if thres is not None and self.params is not None:
                ax.contour(efC_grid, M_grid, z_pred, levels=[thres], colors="red",
                           linewidths=1, linestyles="-", offset=thres, zdir="z")
            ax.set_xlabel("efC"); ax.set_ylabel("M"); ax.set_zlabel("BuildTime")
            ax.view_init(elev=elev, azim=azim)

        fig.suptitle("Build-Time model surface (fitted)", y=1.02)
        plt.tight_layout(); plt.show()

    # ───────────────────────── residual plot ─────────────────────────────
    def plot_residuals(self, view: Tuple[int, int] = (30, 135)) -> None:
        if self.params is None:
            raise RuntimeError("Model must be fitted first.")
        if len(self.df["efC"].unique()) < 2 or len(self.df["M"].unique()) < 2:
            raise RuntimeError("Need at least 2×2 mesh of (efC, M) points.")

        efc_vals = sorted(self.df["efC"].unique())
        m_vals   = sorted(self.df["M"].unique())
        efc_mesh, m_mesh = np.meshgrid(efc_vals, m_vals)

        truth = (self.df.pivot_table(index="M", columns="efC", values="BuildTime", aggfunc="mean")
                 .reindex(index=m_vals, columns=efc_vals).to_numpy())
        pred = self._predict_bulk(np.column_stack([efc_mesh.ravel(), m_mesh.ravel()])).reshape(m_mesh.shape)
        resid = pred - truth

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(efc_mesh, m_mesh, resid, cmap="coolwarm", linewidth=0.3,
                               alpha=0.6, antialiased=False)
        ax.set(title="Residuals (prediction - truth)", xlabel="efC", ylabel="M", zlabel="Residual")
        ax.view_init(elev=view[0], azim=view[1])
        fig.colorbar(surf, shrink=0.6, aspect=10, label="Residual")
        plt.tight_layout(); plt.show()
        
    def plot_surface_with_gt(
        self,
        df: pd.DataFrame,
        *,
        view_angles: Optional[List[Tuple[int, int]]] = None,
        grid_points: int = 30,
    ) -> None:
        """
        예측된 Build-Time 표면과 *외부* ground-truth 산점(efC, M, build_time)을
        한 Figure 에 동시에 표시한다.

        Parameters
        ----------
        df : pd.DataFrame
            컬럼 ``'efC'``, ``'M'``, ``'BuildTime'`` 가 존재해야 한다.
        view_angles : list[tuple[int,int]], optional
            여러 시점을 주면 subplot 으로 나란히 보여 준다. 기본은 네 각도.
        grid_points : int
            예측 표면을 만들 그리드 해상도.
        """
        if df.empty:
            raise ValueError("Provided dataframe is empty.")
        if self.params is None:
            raise RuntimeError("Model must be fitted first (≥5 samples & update()).")

        # ─── 예측 표면 만들기 ────────────────────────────────────────────────
        efC_lin = np.linspace(df["efC"].min(), df["efC"].max(), grid_points)
        M_lin   = np.linspace(df["M"].min(),  df["M"].max(),  grid_points)
        efC_grid, M_grid = np.meshgrid(efC_lin, M_lin)
        grid   = np.column_stack([efC_grid.ravel(), M_grid.ravel()])
        z_pred = self._predict_bulk(grid).reshape(efC_grid.shape)

        # ─── 시점 설정 ──────────────────────────────────────────────────────
        if view_angles is None:
            view_angles = [(30, 60), (30, 150), (30, 240), (30, 330)]

        fig, axes = plt.subplots(
            1, len(view_angles), figsize=(5 * len(view_angles), 4),
            subplot_kw={"projection": "3d"}
        )
        axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]

        for ax, (elev, azim) in zip(axes, view_angles):
            # (a) ground-truth scatter
            sc = ax.scatter(
                df["efC"], df["M"], df["build_time"],
                c=df["build_time"], cmap="coolwarm", s=18, alpha=0.8
            )
            # (b) predicted surface
            ax.plot_trisurf(
                efC_grid.ravel(), M_grid.ravel(), z_pred.ravel(),
                cmap="viridis", alpha=0.45, edgecolor="none"
            )
            ax.set_xlabel("efC"); ax.set_ylabel("M"); ax.set_zlabel("BuildTime")
            ax.view_init(elev=elev, azim=azim)

        fig.colorbar(sc, ax=axes, shrink=0.6, pad=0.02, label="Ground-truth Build-Time (s)")
        fig.suptitle("Predicted Build-Time surface  vs  ground-truth samples", y=1.02)
        plt.tight_layout(); plt.show()


class IndexSizeEstimator:
    """Online model of HNSW *index size* as a function of (efC, M)."""

    def __init__(self, *, N: int, d: int, threshold, smooth: bool = False, sigma: float = 10.0,
                 margin: float = 0.0) -> None:
        if smooth and sigma <= 0:
            raise ValueError("`sigma` must be positive when smoothing is enabled.")
        self.N, self.d = N, d
        self.smooth = smooth
        self.sigma = sigma
        self.margin = margin
        self.threshold = threshold

        self._update_count = 0
        self._is_inited = lambda : self._update_count > 4
        self.df: pd.DataFrame = pd.DataFrame(columns=["efC", "M", "IndexSize"])
        self.params: Optional[np.ndarray] = None  # alpha, overhead
    # ------------------------------------------------------------------
    def update(self, efC: float, M: float, index_size: float) -> None:
        self.df.loc[len(self.df)] = {"efC": efC, "M": M, "IndexSize": index_size}

        if self.smooth and len(self.df) > 1:
            self.df["IndexSize"] = gaussian_filter(
                self.df["IndexSize"].to_numpy(dtype=float), sigma=self.sigma
            )
        if len(self.df) == 5 or len(self.df) % 8 == 0:
            if len(self.df) != 0:
                self._fit_model()
        self._update_count += 1
        if index_size > self.threshold:
            print(f"{index_size} > {self.threshold} @@@@@@@@@@@@@@@@@@@@@@@")
        return index_size <= self.threshold

    # ------------------------------------------------------------------
    def estimate(self, efC: float, M: float) -> float:
        if self.params is None:
            return float("nan")
        alpha, overhead = self.params
        pred = self._formula_index_size(efC, M, alpha, overhead)
        return float(pred - self.margin)

    def _predict_bulk(self, pts: np.ndarray) -> np.ndarray:
        if self.params is None:
            return np.full(pts.shape[0], np.nan)
        alpha, overhead = self.params
        efC, M = pts[:, 0], pts[:, 1]
        return self._formula_index_size(efC, M, alpha, overhead) - self.margin

    # ------------------------------------------------------------------
    def binary_classification(self, efC: float, M: float, *, safe_side: str = "below") -> bool:
        pred = self.estimate(efC, M)
        if self._update_count <= 4:    
            return True
        if safe_side == "below":
            return pred <= self.threshold
        elif safe_side == "above":
            return pred >= self.threshold
        raise ValueError("safe_side must be 'below' or 'above'.")

    # ------------------------------------------------------------------
    def _formula_index_size(self, efC: np.ndarray, M: np.ndarray,
                             alpha: float, overhead: float) -> np.ndarray:
        # efC currently unused – kept for signature symmetry
        N, d, e = self.N, self.d, 2
        base = N * d * 4 / 1e6
        variable = overhead * (N * M * 8) * e / 1e6
        return base + variable + alpha

    def _formula_wrapper(self, xdata: Tuple[np.ndarray, np.ndarray],
                         alpha: float, overhead: float) -> np.ndarray:
        return self._formula_index_size(xdata[0], xdata[1], alpha, overhead)

    def _fit_model(self) -> None:
        X = self.df[["efC", "M"]].to_numpy(dtype=float)
        y = self.df["IndexSize"].to_numpy(dtype=float)
        p0 = np.array([0.0, 1.0])
        try:
            opt, _ = curve_fit(self._formula_wrapper, (X[:, 0], X[:, 1]), y, p0=p0)
            self.params = opt
        except Exception as exc:
            print("Curve-fit failed:", exc)

    # ──────────────────────────── surface plot ───────────────────────────
    def plot_surface(
        self,
        thres: Optional[float] = None,
        view_angles: Optional[List[Tuple[int, int]]] = None,
        grid_points: int = 30,
    ) -> None:
        """3-D surface of fitted index-size model with optional threshold contour."""
        if self.df.empty:
            raise RuntimeError("No data to plot.")

        efC_lin = np.linspace(self.df["efC"].min(), self.df["efC"].max(), grid_points)
        M_lin   = np.linspace(self.df["M"].min(),  self.df["M"].max(),  grid_points)
        efC_grid, M_grid = np.meshgrid(efC_lin, M_lin)
        grid = np.column_stack([efC_grid.ravel(), M_grid.ravel()])
        z_pred = (
            self._predict_bulk(grid).reshape(efC_grid.shape)
            if self.params is not None
            else np.full_like(efC_grid, np.nan)
        )

        if view_angles is None:
            view_angles = [(30, 60), (30, 150), (30, 240), (30, 330)]

        fig, axes = plt.subplots(
            1, len(view_angles), figsize=(5 * len(view_angles), 4),
            subplot_kw={"projection": "3d"}
        )
        axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]

        for ax, (elev, azim) in zip(axes, view_angles):
            ax.scatter(self.df["efC"], self.df["M"], self.df["IndexSize"],
                       c="k", s=20, depthshade=True)
            ax.plot_trisurf(
                efC_grid.ravel(), M_grid.ravel(), z_pred.ravel(),
                cmap="viridis", alpha=0.5, edgecolor="none"
            )
            if thres is not None and self.params is not None:
                ax.contour(
                    efC_grid, M_grid, z_pred, levels=[thres],
                    colors="red", linewidths=1, linestyles="-",
                    offset=thres, zdir="z"
                )
            ax.set_xlabel("efC"); ax.set_ylabel("M"); ax.set_zlabel("IndexSize")
            ax.view_init(elev=elev, azim=azim)

        fig.suptitle("Index-Size model surface (fitted)", y=1.02)
        plt.tight_layout(); plt.show()

    # ───────────────────────── residual plot ─────────────────────────────
    def plot_residuals(self, view: Tuple[int, int] = (30, 135)) -> None:
        """3-D surface of residuals (prediction − truth)."""
        if self.params is None:
            raise RuntimeError("Model must be fitted first.")
        if len(self.df["efC"].unique()) < 2 or len(self.df["M"].unique()) < 2:
            raise RuntimeError("Need at least a 2×2 grid of distinct (efC, M).")

        efc_vals = sorted(self.df["efC"].unique())
        m_vals   = sorted(self.df["M"].unique())
        efc_mesh, m_mesh = np.meshgrid(efc_vals, m_vals)

        truth = (
            self.df.pivot_table(
                index="M", columns="efC", values="IndexSize", aggfunc="mean"
            )
            .reindex(index=m_vals, columns=efc_vals)
            .to_numpy()
        )
        pred = self._predict_bulk(
            np.column_stack([efc_mesh.ravel(), m_mesh.ravel()])
        ).reshape(m_mesh.shape)
        resid = pred - truth

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(
            efc_mesh, m_mesh, resid, cmap="coolwarm",
            linewidth=0.3, alpha=0.6, antialiased=False
        )
        ax.set_title("Residuals (prediction − truth)")
        ax.set_xlabel("efC"); ax.set_ylabel("M"); ax.set_zlabel("Residual")
        ax.view_init(elev=view[0], azim=view[1])
        fig.colorbar(surf, shrink=0.6, aspect=10, label="Residual")
        plt.tight_layout(); plt.show()

    def plot_surface_with_gt(
        self,
        df: pd.DataFrame,
        *,
        view_angles: Optional[List[Tuple[int, int]]] = None,
        grid_points: int = 30,
    ) -> None:
        """
        예측된 Index-Size 표면과 *외부* ground-truth 산점(efC, M, index_size)을
        한 Figure 에 동시에 표시한다.

        Parameters
        ----------
        df : pd.DataFrame
            컬럼 ``'efC'``, ``'M'``, ``'IndexSize'`` 가 존재해야 한다.
        """
        if df.empty:
            raise ValueError("Provided dataframe is empty.")
        if self.params is None:
            raise RuntimeError("Model must be fitted first (≥5 samples & update()).")

        efC_lin = np.linspace(df["efC"].min(), df["efC"].max(), grid_points)
        M_lin   = np.linspace(df["M"].min(),  df["M"].max(),  grid_points)
        efC_grid, M_grid = np.meshgrid(efC_lin, M_lin)
        grid   = np.column_stack([efC_grid.ravel(), M_grid.ravel()])
        z_pred = self._predict_bulk(grid).reshape(efC_grid.shape)

        if view_angles is None:
            view_angles = [(30, 60), (30, 150), (30, 240), (30, 330)]

        fig, axes = plt.subplots(
            1, len(view_angles), figsize=(5 * len(view_angles), 4),
            subplot_kw={"projection": "3d"}
        )
        axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]

        for ax, (elev, azim) in zip(axes, view_angles):
            sc = ax.scatter(
                df["efC"], df["M"], df["index_size"],
                c=df["index_size"], cmap="magma", s=18, alpha=0.8
            )
            ax.plot_trisurf(
                efC_grid.ravel(), M_grid.ravel(), z_pred.ravel(),
                cmap="viridis", alpha=0.45, edgecolor="none"
            )
            ax.set_xlabel("efC"); ax.set_ylabel("M"); ax.set_zlabel("IndexSize")
            ax.view_init(elev=elev, azim=azim)

        fig.colorbar(sc, ax=axes, shrink=0.6, pad=0.02, label="Ground-truth Index-Size (bytes)")
        fig.suptitle("Predicted Index-Size surface  vs  ground-truth samples", y=1.02)
        plt.tight_layout(); plt.show()
