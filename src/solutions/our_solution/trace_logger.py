import os
from typing import List, Tuple, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation


class TraceLogger:
    def __init__(self, M_min: int, M_max: int, efC_min: int, efC_max: int):
        self.M_min = M_min
        self.M_max = M_max
        self.efC_min = efC_min
        self.efC_max = efC_max
        self.logs: List[Tuple[int, int, int, Any]] = []  # [(M, efC, efS, perf)]

    def add(self, M: int, efC: int, efS: int, perf: float):
        self.logs.append((M, efC, efS, perf))

    def plot(
        self,
        out_png: str = "trace_logger.png",
        title_left: str = "Surface: perf over (M, efC)",
        title_right: str = "Surface: efS over (M, efC)",
        dpi: int = 200,
    ) -> str:
        """
        Creates a 1x2 figure with 3D surfaces (triangulated):
          - Left:  x=M, y=efC, z=perf
          - Right: x=M, y=efC, z=efS
        Saves as PNG and returns the output path.
        """
        if not self.logs:
            raise ValueError("TraceLogger.logs is empty. Add some points before plotting.")

        # --- Unpack logs ---
        M = np.array([x[0] for x in self.logs], dtype=float)
        efC = np.array([x[1] for x in self.logs], dtype=float)
        efS = np.array([x[2] for x in self.logs], dtype=float)
        perf = np.array([x[3] for x in self.logs], dtype=float)

        # --- Basic sanity filtering (optional but safer) ---
        mask = (
            (M >= self.M_min) & (M <= self.M_max) &
            (efC >= self.efC_min) & (efC <= self.efC_max) &
            np.isfinite(efS) & np.isfinite(perf)
        )
        M, efC, efS, perf = M[mask], efC[mask], efS[mask], perf[mask]

        if len(M) < 3:
            raise ValueError("Need at least 3 valid points to form a surface (triangulation).")

        # --- Deduplicate identical (M, efC) points to avoid triangulation issues ---
        # If duplicates exist, keep the last occurrence.
        xy = np.column_stack([M, efC])
        _, unique_idx = np.unique(xy, axis=0, return_index=True)
        # unique_idx is first occurrence; keep last occurrence instead:
        # build mapping and take last indices
        key_to_last = {}
        for i, (mx, cx) in enumerate(xy):
            key_to_last[(mx, cx)] = i
        last_idx = np.array(sorted(key_to_last.values()), dtype=int)

        M, efC, efS, perf = M[last_idx], efC[last_idx], efS[last_idx], perf[last_idx]

        if len(M) < 3:
            raise ValueError("After deduplication, fewer than 3 points remain; cannot plot a surface.")

        # --- Triangulation in (M, efC) plane ---
        tri = Triangulation(M, efC)

        # --- Create figure with two 3D subplots side-by-side ---
        fig = plt.figure(figsize=(14, 5))
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")

        # Left surface: z = perf
        surf1 = ax1.plot_trisurf(tri, perf, linewidth=0.2, antialiased=True)
        ax1.set_title(title_left)
        ax1.set_xlabel("M")
        ax1.set_ylabel("efC")
        ax1.set_zlabel("perf")
        ax1.set_xlim(self.M_min, self.M_max)
        ax1.set_ylim(self.efC_min, self.efC_max)
        fig.colorbar(surf1, ax=ax1, shrink=0.7, pad=0.08)

        # Right surface: z = efS
        surf2 = ax2.plot_trisurf(tri, efS, linewidth=0.2, antialiased=True)
        ax2.set_title(title_right)
        ax2.set_xlabel("M")
        ax2.set_ylabel("efC")
        ax2.set_zlabel("efS")
        ax2.set_xlim(self.M_min, self.M_max)
        ax2.set_ylim(self.efC_min, self.efC_max)
        fig.colorbar(surf2, ax=ax2, shrink=0.7, pad=0.08)

        plt.tight_layout()

        out_dir = os.path.dirname(out_png)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return out_png
