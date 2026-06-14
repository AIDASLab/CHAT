from src.constants import EFC_MAX, EFC_MIN, EFS_MAX, EFS_MIN

def get_max_perf(results, M, recall_min=None, qps_min=None):
    assert (recall_min is None) != (qps_min is None), "Only one of recall_min or qps_min should be set."
    if recall_min:
        return max(
            (perf[2] for (m, *_), perf in results if m == M and perf[1] >= recall_min),
            default=0.0
        )
    return max( # if qps_min
        (perf[1] for (m, *_), perf in results if m == M and perf[2] >= qps_min),
        default=0.0
    )

class EfCGetter:
    def __init__(self):
        self.__data = {}
        self.__ratio_candidates = 0.0
        self.__called_count = 0
        
    def put(self, M, efC_min, efC_max):
        if M in self.__data:
            existing_left, existing_right = self.__data[M]
            efC_min = max(efC_min, existing_left)
            efC_max = min(efC_max, existing_right)
        self.__data[M] = (efC_min, efC_max)
    
    def get(self, M) -> tuple[int, int]:
        self.__called_count += 1
        if M in self.__data:
            return self.__data[M]
        sorted_keys = sorted(self.__data.keys())
        lower_M = [m for m in sorted_keys if m < M]
        upper_M = [m for m in sorted_keys if m > M]
        efC_min, efC_max = EFC_MIN, EFC_MAX
        if lower_M:
            closest_lower_M = max(lower_M)
            efC_max = min(efC_max, self.__data[closest_lower_M][1])
        if upper_M:
            closest_upper_M = min(upper_M)
            efC_min = max(efC_min, self.__data[closest_upper_M][0])
        self.__ratio_candidates += (efC_max - efC_min) / (EFC_MAX - EFC_MIN)
        return efC_min, efC_max

    def clear(self):
        self.__data.clear()

    def __contains__(self, M):
        return M in self.__data
    
    def stats(self):
        return self.__ratio_candidates / self.__called_count if self.__called_count > 0 else 1.0


class EfCGetterBase:
    def __init__(self):
        self.__data = {}
        
    def put(self, M, efC_min, efC_max):
        pass

    def get(self, M) -> tuple[int, int]:
        return EFC_MIN, EFC_MAX

    def clear(self):
        self.__data.clear()

    def __contains__(self, M):
        return M in self.__data

    def stats(self):
        return 1.0

class EfSGetter:
    """
    Manages and infers the search range of 'efSearch' (efS) for a given efC.
    This class is specific to a single M value.

    This version combines two heuristics for efficient and aggressive search
    space reduction without sacrificing performance:
    1.  Inference from the single closest neighbor.
    2.  Linear extrapolation using the two closest neighbors.
    """
    def __init__(self):
        # Stores known (efC, efS) pairs.
        self.__data = {}
        # Stores analytics on how much the search space was shrunk.
        self.__shrinked_degrees = []

    def put(self, efC: int, efS: int):
        """Stores the observed optimal efS for a given efC."""
        self.__data[efC] = efS

    def get(self, efC: int) -> tuple[int, int]:
        if efC in self.__data:
            known_efS = self.__data[efC]
            return known_efS, known_efS

        if len(self.__data) < 2:
            return EFS_MIN, EFS_MAX

        sorted_keys = sorted(self.__data.keys())
        lower_efCs = [k for k in sorted_keys if k < efC]
        upper_efCs = [k for k in sorted_keys if k > efC]

        efS_min, efS_max = EFS_MIN, EFS_MAX
        if lower_efCs:
            efS_min = max(efS_min, self.__data[lower_efCs[-1]])
        if upper_efCs:
            efS_max = min(efS_max, self.__data[upper_efCs[0]])
        
        initial_min, initial_max = efS_min, efS_max
        if len(lower_efCs) >= 2:
            efc1, efc2 = lower_efCs[-2], lower_efCs[-1]
            efs1, efs2 = self.__data[efc1], self.__data[efc2]
            
            if efc2 > efc1:
                slope = (efs2 - efs1) / (efc2 - efc1)
                extrapolated_min = efs2 + slope * (efC - efc2)
                efS_min = max(efS_min, int(extrapolated_min))

        if len(upper_efCs) >= 2:
            efc2, efc3 = upper_efCs[0], upper_efCs[1]
            efs2, efs3 = self.__data[efc2], self.__data[efc3]

            if efc3 > efc2:
                slope = (efs3 - efs2) / (efc3 - efc2)
                extrapolated_max = efs2 - slope * (efc2 - efC)
                efS_max = min(efS_max, int(extrapolated_max))

        if efS_min > efS_max:
            efS_min, efS_max = initial_min, initial_max
        if (initial_max - initial_min) > 0:
            size_shrink = (initial_max - initial_min) - (efS_max - efS_min)
            if size_shrink > 0:
                ratio_shrink = size_shrink / (initial_max - initial_min)
                self.__shrinked_degrees.append((size_shrink, ratio_shrink))
        return efS_min, efS_max

    def clear(self):
        """Clears all stored data for a new run."""
        self.__data.clear()
        self.__shrinked_degrees.clear()

    def __contains__(self, efC: int) -> bool:
        """Allows using the 'in' operator (e.g., `if efC in efs_getter:`)."""
        return efC in self.__data
    
    def get_shrinked_degree_stats(self):
        """Calculates and returns statistics on search space reduction."""
        if not self.__shrinked_degrees:
            return {"avg_size_shrink": 0.0, "avg_ratio_shrink": 0.0}
        total_size_shrink = sum(s for s, r in self.__shrinked_degrees)
        total_ratio_shrink = sum(r for s, r in self.__shrinked_degrees)
        count = len(self.__shrinked_degrees)
        return {
            "avg_size_shrink": total_size_shrink / count,
            "avg_ratio_shrink": total_ratio_shrink / count
        }

class EfSGetterV2:
    """
    Manages and infers the search range of 'efSearch' (efS) for a given efC.
    This class is specific to a single M value.

    This version combines two heuristics for efficient and aggressive search
    space reduction without sacrificing performance:
    1. Inference from the single closest neighbor.
    2. Linear extrapolation using the two closest neighbors.
    3. Using the Other M values to shrink the search space further.
    """
    def __init__(self, mode="full"):
        self.__data = {}  # M -> efC -> efS
        # Stores analytics on how much the search space was shrunk.
        self.__shrinked_degrees_of_heuristic1 = []
        self.__shrinked_degrees_of_heuristic2 = []
        self.__shrinked_degrees_of_heuristic3 = []
        self.__shrinked_degress_of_total = []
        self.mode = mode    # "full" or "base" or "h1" or "h2"

    def put(self, M: int, efC: int, efS: int):
        if M not in self.__data:
            self.__data[M] = {}
        self.__data[M][efC] = efS

    def get(self, M: int, efC: int) -> tuple[int, int]:
        if M in self.__data and efC in self.__data[M]:
            known_efS = self.__data[M][efC]
            return known_efS, known_efS
        
        if M not in self.__data or not len(self.__data[M]):
            return EFS_MIN, EFS_MAX

        # Heuristic1 : Inference from the single closest neighbor
        sorted_keys = sorted(self.__data[M].keys())
        lower_efCs = [k for k in sorted_keys if k < efC]
        upper_efCs = [k for k in sorted_keys if k > efC]

        efS_min, efS_max = EFS_MIN, EFS_MAX
        initial_min1, initial_max1 = efS_min, efS_max
        if lower_efCs:
            efS_max = min(efS_max, self.__data[M][lower_efCs[-1]])
        if upper_efCs:
            efS_min = max(efS_min, self.__data[M][upper_efCs[0]])
        size_shrink = (initial_max1 - initial_min1) - (efS_max - efS_min)
        ratio_shrink = size_shrink / (initial_max1 - initial_min1) if (initial_max1 - initial_min1) > 0 else 0.0
        self.__shrinked_degrees_of_heuristic1.append((size_shrink, ratio_shrink))
        # Heuristic2 : Linear extrapolation using the two closest neighbors 
        initial_min2, initial_max2 = efS_min, efS_max
        if len(lower_efCs) >= 2:
            efc1, efc2 = lower_efCs[-2], lower_efCs[-1]
            efs1, efs2 = self.__data[M][efc1], self.__data[M][efc2]
            
            if efc2 > efc1:
                slope = (efs2 - efs1) / (efc2 - efc1)
                extrapolated_min = efs2 + slope * (efC - efc2)
                efS_min = max(efS_min, int(extrapolated_min))

        if len(upper_efCs) >= 2:
            efc2, efc3 = upper_efCs[0], upper_efCs[1]
            efs2, efs3 = self.__data[M][efc2], self.__data[M][efc3]

            if efc3 > efc2:
                slope = (efs3 - efs2) / (efc3 - efc2)
                extrapolated_min = efs2 - slope * (efc2 - efC)
                efS_min = max(efS_min, int(extrapolated_min))
        if upper_efCs and lower_efCs:
            efc1, efc2 = lower_efCs[-1], upper_efCs[0]
            efs1, efs2 = self.__data[M][efc1], self.__data[M][efc2]
            if efc2 > efc1:
                slope = (efs2 - efs1) / (efc2 - efc1)
                extrapolated_max = efs2 + slope * (efC - efc2)
                efS_max = min(efS_max, int(extrapolated_max))
        size_shrink = (initial_max2 - initial_min2) - (efS_max - efS_min)
        ratio_shrink = size_shrink / (initial_max2 - initial_min2) if (initial_max2 - initial_min2) > 0 else 0.0
        self.__shrinked_degrees_of_heuristic2.append((size_shrink, ratio_shrink))
        
        # Heuristic3 : Use the Other M values to shrink the search space further
        initial_min3, initial_max3 = efS_min, efS_max

        lower_Ms = [m for m in self.__data if m < M]
        upper_Ms = [m for m in self.__data if m > M]
        
        for lower_M in lower_Ms:
            lower_efCs = [k for k in self.__data[lower_M] if k <= efC]
            for lower_efC in lower_efCs:
                efS_max = min(efS_max, self.__data[lower_M][lower_efC])

        for upper_M in upper_Ms:
            upper_efCs = [k for k in self.__data[upper_M] if k >= efC]
            for upper_efC in upper_efCs:
                efS_min = max(efS_min, self.__data[upper_M][upper_efC])
        print(f"{initial_max3} {initial_min3} {efS_max} {efS_min}")
        ratio_shrink = size_shrink / (initial_max3 - initial_min3) if (initial_max3 - initial_min3) > 0 else 0.0
        self.__shrinked_degrees_of_heuristic3.append((size_shrink, ratio_shrink))
        
        size_shrink = (EFS_MAX - EFS_MIN) - (efS_max - efS_min)
        ratio_shrink = size_shrink / (EFS_MAX - EFS_MIN)
        self.__shrinked_degress_of_total.append((size_shrink, ratio_shrink))
        return efS_min, efS_max

    def clear(self):
        """Clears all stored data for a new run."""
        self.__data.clear()
        self.__shrinked_degrees_of_heuristic1.clear()

    def get_shrinked_degree_stats(self):
        """Calculates and returns statistics on search space reduction."""
        stats = {}
        if not self.__shrinked_degrees_of_heuristic1:
            stats["heuristic1"] = {"avg_size_shrink": 0.0, "avg_ratio_shrink": 0.0}
        else:
            total_size_shrink = sum(s for s, r in self.__shrinked_degrees_of_heuristic1)
            total_ratio_shrink = sum(r for s, r in self.__shrinked_degrees_of_heuristic1)
            count = len(self.__shrinked_degrees_of_heuristic1)
            stats["heuristic1"] = {
                "avg_size_shrink": total_size_shrink / count,
                "avg_ratio_shrink": total_ratio_shrink / count
            }
        
        if not self.__shrinked_degrees_of_heuristic2:
            stats["heuristic2"] = {"avg_size_shrink": 0.0, "avg_ratio_shrink": 0.0}
        else:
            total_size_shrink = sum(s for s, r in self.__shrinked_degrees_of_heuristic2)
            total_ratio_shrink = sum(r for s, r in self.__shrinked_degrees_of_heuristic2)
            count = len(self.__shrinked_degrees_of_heuristic2)
            stats["heuristic2"] = {
                "avg_size_shrink": total_size_shrink / count,
                "avg_ratio_shrink": total_ratio_shrink / count
            }
        if not self.__shrinked_degrees_of_heuristic3:
            stats["heuristic3"] = {"avg_size_shrink": 0.0, "avg_ratio_shrink": 0.0}
        else:
            total_size_shrink = sum(s for s, r in self.__shrinked_degrees_of_heuristic3)
            total_ratio_shrink = sum(r for s, r in self.__shrinked_degrees_of_heuristic3)
            count = len(self.__shrinked_degrees_of_heuristic3)
            stats["heuristic3"] = {
                "avg_size_shrink": total_size_shrink / count,
                "avg_ratio_shrink": total_ratio_shrink / count
            }
        if not self.__shrinked_degress_of_total:
            stats["total"] = {"avg_size_shrink": 0.0, "avg_ratio_shrink": 0.0}
        else:
            total_size_shrink = sum(s for s, r in self.__shrinked_degress_of_total)
            total_ratio_shrink = sum(r for s, r in self.__shrinked_degress_of_total)
            count = len(self.__shrinked_degress_of_total)
            stats["total"] = {
                "avg_size_shrink": total_size_shrink / count,
                "avg_ratio_shrink": total_ratio_shrink / count
            }
        return stats
    
    def __contains__(self, item: tuple[int, int]) -> bool:
        if not isinstance(item, tuple) or len(item) != 2:
            return False
        M, efC = item
        return M in self.__data and efC in self.__data[M]

class EfSGetterV3:
    """
    Manages and infers the search range of 'efSearch' (efS) for a given efC.
    This class is specific to a single M value.

    This version combines two heuristics for efficient and aggressive search
    space reduction without sacrificing performance:
    1. Inference from the single closest neighbor.
    2. Linear extrapolation using the two closest neighbors.
    3. Using the Other M values to shrink the search space further.
    """
    def __init__(self, mode="heuristic"):
        self.__data = {}  # M -> efC -> efS
        # Stores analytics on how much the search space was shrunk.
        self.__called_count = 0
        self.__ratio_candidates = 0.0
        self.mode = mode    # heuristic or base

    def put(self, M: int, efC: int, efS: int):
        if M not in self.__data:
            self.__data[M] = {}
        self.__data[M][efC] = efS

    def get(self, M: int, efC: int) -> tuple[int, int]:
        if M in self.__data and efC in self.__data[M]:
            known_efS = self.__data[M][efC]
            return known_efS, known_efS
        
        if M not in self.__data or not len(self.__data[M]):
            return EFS_MIN, EFS_MAX

        # Heuristic1 : Inference from the single closest neighbor
        sorted_keys = sorted(self.__data[M].keys())
        lower_efCs = [k for k in sorted_keys if k < efC]
        upper_efCs = [k for k in sorted_keys if k > efC]

        efS_min, efS_max = EFS_MIN, EFS_MAX
        if self.mode == "heuristic":
            if lower_efCs:
                efS_max = min(efS_max, self.__data[M][lower_efCs[-1]])
            if upper_efCs:
                efS_min = max(efS_min, self.__data[M][upper_efCs[0]])
            lower_Ms = [m for m in self.__data if m < M]
            upper_Ms = [m for m in self.__data if m > M]
            
            for lower_M in lower_Ms:
                lower_efCs = [k for k in self.__data[lower_M] if k <= efC]
                for lower_efC in lower_efCs:
                    efS_max = min(efS_max, self.__data[lower_M][lower_efC])

            for upper_M in upper_Ms:
                upper_efCs = [k for k in self.__data[upper_M] if k >= efC]
                for upper_efC in upper_efCs:
                    efS_min = max(efS_min, self.__data[upper_M][upper_efC])

        self.__called_count += 1
        self.__ratio_candidates += (efS_max - efS_min) / (EFS_MAX - EFS_MIN)
        if self.mode != "heuristic":
            return EFS_MIN, EFS_MAX 
        return efS_min, efS_max

    def clear(self):
        """Clears all stored data for a new run."""
        self.__data.clear()

    def stats(self):
        return self.__ratio_candidates / self.__called_count if self.__called_count > 0 else 1.0
    
    def __contains__(self, item: tuple[int, int]) -> bool:
        if not isinstance(item, tuple) or len(item) != 2:
            return False
        M, efC = item
        return M in self.__data and efC in self.__data[M]
