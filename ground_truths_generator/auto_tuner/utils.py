import time
import os
import psutil
import functools
import threading
import numpy as np

execution_depth = threading.local()

def log_execution():
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not hasattr(execution_depth, "level"):
                execution_depth.level = 0
            depth = execution_depth.level
            indent = " " * (depth * 4)

            print(f"{indent}[Running] {func.__name__} ...")

            execution_depth.level += 1
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
            finally:
                execution_depth.level -= 1

            elapsed_time = time.time() - start_time
            print(f"{indent}[Finished] {func.__name__} (time: {elapsed_time:.2f}s)")
            return result
        return wrapper
    return decorator

def set_cpu_affinity(cores:list):
    p = psutil.Process(os.getpid())
    p.cpu_affinity(cores)