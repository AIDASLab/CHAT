import sys

from src.constants import TUNING_BUDGET
from data.ground_truths.ground_truth import GroundTruth 

import joblib
from scipy.stats import qmc
import json
import numpy as np
import time
import subprocess as sp
import random
# from configure import *

KNOB_PATH = r'/home/vd_tuner/VDTuner/auto-configure/vdtuner/dev/whole_param.json'
RUN_ENGINE_PATH = r'/home/vd_tuner/VDTuner/vector-db-benchmark-master/run_engine.sh'

class KnobStand:
    def __init__(self, path) -> None:
        self.path = path    #* KNOB_PATH
        with open(path, 'r') as f:
            self.knobs_detail = json.load(f)
            #* self.knobs_detail : param -> {class, type, default, *values}

    def scale_back(self, knob_name, zero_one_val):  #* zero_one_val -> real_val : Denormalize to [min,max] or enum_value
        knob = self.knobs_detail[knob_name]
        if knob['type'] == 'integer':
            real_val = zero_one_val * (knob['max'] - knob['min']) + knob['min']
            return int(real_val), int(real_val)

        elif knob['type'] == 'enum':
            enum_size = len(knob['enum_values'])
            enum_index = int(enum_size * zero_one_val)
            enum_index = min(enum_size - 1, enum_index)
            real_val = knob['enum_values'][enum_index]
            return enum_index, real_val
    
    def scale_forward(self, knob_name, real_val):   #* real_val -> zero_one_val : Normalize to [0,1]
        knob = self.knobs_detail[knob_name]
        #! print(f"{knob_name} -> {knob}")
        if knob['type'] == 'integer':
            zero_one_val = (real_val - knob['min']) / (knob['max'] - knob['min'])
            return zero_one_val

        elif knob['type'] == 'enum':
            enum_size = len(knob['enum_values'])
            zero_one_val = knob['enum_values'].index(real_val) / enum_size
            return zero_one_val

class MockEnv:
    def __init__(self, model:GroundTruth=None, knob_path=None, tuning_budget=TUNING_BUDGET, search_efS=True) -> None:
        self.model = model
        self.knob_stand = KnobStand(knob_path)
        self.names = list(self.knob_stand.knobs_detail.keys())
        self.t1 = time.time()
        self.tuning_budget = tuning_budget
        self.sampled_times = 0
        self.search_efS = search_efS
        #### 
        self.X_record = []
        self.Y1_record = [] #* all QPS
        self.Y2_record = [] #* all recall
        self.Y_record = []
        self.T_record = []  #* search time up to now
    
    def get_state(self, knob_vals_arr):
        Y1, Y2, Y3 = [], [], []
        for i,record in enumerate(knob_vals_arr):
            conf_value = [self.knob_stand.scale_back(self.names[j], knob_val)[0] for j,knob_val in enumerate(record)]
            print(f"conf_value : {conf_value}")
            _M = conf_value[1]
            _efC = conf_value[2]
            _efS = conf_value[3] if self.search_efS else _efC
            y = self.model.get(M=_M, efC=_efC, efS=_efS)
            y1 = y[1]   #* QPS (It's not a mistake, y[1] is QPS)
            y2 = y[0]   #* recall
            y3 = y[2]   #* total time
            if self.model.tuning_time + (time.time() - self.t1) > self.tuning_budget:
                print(f"{self.model.tuning_time} + {time.time() - self.t1} = {self.model.tuning_time + time.time() - self.t1} > {self.tuning_budget}")
                raise TimeoutError("Tuning budget exceeded")
            self.X_record.append((_M, _efC, _efS))  #* (M, efC, efS)
            self.Y1_record.append(y1)
            self.Y2_record.append(y2)
            self.Y_record.append((y2, y1))  #* (y2, y1) == (recall, QPS)
            self.T_record.append(self.model.tuning_time)
            print(f"self.tuning_time : {self.model.tuning_time}")
            print(f"T_record : {self.T_record[-1]}\n") 
            self.sampled_times += 1
            # print(f'[{self.sampled_times}] {int(time.time()-self.t1)} {y1:.2f} {y2:.2f} {y3:.2f} ({self.model.tuning_time:.2f})\n')
            print(f'[{self.sampled_times}] {y1:.2f} {y2:.2f} {y3:.2f}  ({int(self.model.tuning_time / (TUNING_BUDGET) * 100)}%)\n')
            
            Y1.append(y1)
            Y2.append(y2)
            Y3.append(y3)
        return np.array([Y1,Y2,Y3]).T
    
    def default_conf(self):
        return [self.knob_stand.scale_forward(k, v['default']) for k,v in self.knob_stand.knobs_detail.items()]

class TestEnv:
    def __init__(self, model=None, knob_path=None, tuning_budget=None):
        self.tuning_budget = tuning_budget
        self.X_record = []
        self.Y1_record = []
        self.Y2_record = []
        self.Y_record = []
        self.T_record = []
        self.model = {"tuning_time": 0.0}
    
    def get_state(self, knob_vals_arr):
        Y1, Y2, Y3 = [], [], []
        for i,record in enumerate(knob_vals_arr):
            conf_value = [self.knob_stand.scale_back(self.names[j], knob_val)[0] for j,knob_val in enumerate(record)]
            print(f"conf_value : {conf_value}")
            _X = conf_value[1]
            _Y = conf_value[2]
            y1 = -(_X - 50)**2 - (_Y - 50)**2 + 10000
            y2 = 5000
            y3 = 100.0
            if self.model.tuning_time > self.tuning_budget:
                raise TimeoutError("Tuning budget exceeded")
            self.X_record.append((_X, _Y))
            self.Y1_record.append(y1)
            self.Y2_record.append(y2)
            self.Y_record.append((y1, y2))
            self.T_record.append(self.model.tuning_time)
            print(f"[{self.sampled_times}] {y1:.2f} {y2:.2f} {y3:.2f} ({self.model.tuning_time:.2f})\n")
            self.sampled_times += 1
            self.model.tuning_time += 100
            Y1.append(y1)
            Y2.append(y2)
            Y3.append(y3)
        return np.array([Y1,Y2,Y3]).T
