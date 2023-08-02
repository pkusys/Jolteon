import numpy as np
import matplotlib.pyplot as plt
import math
import json
from utils import Distribution
import time

# config_pairs = [[1792, 1], [1792, 2], [1792, 4], [1792, 8],
#                 [3584, 1], [3584, 2], [3584, 4], [3584, 8],
#                 [7168, 1], [7168, 2], [7168, 4], [7168, 8]]

config_pairs = [[1024, 2], [1024, 4], [1024, 8],
                [1792, 1], [1792, 2], [1792, 4], [1792, 8],
                [2048, 1], [2048, 2], [2048, 4], [2048, 8],
                [3584, 1], [3584, 2], [3584, 4], [3584, 8],
                [7168, 1], [7168, 2], [7168, 4], [7168, 8]]

def eq_vcpu_alloc(mem, num_func):
    num_vcpu = mem / 1792
    # num_vcpu = math.ceil(mem / 1792)
    # num_vcpu = math.floor(mem / 1792)
    # num_vcpu = max(1, num_vcpu)
    return round(num_vcpu * num_func, 1)

# Only for one stage
class DistPerfModel:
    def __init__(self, stage_name) -> None:
        assert isinstance(stage_name, str)
        self.stage_name = stage_name

        self.allow_parallel = True

        self.distributions = {}  # k*d -> Distribution
        
        self.max_func_size = None
        
        self.up_models = []
        
        self.func_size = 1.0
        
    def update_allow_parallel(self, allow_parallel) -> None:
        assert isinstance(allow_parallel, bool)
        self.allow_parallel = allow_parallel
        
    def set_func_size(self, func_size) -> None:
        try:
            func_size = round(float(func_size), 1)
        except:
            raise ValueError('Invalid function size: %s' % func_size)
        self.func_size = func_size
        
    def train(self, profile_path) -> None:
        assert isinstance(profile_path, str) and profile_path.endswith('.json')
        profile = None
        with open(profile_path, 'r') as f:
            profile = json.load(f)
        assert isinstance(profile, dict) and self.stage_name in profile
        stage_profile = profile[self.stage_name]
        assert isinstance(stage_profile, dict) and 'cold' in stage_profile and \
            'read' in stage_profile and 'compute' in stage_profile and \
            'write' in stage_profile

        print('Training Orion performance model for %s' % self.stage_name)
        
        read_arr = np.array(stage_profile['read'])[1:,:,1]
        com_arr = np.array(stage_profile['compute'])[1:,:,1]
        write_arr = np.array(stage_profile['write'])[1:,:,1]
        cold_arr = np.array(stage_profile['cold'])[1:,:,1]
        
        stage_arr = read_arr + com_arr + write_arr + cold_arr
        size2points = {}
        
        for idx, config in enumerate(config_pairs):
            mem = config[0]
            num_func = config[1]
            # adapt to parallel mode
            if self.allow_parallel:
                func_size = eq_vcpu_alloc(mem, num_func)
            else:
                func_size = eq_vcpu_alloc(mem, 1)
                
            if func_size not in size2points:
                size2points[func_size] = []
            for ep in stage_arr:
                size2points[func_size].append(ep[idx])
        
        for func_size in size2points:
            self.distributions[func_size] = Distribution(size2points[func_size])
            
        self.max_func_size = max(self.distributions.keys())
    
    # recursive function
    def calculate(self):
        cur_dist = self.interpolation(self.func_size)
        
        if len(self.up_models) == 0:
            return cur_dist
        
        sub_dist = None
        
        for sub_model in self.up_models:
            dist = sub_model.calculate()
            if sub_dist is None:
                sub_dist = dist
            else:
                sub_dist.combine(dist, 1)
        cur_dist.combine(sub_dist, 0)
        
        return cur_dist
        
    
    # Extract the distribution for a specific function size with percentile-wise linear interpolation
    def interpolation(self, func_size):
        assert func_size >= 0
        
        if func_size in self.distributions:
            return self.distributions[func_size].copy()
        
        if func_size >= self.max_func_size:
            return self.distributions[max(self.distributions.keys())].copy()
        
        func_size_list = sorted(self.distributions.keys())
        for idx, val in enumerate(func_size_list):
            if val > func_size:
                break
        lower_idx = idx - 1
        upper_idx = idx
        
        lower_dist = self.distributions[func_size_list[lower_idx]] if lower_idx >= 0 else Distribution([0])
        upper_dist = self.distributions[func_size_list[upper_idx]]
        
        lower_size = func_size_list[lower_idx] if lower_idx >= 0 else 0
        upper_size = func_size_list[upper_idx]
        
        assert func_size > lower_size and func_size < upper_size
        
        # y1 + (y2 - y1) * (x - x1) / (x2 - x1)
        slope = (func_size - lower_size) / (upper_size - lower_size)
        new_data = []
        new_prob = []
        
        for i in range(len(lower_dist.data)):
            for j in range(len(upper_dist.data)):
                new_data.append(lower_dist.data[i] + (upper_dist.data[j] - lower_dist.data[i]) * slope)
                new_prob.append(lower_dist.prob[i] * upper_dist.prob[j])
                
        return Distribution(new_data, new_prob)
    
    def add_up_model(self, model):
        assert isinstance(model, DistPerfModel)
        self.up_models.append(model)
    
if __name__ == '__main__':
    perfmodel0 = DistPerfModel('stage0')
    perfmodel1 = DistPerfModel('stage1')
    perfmodel2 = DistPerfModel('stage2')
    perfmodel3 = DistPerfModel('stage3')
    models = [perfmodel0, perfmodel1, perfmodel2, perfmodel3]
    pro_file = '/home/ubuntu/workspace/chaojin-dev/serverless-bound/profiles/ML-Pipeline_profile.json'
    perfmodel0.update_allow_parallel(False)
    perfmodel3.update_allow_parallel(False)
    
    perfmodel3.up_models.append(perfmodel2)
    perfmodel2.up_models.append(perfmodel1)
    perfmodel1.up_models.append(perfmodel0)
    
    for m in models:
        m.train(pro_file)
        
    perfmodel1.set_func_size(12)
    t0 = time.time()
    dist = perfmodel3.calculate()
    print(dist)
    dist = perfmodel3.calculate()
    print(dist)
    dist = perfmodel3.calculate()
    t1 = time.time()
    
    print('Time cost: %f' % (t1 - t0))
    
    print(dist)
    print(dist.probility(23))