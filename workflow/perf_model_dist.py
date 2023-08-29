import numpy as np
import math
import json
import time

from utils import Distribution

config_pairs = [[1024, 8], [1024, 16], [1024, 32], 
                [1792, 4], [1792, 8], [1792, 16], [1792, 32],
                [3584, 4], [3584, 8], [3584, 16], [3584, 32], 
                [5120, 4], [5120, 8], [5120, 16], [5120, 32], 
                [6144, 4], [6144, 8], [6144, 16], [6144, 32], 
                [7168, 4], [7168, 8], [7168, 16], [7168, 32]]

def get_config_pairs_dist(wf_name):
    # the len
    if wf_name == 'ML-Pipeline':
        pairs = [[1024, 8], [1024, 16], [1024, 32], 
                    [1792, 4], [1792, 8], [1792, 16], [1792, 32],
                    [3584, 4], [3584, 8], [3584, 16], [3584, 32], 
                    [5120, 4], [5120, 8], [5120, 16], [5120, 32], 
                    [6144, 4], [6144, 8], [6144, 16], [6144, 32], 
                    [7168, 4], [7168, 8], [7168, 16], [7168, 32]]
    elif wf_name == 'Video-Analytics':
        pairs = [[1792, 4], [1792, 8], [1792, 16], [1792, 32],
                    [3584, 4], [3584, 8], [3584, 16], [3584, 32], 
                    [5120, 4], [5120, 8], [5120, 16], [5120, 32], 
                    [6144, 4], [6144, 8], [6144, 16], [6144, 32], 
                    [7168, 4], [7168, 8], [7168, 16], [7168, 32], 
                    [8960, 4], [8960, 8], [8960, 16], [8960, 32]]
    elif wf_name == 'tpcds/dsq95':
        pairs = [[892, 16], [892, 24], [892, 32], [892, 48], [892, 64], 
                 [1078, 16], [1078, 24], [1078, 32], [1078, 48], [1078, 64], 
                 [1258, 16], [1258, 24], [1258, 32], [1258, 48], [1258, 64],
                 [1437, 16], [1437, 24], [1437, 32], [1437, 48], [1437, 64],
                 [1617, 16], [1617, 24], [1617, 32], [1617, 48], [1617, 64],
                 [1792, 8], [1792, 16], [1792, 24], [1792, 32], [1792, 48], [1792, 64]]
    else:
        raise ValueError('Unknown workflow name: %s' % wf_name)
    
    global config_pairs
    config_pairs = pairs
    return config_pairs

def eq_vcpu_alloc(mem, num_func):
    num_vcpu = mem / 1792
    # num_vcpu = math.ceil(mem / 1792)
    # num_vcpu = math.floor(mem / 1792)
    # num_vcpu = max(1, num_vcpu)
    return round(num_vcpu * num_func, 1)

'''
DistPerfModel records the latency distributions of a stage's performance model under different function config.
Advantage: it is more accurate than analytic model.
Shortcoming: its search space is too large for configuration under function size and parallelism.
The interpolation is not accurate enough.
The search result is not optimal and is also not bounded with a constraint.
'''
class DistPerfModel:
    def __init__(self, _stage_id, _stage_name) -> None:
        assert isinstance(_stage_name, str)
        assert isinstance(_stage_id, int) and _stage_id >= 0
        self.stage_name = _stage_name
        self.stage_id = _stage_id

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
        check_1 = isinstance(stage_profile, dict) and 'cold' in stage_profile and \
            'read' in stage_profile and 'compute' in stage_profile and \
            'write' in stage_profile
        check_2 = isinstance(stage_profile, dict) and 'e2e' in stage_profile
        assert check_1 or check_2

        # print('Training Orion performance model for %s' % self.stage_name)
        if 'e2e' not in stage_profile:
            read_arr = np.array(stage_profile['read'])[1:,:,1]
            com_arr = np.array(stage_profile['compute'])[1:,:,1]
            write_arr = np.array(stage_profile['write'])[1:,:,1]
            cold_arr = np.array(stage_profile['cold'])[1:,:,1]
            
            stage_arr = read_arr + com_arr + write_arr + cold_arr
        else:
            stage_arr = np.array(stage_profile['e2e'])[1:,:]
        size2points = {}
        
        for idx, config in enumerate(config_pairs):
            # if idx == 14 or idx == 17:
            #     continue
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
    perfmodel0 = DistPerfModel(0, 'stage0')
    perfmodel1 = DistPerfModel(1, 'stage1')
    perfmodel2 = DistPerfModel(2, 'stage2')
    perfmodel3 = DistPerfModel(3, 'stage3')
    perfmodel4 = DistPerfModel(4, 'stage4')
    perfmodel5 = DistPerfModel(5, 'stage5')
    perfmodel6 = DistPerfModel(6, 'stage6')
    perfmodel7 = DistPerfModel(7, 'stage7')
    models = [perfmodel0, perfmodel1, perfmodel2, perfmodel3, 
                perfmodel4, perfmodel5, perfmodel6, perfmodel7]
    pro_file = '/home/ubuntu/workspace/chaojin-dev/serverless-bound/profiles/tpcds-dsq95_profile.json'
    perfmodel1.update_allow_parallel(False)
    perfmodel7.update_allow_parallel(False)
    
    perfmodel7.up_models.append(perfmodel6)
    perfmodel6.up_models.append(perfmodel4)
    perfmodel6.up_models.append(perfmodel5)
    perfmodel4.up_models.append(perfmodel3)
    perfmodel4.up_models.append(perfmodel2)
    perfmodel4.up_models.append(perfmodel1)
    perfmodel1.up_models.append(perfmodel0)

    config_pairs = get_config_pairs_dist('tpcds/dsq95')
    
    for m in models:
        m.train(pro_file)
        
    perfmodel0.set_func_size(20.*0.5)
    perfmodel1.set_func_size(1.*0.5)
    perfmodel2.set_func_size(20.*0.5)
    perfmodel3.set_func_size(4.*0.5)
    perfmodel4.set_func_size(4.*0.5)
    perfmodel5.set_func_size(4.*0.5)
    perfmodel6.set_func_size(4.*0.5)
    perfmodel7.set_func_size(1.*0.5)

    t0 = time.time()
    dist = perfmodel7.calculate()
    t1 = time.time()
    print(dist)
        
    # print(sorted(list(perfmodel7.distributions.keys())))