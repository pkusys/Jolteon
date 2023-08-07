import numpy as np
import math
import json
import time
import scipy.optimize as scipy_opt

from perf_model import config_pairs, eq_vcpu_alloc

step_names = ['read', 'compute', 'write']

def io_func(x, a, b):
    return a / x + b

def comp_func(x, a, b, c, d):
    return a / x + b

'''
AnaPerfModel records the mean parameter value.
Advantage: it is fast and accurate enough to optimize the average performance.
Shortcoming: it dose not guarantee the bounded performance.
'''
class AnaPerfModel:
    def __init__(self, _stage_id, _stage_name) -> None:
        assert isinstance(_stage_name, str)
        assert isinstance(_stage_id, int) and _stage_id >= 0
        self.stage_name = _stage_name
        self.stage_id = _stage_id

        self.allow_parallel = True
        
        # Init in train, list with size two
        self.write_params = None
        self.read_params = None
        self.comp_params = None
        
    def update_allow_parallel(self, allow_parallel) -> None:
        assert isinstance(allow_parallel, bool)
        self.allow_parallel = allow_parallel
        
    def fit_params(self, data, func):
        assert isinstance(data, dict)
        arr_x = list(data.keys())
        arr_y = [data[x] for x in arr_x]
        
        arr_x = np.array(arr_x)
        arr_y = np.array(arr_y)
        
        params, _ = scipy_opt.curve_fit(func, arr_x, arr_y)
        
        return params.tolist()
        
    def train(self, profile_path) -> None:
        assert isinstance(profile_path, str) and profile_path.endswith('.json')
        profile = None
        with open(profile_path, 'r') as f:
            profile = json.load(f)
        assert isinstance(profile, dict) and self.stage_name in profile
        stage_profile = profile[self.stage_name]
        assert isinstance(stage_profile, dict) and \
            'read' in stage_profile and 'compute' in stage_profile and \
            'write' in stage_profile

        print('Training Analytical performance model for %s' % self.stage_name)
        
        read_arr = np.array(stage_profile['read'])[1:,:,0]
        comp_arr = np.array(stage_profile['compute'])[1:,:,0]
        write_arr = np.array(stage_profile['write'])[1:,:,0]
        
        size2points_read = {}
        size2points_comp = {}
        size2points_write = {}
        
        for idx, config in enumerate(config_pairs):
            mem = config[0]
            num_func = config[1]
            # adapt to parallel mode
            if self.allow_parallel:
                func_size = eq_vcpu_alloc(mem, num_func)
            else:
                func_size = eq_vcpu_alloc(mem, 1)
            
            # collect data for read step
            if func_size not in size2points_read:
                size2points_read[func_size] = []
            for ep in read_arr:
                size2points_read[func_size].append(ep[idx])
            
            # collect data for comp step
            if func_size not in size2points_comp:
                size2points_comp[func_size] = []
            for ep in comp_arr:
                size2points_comp[func_size].append(ep[idx])
            
            # collect data for write step
            if func_size not in size2points_write:
                size2points_write[func_size] = []
            for ep in write_arr:
                size2points_write[func_size].append(ep[idx])
            
            
        # average the data
        for config in size2points_read:
            arr = np.array(size2points_read[config])
            size2points_read[config] = np.mean(arr)
            
        for config in size2points_comp:
            arr = np.array(size2points_comp[config])
            size2points_comp[config] = np.mean(arr)
            
        for config in size2points_write:
            arr = np.array(size2points_write[config])
            size2points_write[config] = np.mean(arr)
                
        # fit the parameters
        self.read_params = self.fit_params(size2points_read, io_func)
        self.comp_params = self.fit_params(size2points_comp, comp_func)
        self.write_params = self.fit_params(size2points_write, io_func)
        
        target_dict = size2points_comp
        arr_x = list(target_dict.keys())
        self.visualize(arr_x, [target_dict[x] for x in arr_x], io_func, self.comp_params)
        
    def get_params(self):
        a = sum([self.read_params[0], self.comp_params[0], self.write_params[0]])
        b = sum([self.read_params[1], self.comp_params[1], self.write_params[1]])
        
        return a, b
            
    def visualize(self, arr_x, arr_y, func, params):
        import matplotlib.pyplot as plt
        plt.scatter(arr_x, arr_y)

        arr_x = np.linspace(min(arr_x), max(arr_x), 100)
        pred_y = [func(x, params[0], params[1]) for x in arr_x]
        plt.plot(arr_x, pred_y, 'r-')
        plt.xlim(0, max(arr_x) * 1.1)
        plt.ylim(0, max(max(arr_y), max(pred_y)) * 1.1)
        plt.savefig('vis.png')

if __name__ == '__main__':
    perfmodel = AnaPerfModel(1, 'stage1')
    pro_file = '/home/ubuntu/workspace/chaojin-dev/serverless-bound/profiles/ML-Pipeline_profile.json'
    perfmodel.update_allow_parallel(False)
    
    perfmodel.train(pro_file)
    
    print(perfmodel.get_params())