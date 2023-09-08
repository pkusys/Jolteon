import numpy as np
import math
import json
import time
import scipy.optimize as scipy_opt

from perf_model import config_pairs, eq_vcpu_alloc

step_names = ['read', 'compute', 'write']

def io_func(x, a, b):
    return a / x + b

def comp_func(x, a, b):
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

        self.cold_params = None
        
    def update_allow_parallel(self, allow_parallel) -> None:
        assert isinstance(allow_parallel, bool)
        self.allow_parallel = allow_parallel
        
    def fit_params(self, data, func):
        assert isinstance(data, dict)
        arr_x = list(data.keys())
        arr_y = [data[x] for x in arr_x]
        
        arr_x = np.array(arr_x)
        arr_y = np.array(arr_y)

        initial_guess = [1, 1]
        
        # Two parameters' covariance could not be estimated when use curve_fit, 
        # so we use leastsq instead.
        params, _ = scipy_opt.leastsq(lambda para, x, y: func(x, *para) - y, initial_guess, args=(arr_x, arr_y))
        
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

        self.cold_params = np.array(stage_profile['cold'])[1:,:,0].reshape(-1)
        self.cold_params = np.mean(self.cold_params)
        
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
        
        # target_dict = size2points_comp
        # arr_x = list(target_dict.keys())
        # self.visualize(arr_x, [target_dict[x] for x in arr_x], io_func, self.comp_params)
        
    def get_params(self):
        a = sum([self.read_params[0], self.comp_params[0], self.write_params[0]])
        b = sum([self.read_params[1], self.comp_params[1], self.write_params[1]])
        
        return a, b

    def predict(self, num_vcpu, num_func, mode='latency', parent_d=0, cold_percent=60):
        a, b = self.get_params()
        assert num_func > 0
        return self.cold_params + a / num_func + b

    def predict_tile(self, config, profile_path, num_samples, tile=95):
        mem, num_func = config  # Note that the config should be in config_pairs
        assert config in config_pairs
        num_vcpu = mem / 1792
        assert num_vcpu > 0 and num_vcpu <= 10
        assert num_func > 0

        with open(profile_path, 'r') as f:
            profile = json.load(f)
        assert isinstance(profile, dict) and self.stage_name in profile
        stage_profile = profile[self.stage_name]
        assert isinstance(stage_profile, dict) and 'cold' in stage_profile and \
            'read' in stage_profile and 'compute' in stage_profile and \
            'write' in stage_profile
        y_s = np.array(stage_profile['cold'])
        num_epochs = y_s.shape[0]
        assert num_epochs >= 2
        num_epochs -= 1  # Remove the first cold start epoch
        y_s = y_s[1:][:,:,0]
        y_r = np.array(stage_profile['read'])[1:][:,:,0]
        y_c = np.array(stage_profile['compute'])[1:][:,:,0]
        y_w = np.array(stage_profile['write'])[1:][:,:,0]

        cfg_idx = config_pairs.index(config)

        actuals = []
        for i in range(num_epochs):
            y = y_s[i][cfg_idx] + y_r[i][cfg_idx] + y_c[i][cfg_idx] + y_w[i][cfg_idx]
            actuals.append(y)
        actuals = np.array(actuals)
        actual95 = np.percentile(actuals, 95)
        actual50 = np.percentile(actuals, 50)

        a, b = self.get_params()
        assert num_func > 0
        pred = self.cold_params + a / num_func + b

        err95 = (pred - actual95) / actual95
        err50 = (pred - actual50) / actual50
        print('Predicted: %.2f, Actual95: %.2f, Actual50: %.2f, Err95: %.2f, Err50: %.2f' % \
            (pred, actual95, actual50, err95, err50))
            
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