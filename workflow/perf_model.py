from utils.solver import PCPSolver
import numpy as np
import json
import scipy.optimize as scipy_opt


 # A config example in profiling, should be decided later
config_pairs = [[128, 20], [128, 30], [128, 60], [128, 120], [128, 240], 
                [512, 20], [512, 30], [512, 60], [512, 120], [512, 240], 
                [2048, 20], [2048, 30], [2048, 60], [2048, 120], [2048, 240]]
# Better practice: no duplicate equivalent vCPU allocation in the config pairs

step_names = ['delay', 'cold', 'read', 'compute', 'write']

def eq_vcpu_alloc(config):
    assert isinstance(config, list) and len(config) == 2 and isinstance(config[0], int) and \
        isinstance(config[1], int)
    mem, num_func = config
    return mem / 1792 * num_func

def io_func(x, a, b):
    return a / x + b

def comp_func(x, a, b, c, d):
    return a / x - b * np.log(x) / x + c / x**2 + d

'''
StagePerfModel records the parameter distributions of a stage's performance model
Advantage: it is more accurate for a single stage and needs less profiling samples to be trained
Shortcoming: it ignores the partitioning reading overhead (k*d, d is the degree of parallelism
of the stage's parent stage, the overhead is usually high for all-to-all shuffle)
TODO: Compare stage-level model to workflow-level model
'''
class StagePerfModel:
    # TODO: add the support for configuring the default input size in class Stage
    def __init__(self, stage_name, default_input_size=1024) -> None:
        assert isinstance(stage_name, str)
        self.stage_name = stage_name

        assert isinstance(default_input_size, int) and default_input_size > 0
        self.default_input_size = default_input_size  # MB

        self.delay_params = []  # random variable
        self.cold_params = []  # random variable
        self.read_params = []  # A/d + B, d is the equivalent vCPU allocation
        self.compute_params = []  # A/d - B*log(d)/d + C/d**2 + D
        self.write_params = []  # A/d + B

    def train(self, profile_path) -> None:
        assert isinstance(profile_path, str) and profile_path.endswith('.json')
        profile = None
        with open(profile_path, 'r') as f:
            profile = json.load(f)
        assert isinstance(profile, dict) and self.stage_name in profile
        stage_profile = profile[self.stage_name]
        assert isinstance(stage_profile, dict) and 'delay' in stage_profile and \
            'cold' in stage_profile and 'read' in stage_profile and \
            'compute' in stage_profile and 'write' in stage_profile

        # For scheduling delay and cold start, just a random sample
        self.delay_params = np.array(stage_profile['delay'])
        self.delay_params = np.sort(self.delay_params.flatten())
        self.cold_params = np.array(stage_profile['cold'])
        self.cold_params = np.sort(self.cold_params.flatten())

        # d is the equivalent vCPU allocation, i.e., vCPU/func * num_func
        d = np.array([eq_vcpu_alloc(c) for c in config_pairs])

        y_r = np.array(stage_profile['read'])
        y_c = np.array(stage_profile['compute'])
        y_w = np.array(stage_profile['write'])
        num_epochs = y_r.shape[0] 
        for i in range(num_epochs):
            # use non-linear least square to fit the parameters of the IO model
            popt, pcov = scipy_opt.curve_fit(io_func, d, y_r[i])
            self.read_params.append(popt)
            popt, pcov = scipy_opt.curve_fit(comp_func, d, y_c[i])
            self.compute_params.append(popt)
            popt, pcov = scipy_opt.curve_fit(io_func, d, y_w[i])
            self.write_params.append(popt)

        self.read_params = np.array(self.read_params).T
        self.compute_params = np.array(self.compute_params).T
        self.write_params = np.array(self.write_params).T

    def predict(self, input_size, num_vcpu, num_func, mode='latency') -> float:
        # input_size uses MB as unit
        assert (isinstance(input_size, int) or isinstance(input_size, float)) and input_size > 0
        assert (isinstance(num_vcpu, int) or isinstance(num_vcpu, float)) and num_vcpu > 0
        assert isinstance(num_func, int) and num_func > 0
        assert mode in ['latency', 'cost']

        d = num_vcpu * num_func
        pred = 0

        # For scheduling delay and cold start, just a random sample
        pred += np.random.choice(self.delay_params) + np.random.choice(self.cold_params)
        if mode == 'latency':
            pred += np.max(io_func(d, self.read_params[0], self.read_params[1]))
            pred += np.max(comp_func(d, self.compute_params[0], self.compute_params[1], 
                           self.compute_params[2], self.compute_params[3]))
            pred += np.max(io_func(d, self.write_params[0], self.write_params[1]))
            pred *= input_size / self.default_input_size
            return pred
        else:
            pred += np.sum(io_func(d, self.read_params[0], self.read_params[1]))
            pred += np.sum(comp_func(d, self.compute_params[0], self.compute_params[1], 
                           self.compute_params[2], self.compute_params[3]))
            pred += np.sum(io_func(d, self.write_params[0], self.write_params[1]))
            pred *= input_size / self.default_input_size
            return pred * d * 1792 / 1024 * 0.0000000167 + 0.2 / 1000000


    def sample_offline(self, num_samples):
        assert isinstance(num_samples, int) and num_samples > 0
        # Sample for num_samples times
        res = []
        res.append(np.random.choice(self.delay_params, num_samples).tolist())
        res.append(np.random.choice(self.cold_params, num_samples).tolist())
        for i in range(self.read_params.shape[0]):
            res.append(np.random.choice(self.read_params[i], num_samples).tolist())
        for i in range(self.compute_params.shape[0]):
            res.append(np.random.choice(self.compute_params[i], num_samples).tolist())
        for i in range(self.write_params.shape[0]):
            res.append(np.random.choice(self.write_params[i], num_samples).tolist())
        return res

    def __str__(self):
        return self.stage_name
    
    def __getstate__(self):
        self_dict = self.__dict__.copy()
        return self_dict
    
    def __del__(self):
        pass