from utils.solver import PCPSolver
import numpy as np
import math
import json
import scipy.optimize as scipy_opt


# A config example in profiling, should be decided later
config_pairs = [[1024, 2], [1024, 4], [1024, 8],
                [1792, 1], [1792, 2], [1792, 4], [1792, 8],
                [2048, 1], [2048, 2], [2048, 4], [2048, 8],
                [3584, 1], [3584, 2], [3584, 4], [3584, 8],
                [7168, 1], [7168, 2], [7168, 4], [7168, 8]]

# Better practice: no duplicate equivalent vCPU allocation in the config pairs

step_names = ['cold', 'read', 'compute', 'write']

def eq_vcpu_alloc(mem, num_func):
    num_vcpu = mem / 1792
    # num_vcpu = math.ceil(mem / 1792)
    # num_vcpu = math.floor(mem / 1792)
    # num_vcpu = max(1, num_vcpu)
    return num_vcpu * num_func

def io_func(x, a, b):
    return a / x + b

def comp_func(x, a, b, c, d):
    return a / x - b * np.log(x) / x + c / x**2 + d

'''
StagePerfModel records the parameter distributions of a stage's performance model
Advantage: it is more accurate for a single stage and needs less profiling samples to be trained
Shortcoming: it ignores the partitioning reading overhead (k*d, d is the degree of parallelism
of the stage's parent stage, the overhead is usually high for all-to-all shuffle)
'''
class StagePerfModel:
    def __init__(self, stage_name, default_input_size=1024) -> None:
        assert isinstance(stage_name, str)
        self.stage_name = stage_name

        self.allow_parallel = True

        assert isinstance(default_input_size, int) and default_input_size > 0
        self.default_input_size = default_input_size  # MB or GB

        self.cold_params_avg = []  # random variable
        self.read_params_avg = []  # A/d + B, d is the equivalent vCPU allocation
        self.compute_params_avg = []  # A/d - B*log(d)/d + C/d**2 + D
        self.write_params_avg = []  # A/d + B
        self.cold_params_max = [] 
        self.read_params_max = []
        self.compute_params_max = []  
        self.write_params_max = []
        self.can_intra_parallel = [True, True, True]  # stands for read, compute, write

    def update_allow_parallel(self, allow_parallel) -> None:
        assert isinstance(allow_parallel, bool)
        self.allow_parallel = allow_parallel

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

        # For scheduling delay and cold start, just a random sample
        self.cold_params_avg = np.array(stage_profile['cold'])[:, :, 0].reshape(-1)
        self.cold_params_max = np.array(stage_profile['cold'])[:, :, 1].reshape(-1)

        print('cold_params_avg', self.cold_params_avg)

        # d is the equivalent vCPU allocation, i.e., vCPU/func * num_func
        d = np.array([eq_vcpu_alloc(mem, num_func) for mem, num_func in config_pairs])
        if not self.allow_parallel:
            d = np.array([eq_vcpu_alloc(mem, 1) for mem, num_func in config_pairs])

        y_r = np.array(stage_profile['read'])
        y_c = np.array(stage_profile['compute'])
        y_w = np.array(stage_profile['write'])
        num_epochs = y_r.shape[0] 
        for i in range(num_epochs):
            # use non-linear least square to fit the parameters of the IO model
            popt, pcov = scipy_opt.curve_fit(io_func, d, y_r[i][:, 0])
            self.read_params_avg.append(popt)
            yy = io_func(d, popt[0], popt[1])
            yy_ = y_r[i][:, 0]
            err = (yy - yy_) / yy_
            avg_err = np.mean(np.abs(err))
            print('--------------')
            print('actual', yy_)
            print('predict', yy)
            print('err', err)
            print('avg_err', avg_err)
            print('--------------')
            popt, pcov = scipy_opt.curve_fit(io_func, d, y_r[i][:, 1])
            self.read_params_max.append(popt)

            popt, pcov = scipy_opt.curve_fit(comp_func, d, y_c[i][:, 0])
            self.compute_params_avg.append(popt)
            popt, pcov = scipy_opt.curve_fit(comp_func, d, y_c[i][:, 1])
            self.compute_params_max.append(popt)

            popt, pcov = scipy_opt.curve_fit(io_func, d, y_w[i][:, 0])
            self.write_params_avg.append(popt)
            popt, pcov = scipy_opt.curve_fit(io_func, d, y_w[i][:, 1])
            self.write_params_max.append(popt)

        self.read_params_avg = np.array(self.read_params_avg).T
        self.read_params_max = np.array(self.read_params_max).T
        self.compute_params_avg = np.array(self.compute_params_avg).T
        self.compute_params_max = np.array(self.compute_params_max).T
        self.write_params_avg = np.array(self.write_params_avg).T
        self.write_params_max = np.array(self.write_params_max).T

    def predict(self, mem, num_func, mode='latency', input_size=None) -> float:
        # input_size uses MB as unit
        assert (isinstance(mem, int) or isinstance(mem, float)) and mem >= 128 and mem <= 10240
        assert isinstance(num_func, int) and num_func > 0
        assert mode in ['latency', 'cost']
        if input_size is not None:
            assert (isinstance(input_size, int) or isinstance(input_size, float)) and input_size > 0

        d = [num_func, num_func, num_func]
        for i in range(3):
            if self.can_intra_parallel[i]:
                d[i] = eq_vcpu_alloc(mem, num_func)
        pred = 0

        # For scheduling delay and cold start, just a random sample
        if mode == 'latency':
            pred += np.random.choice(self.cold_params_max)
            pred += np.mean(io_func(d[0], self.read_params_max[0], self.read_params_max[1]))
            pred += np.mean(comp_func(d[1], self.compute_params_max[0], self.compute_params_max[1], 
                           self.compute_params_max[2], self.compute_params_max[3]))
            pred += np.mean(io_func(d[2], self.write_params_max[0], self.write_params_max[1]))
            if input_size is not None:
                pred *= input_size / self.default_input_size
            return pred
        else:
            pred += np.random.choice(self.cold_params_avg)
            pred += np.mean(io_func(d, self.read_params_avg[0], self.read_params_avg[1]))
            pred += np.mean(comp_func(d, self.compute_params_avg[0], self.compute_params_avg[1], 
                           self.compute_params_avg[2], self.compute_params_avg[3]))
            pred += np.mean(io_func(d, self.write_params_avg[0], self.write_params_avg[1]))
            if input_size is not None:
                pred *= input_size / self.default_input_size
            return pred * num_func * mem / 1024 * 0.0000000167 + 0.2 * num_func / 1000000


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