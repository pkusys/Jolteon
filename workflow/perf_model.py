import numpy as np
import matplotlib.pyplot as plt
import math
import time
import json
import scipy.optimize as scipy_opt

# A config example in profiling, should be decided later
config_pairs = [[1024, 8], [1024, 16], [1024, 32], 
                [1792, 4], [1792, 8], [1792, 16], [1792, 32],
                [3584, 4], [3584, 8], [3584, 16], [3584, 32], 
                [5120, 4], [5120, 8], [5120, 16], [5120, 32], 
                [6144, 4], [6144, 8], [6144, 16], [6144, 32], 
                [7168, 4], [7168, 8], [7168, 16], [7168, 32]]

def get_config_pairs(wf_name):
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

# Better practice: no duplicate equivalent vCPU allocation in the config pairs

step_names = ['cold', 'read', 'compute', 'write']

def eq_vcpu_alloc(mem, num_func):
    num_vcpu = mem / 1792
    # num_vcpu = math.ceil(mem / 1792)
    # num_vcpu = math.floor(mem / 1792)
    # num_vcpu = max(1, num_vcpu)
    return round(num_vcpu * num_func, 1)

def io_func(x, a, b):
    return a / x + b

def io_func2(x, a, b, c):   # io_func2 is for parent relavent read
    return a / x[0] + b * x[1] + c

def comp_func(x, a, b, c, d):
    return a / x + b * np.log(x) / x + c / x**2 + d

'''
StagePerfModel records the parameter distributions of a stage's performance model
'''
class StagePerfModel:
    def __init__(self, stage_id, stage_name, default_input_size=1024) -> None:
        assert isinstance(stage_name, str)
        assert isinstance(stage_id, int) and stage_id >= 0
        self.stage_name = stage_name
        self.stage_id = stage_id

        self.allow_parallel = True
        self.has_parent = False

        assert isinstance(default_input_size, int) and default_input_size > 0
        self.default_input_size = default_input_size  # MB

        self.cold_params_avg = []  # random variable
        self.read_params_avg = []  # A/d + B, d is the equivalent vCPU allocation
        self.compute_params_avg = []  # A/d - B*log(d)/d + C/d**2 + D
        self.write_params_avg = []  # A/d + B
        self.read_cov_avg = []  # covariance matrix
        self.compute_cov_avg = []
        self.write_cov_avg = []

        self.can_intra_parallel = [True, True, True]  # stands for read, compute, write
        self.parent_relavent = False  # only use for not allow parallel and related to parent stage

        # Reduce the dimension of the parameters from 8 to 5, excluding cold start
        # By merging the parameters of read, compute, and write as follows:
        # allow_parallel: a/d + b/(kd) + c*log(x)/x + e/x**2 + f, x can be d or kd
        # not allow_parallel: a/k + b*d + c*log(k)/k + e/k**2 + f, 
        self.x_coeff = 0  # the coefficient of 1/d or 1/k in the stage, x can be d or kd
        self.kd_d_coeff = 0  # the coefficient of 1/(kd) or d in the stage
        self.logx_coeff = 0  # the coefficient of log(x)/x in the stage, x can be d or kd
        self.x2_coeff = 0  # the coefficient of 1/x**2 in the stage, x can be d or kd
        self.const_coeff = 0  # the constant coefficient in the stage

    def update_allow_parallel(self, allow_parallel) -> None:
        assert isinstance(allow_parallel, bool)
        self.allow_parallel = allow_parallel
    
    def update_has_parent(self, has_parent) -> None:
        assert isinstance(has_parent, bool)
        self.has_parent = has_parent

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

        # print('Training performance model for %s' % self.stage_name)

        # For scheduling delay and cold start, just a random variable
        y_s = np.array(stage_profile['cold'])
        num_epochs = y_s.shape[0]
        assert num_epochs >= 2  
        num_epochs -= 1  # Remove the first cold start epoch
        y_s = y_s[1:][:,:,0].reshape(-1)  # Only consider warm start
        self.cold_params_avg = y_s

        y_r = np.array(stage_profile['read'])[1:][:,:,0].reshape(-1)  # Only use the average time data
        y_c = np.array(stage_profile['compute'])[1:][:,:,0].reshape(-1)
        y_w = np.array(stage_profile['write'])[1:][:,:,0].reshape(-1)

        if self.allow_parallel:
            # kd is the equivalent vCPU allocation, d is the number of functions
            d = np.array([num_func for mem, num_func in config_pairs] * num_epochs)
            kd = np.array([eq_vcpu_alloc(mem, num_func) for mem, num_func in config_pairs] * num_epochs)

            # Use non-linear least squares to fit the parameters for average time
            # Read
            popt1, pcov1 = scipy_opt.curve_fit(io_func, d, y_r)
            y_ = io_func(d, popt1[0], popt1[1])
            err1 = (y_ - y_r) / y_r
            popt2, pcov2 = scipy_opt.curve_fit(io_func, kd, y_r)
            y_ = io_func(kd, popt2[0], popt2[1])
            err2 = (y_ - y_r) / y_r
            # Choose the better one
            s_err1 = np.mean(np.abs(err1))  # abs mean error
            s_err2 = np.mean(np.abs(err2))
            m_err1 = np.mean(err1)
            m_err2 = np.mean(err2)
            if s_err1 < s_err2:
                self.can_intra_parallel[0] = False
                self.read_params_avg = popt1
                self.read_cov_avg = pcov1
            else:
                self.can_intra_parallel[0] = True
                self.read_params_avg = popt2
                self.read_cov_avg = pcov2
            # print('Read')
            # print('d error avg:', err1)
            # print('kd error avg:', err2)
            # print('d abs mean error:', s_err1)
            # print('kd abs mean error:', s_err2)
            # print('d mean error:', m_err1)
            # print('kd mean error:', m_err2)
            # print('--------------------------------')

            # Compute
            popt1, pcov1 = scipy_opt.curve_fit(comp_func, d, y_c)
            y_ = comp_func(d, popt1[0], popt1[1], popt1[2], popt1[3])
            err1 = (y_ - y_c) / y_c
            popt2, pcov2 = scipy_opt.curve_fit(comp_func, kd, y_c)
            y_ = comp_func(kd, popt2[0], popt2[1], popt2[2], popt2[3])
            err2 = (y_ - y_c) / y_c
            # Choose the better one
            s_err1 = np.mean(np.abs(err1))
            s_err2 = np.mean(np.abs(err2))
            m_err1 = np.mean(err1)
            m_err2 = np.mean(err2)
            if s_err1 < s_err2 and abs(m_err1) < abs(m_err2):
                self.can_intra_parallel[1] = False
                self.compute_params_avg = popt1
                self.compute_cov_avg = pcov1
            else:
                self.can_intra_parallel[1] = True
                self.compute_params_avg = popt2
                self.compute_cov_avg = pcov2
            # print('Compute')
            # print('d error avg:', err1)
            # print('kd error avg:', err2)
            # print('d abs mean error:', s_err1)
            # print('kd abs mean error:', s_err2)
            # print('d mean error:', m_err1)
            # print('kd mean error:', m_err2)
            # print('--------------------------------')
            
            # Write
            popt1, pcov1 = scipy_opt.curve_fit(io_func, d, y_w)
            y_ = io_func(d, popt1[0], popt1[1])
            err1 = (y_ - y_w) / y_w
            popt2, pcov2 = scipy_opt.curve_fit(io_func, kd, y_w)
            y_ = io_func(kd, popt2[0], popt2[1])
            err2 = (y_ - y_w) / y_w
            # Choose the better one
            s_err1 = np.mean(np.abs(err1))
            s_err2 = np.mean(np.abs(err2))
            m_err1 = np.mean(err1)
            m_err2 = np.mean(err2)
            if s_err1 < s_err2:
                self.can_intra_parallel[2] = False
                self.write_params_avg = popt1
                self.write_cov_avg = pcov1
            else:
                self.can_intra_parallel[2] = True
                self.write_params_avg = popt2
                self.write_cov_avg = pcov2
            # print('Write')
            # print('d error avg:', err1)
            # print('kd error avg:', err2)
            # print('d abs mean error:', s_err1)
            # print('kd abs mean error:', s_err2)
            # print('d mean error:', m_err1)
            # print('kd mean error:', m_err2)
            # print('--------------------------------')
            # print('Intra parallel:', self.can_intra_parallel)
            # print('--------------------------------')

            # Compute the coefficients
            if self.can_intra_parallel[0]:
                self.kd_d_coeff += self.read_params_avg[0]
            else:
                self.x_coeff += self.read_params_avg[0]
            if self.can_intra_parallel[1]:
                self.kd_d_coeff += self.compute_params_avg[0]
            else:
                self.x_coeff += self.compute_params_avg[0]
            if self.can_intra_parallel[2]:
                self.kd_d_coeff += self.write_params_avg[0]
            else:
                self.x_coeff += self.write_params_avg[0]
            self.logx_coeff += self.compute_params_avg[1]
            self.x2_coeff += self.compute_params_avg[2]
            self.const_coeff += self.read_params_avg[1] + self.compute_params_avg[3] + \
                                self.write_params_avg[1]
            
            # Compute the error for the stage
            y_actual = y_r + y_c + y_w + y_s
            y_pred = self.x_coeff / d + self.kd_d_coeff / kd + self.const_coeff + np.mean(y_s)
            if self.can_intra_parallel[1]:
                y_pred += self.logx_coeff * np.log(kd) / kd + self.x2_coeff / kd**2
            else:
                y_pred += self.logx_coeff * np.log(d) / d + self.x2_coeff / d**2
            err = (y_pred - y_actual) / y_actual
            s_err = np.mean(np.abs(err))
            m_err = np.mean(err)
            # print('Stage Error:', err)
            print('Stage {} mean abs error:'.format(self.stage_id), '%.2f'%(s_err*100), '%')
            print('Stage {} mean error:'.format(self.stage_id), '%.2f'%(m_err*100), '%')
            
        else:
            # k is the vCPU allocation
            # k_d means the read time may be related to the parent stage's number of functions
            k = np.array([eq_vcpu_alloc(mem, 1) for mem, num_func in config_pairs] * num_epochs)
            k_d = np.array([[eq_vcpu_alloc(mem, 1), num_func] for mem, num_func in config_pairs] * num_epochs)

            # Use non-linear least squares to fit the parameters for average time
            # Read
            popt1, pcov1 = scipy_opt.curve_fit(io_func, k, y_r)
            y_ = io_func(k, popt1[0], popt1[1])
            err1 = (y_ - y_r) / y_r
            popt2, pcov2 = scipy_opt.curve_fit(io_func2, k_d.T, y_r)
            y_ = io_func2(k_d.T, popt2[0], popt2[1], popt2[2])
            err2 = (y_ - y_r) / y_r
            # # Choose the better one
            s_err1 = np.mean(np.abs(err1))  # abs mean error
            s_err2 = np.mean(np.abs(err2))
            m_err1 = np.mean(err1)
            m_err2 = np.mean(err2)
            if s_err1 < s_err2 or self.has_parent == False:
                self.parent_relavent = False
                self.read_params_avg = popt1
                self.read_cov_avg = pcov1
            else:
                self.parent_relavent = True
                self.read_params_avg = popt2
                self.read_cov_avg = pcov2
            # print('Read')
            # print('k error avg:', err1)
            # print('k_d error avg:', err2)
            # print('k abs mean error:', s_err1)
            # print('k_d abs mean error:', s_err2)
            # print('k mean error:', m_err1)
            # print('k_d mean error:', m_err2)
            # print('--------------------------------')
            # print('Parent relavent:', self.parent_relavent)
            # print('--------------------------------')

            # Compute, directly use k to fit
            popt1, pcov1 = scipy_opt.curve_fit(comp_func, k, y_c)
            y_ = comp_func(k, popt1[0], popt1[1], popt1[2], popt1[3])
            err1 = (y_ - y_c) / y_c
            s_err1 = np.mean(np.abs(err1))  # abs mean error
            m_err1 = np.mean(err1)
            self.compute_params_avg = popt1
            self.compute_cov_avg = pcov1
            # print('Compute')
            # print('k error avg:', err1)
            # print('k abs mean error:', s_err1)
            # print('k mean error:', m_err1)
            # print('--------------------------------')

            # Write, directly use k to fit
            popt1, pcov1 = scipy_opt.curve_fit(io_func, k, y_w)
            y_ = io_func(k, popt1[0], popt1[1])
            if y_w[0] > 1e-6:  # Avoid divide by zero, typically happens at the last stage's write
                err1 = (y_ - y_w) / y_w
                s_err1 = np.mean(np.abs(err1))  # abs mean error
                m_err1 = np.mean(err1)
            else:
                err1 = np.zeros(y_w.shape)
                s_err1 = 0
                m_err1 = 0
            self.write_params_avg = popt1
            self.write_cov_avg = pcov1
            # print('Write')
            # print('k error avg:', err1)
            # print('k abs mean error:', s_err1)
            # print('k mean error:', m_err1)
            # print('--------------------------------')

            # Compute the coefficients
            self.x_coeff += self.read_params_avg[0] + self.compute_params_avg[0] + \
                            self.write_params_avg[0]
            if self.parent_relavent:
                self.kd_d_coeff += self.read_params_avg[1]
                self.const_coeff += self.read_params_avg[2]
            else:
                self.const_coeff += self.read_params_avg[1]
            self.logx_coeff += self.compute_params_avg[1]
            self.x2_coeff += self.compute_params_avg[2]
            self.const_coeff += self.compute_params_avg[3] + self.write_params_avg[1]

            # Compute the error for the stage
            y_actual = y_r + y_c + y_w + y_s
            y_pred = self.x_coeff / k + self.kd_d_coeff * k_d.T[1] + self.const_coeff + np.mean(y_s) + \
                    self.logx_coeff * np.log(k) / k + self.x2_coeff / k**2
            err = (y_pred - y_actual) / y_actual
            s_err = np.mean(np.abs(err))
            m_err = np.mean(err)
            # print('Stage Error:', err)
            print('Stage {} mean abs error:'.format(self.stage_id), '%.2f'%(s_err*100), '%')
            print('Stage {} mean error:'.format(self.stage_id), '%.2f'%(m_err*100), '%')
        
        print()

    def predict(self, num_vcpu, num_func, mode='latency', parent_d=0, cold_percent=60, input_size = 1024) -> float:
        # input_size uses MB as unit
        assert num_vcpu > 0 and num_vcpu <= 10
        assert num_func > 0
        assert mode in ['latency', 'cost']

        k = eq_vcpu_alloc(num_vcpu*1792, 1)
        kd = eq_vcpu_alloc(num_vcpu*1792, num_func)
        d = num_func
        x = [1.0/d, 1.0/kd, np.log(d)/d, 1.0/d**2, 1.0]
        if self.allow_parallel:
            if self.can_intra_parallel[1]:
                x[2] = np.log(kd)/kd
                x[3] = 1.0/kd**2
        else:
            x = [1.0/k, parent_d, np.log(k)/k, 1.0/k**2, 1.0]
            if not self.parent_relavent:
                x[1] = 0
        
        params = self.params()
        pred = np.dot(params[1:], x)
        if input_size != 1024:
            pred *= input_size / self.default_input_size
        if mode == 'latency':
            pred += np.percentile(self.cold_params_avg, cold_percent)
            return pred
        else:
            # 1792 / 1024 * 0.0000000167 * 1000
            return (pred * num_func * num_vcpu * 2.9225  + 0.02 * num_func) / 100000

    def predict_tile(self, config, profile_path, num_samples, tile=95):
        mem, num_func = config  # Note that the config should be in config_pairs
        assert config in config_pairs
        num_vcpu = mem / 1792
        assert num_vcpu > 0 and num_vcpu <= 10
        assert num_func > 0

        kd = eq_vcpu_alloc(mem, num_func)
        d = num_func
        x = [1.0/d, 1.0/kd, np.log(d)/d, 1.0/d**2, 1.0]

        # Only used for allow_parallel stages
        if self.can_intra_parallel[1]:
            x[2] = np.log(kd)/kd
            x[3] = 1.0/kd**2
        
        sample_params = self.sample_offline(num_samples)
        preds = []
        for i in range(num_samples):
            preds.append(np.dot(sample_params[i][1:], x) + sample_params[i][0])
        pred = np.percentile(np.array(preds), tile)

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
        
        actual = np.percentile(actuals, tile)
        err = (pred - actual) / actual
        print('Predicted', tile, err)

    def params(self, cold_percent=60):
        cold_coeff = np.percentile(self.cold_params_avg, cold_percent)
        res = np.array([cold_coeff, self.x_coeff, self.kd_d_coeff, self.logx_coeff,
                        self.x2_coeff, self.const_coeff])
        return res

    def sample_offline(self, num_samples):
        assert isinstance(num_samples, int) and num_samples > 0
        # Sample for num_samples times
        res = {'cold': [], 'read': [], 'compute': [], 'write': []}
        # seed_val = int(time.time())
        seed_val = 0
        rng = np.random.default_rng(seed=seed_val)
        res['cold'] = rng.choice(self.cold_params_avg, num_samples)
        res['read'] = rng.multivariate_normal(self.read_params_avg, self.read_cov_avg, 
                                              num_samples)
        res['compute'] = rng.multivariate_normal(self.compute_params_avg, 
                                                 self.compute_cov_avg, 
                                                 num_samples)
        res['write'] = rng.multivariate_normal(self.write_params_avg, 
                                               self.write_cov_avg,
                                               num_samples)
        # Organize into coefficient form
        coeffs = np.zeros((num_samples, 6))
        coeffs[:, 0] = res['cold']
        if self.allow_parallel:
            if self.can_intra_parallel[0]:
                coeffs[:, 2] += res['read'].T[0]  # 1/(kd)
            else:
                coeffs[:, 1] += res['read'].T[0]  # 1/d
            if self.can_intra_parallel[1]:
                coeffs[:, 2] += res['compute'].T[0]
            else:
                coeffs[:, 1] += res['compute'].T[0]
            if self.can_intra_parallel[2]:
                coeffs[:, 2] += res['write'].T[0]
            else:
                coeffs[:, 1] += res['write'].T[0]
            coeffs[:, 3] += res['compute'].T[1]  # log(x)/x
            coeffs[:, 4] += res['compute'].T[2]  # 1/x**2
            coeffs[:, 5] += res['read'].T[1] + res['compute'].T[3] + res['write'].T[1]
        else:
            coeffs[:, 1] += res['read'].T[0] + res['compute'].T[0] + res['write'].T[0]
            if self.parent_relavent:
                coeffs[:, 2] += res['read'].T[1]
                coeffs[:, 5] += res['read'].T[2]
            else:
                coeffs[:, 5] += res['read'].T[1]
            coeffs[:, 3] += res['compute'].T[1]
            coeffs[:, 4] += res['compute'].T[2]

        return coeffs

    def generate_func_code(self, mode, var, param, parent_id=-1, solver_type='scipy') -> str:
        assert isinstance(parent_id, int)
        assert mode in ['latency', 'cost']
        assert isinstance(var, str) and isinstance(param, str)
        assert solver_type == 'scipy'

        # 6 param indices and 2 var indices for each stage
        # 0: cold, 1: x, 2: kd/d, 3: log(x)/x, 4: 1/x**2, 5: const
        # 0: var d, 1: var k

        s = ''
        offset = 0 if solver_type == 'scipy' else 1
        cold_param = param + '[%d]'%(self.stage_id*6 + offset)
        x_param = param + '[%d]'%(self.stage_id*6 + 1 + offset)
        kd_d_param = param + '[%d]'%(self.stage_id*6 + 2 + offset)
        logx_param = param + '[%d]'%(self.stage_id*6 + 3 + offset)
        x2_param = param + '[%d]'%(self.stage_id*6 + 4 + offset)
        const_param = param + '[%d]'%(self.stage_id*6 + 5 + offset)

        var_d = var + '[%d]'%(self.stage_id*2 + offset)
        if not self.allow_parallel:
            var_d = '1'
        var_k = var + '[%d]'%(self.stage_id*2 + 1 + offset)
        var_x = ''
        if self.can_intra_parallel[1]:
            var_x = var_k + '*' + var_d
        else:
            var_x = var_d
        var_x = '(' + var_x + ')'

        log_method = 'np.log'

        if self.allow_parallel:
            s += x_param + '/' + var_d + ' + '
            s += kd_d_param + '/' + '(' + var_k + '*' + var_d + ')' + ' + '
            s += logx_param + '*' + log_method + var_x + '/' + var_x + ' + '
            s += x2_param + '/' + var_x + '**2' + ' + '
            s += const_param
        else:
            s += x_param + '/' + var_k + ' + ' 
            if self.parent_relavent and parent_id >= 0:
                var_pd = var + '[%d]'%(parent_id*2)  # parent d
                s += kd_d_param + '*' + var_pd + ' + '
            s += logx_param + '*' + log_method + '(' + var_k + ')' + '/' + var_k + ' + '
            s += x2_param + '/' + var_k + '**2' + ' + '
            s += const_param
        if mode == 'latency':
            s = cold_param + ' + ' + s
        else:
            # 1792 / 1024 * 0.0000000167 * 1000 = 0.000029225 
            # 1000 is to convert from ms to s
            # We multiply 1e5 to the cost to make it more readable
            # s = cold_param + ' / 2 + ' + s
            s = '(' + s + ') * ' + var_k + ' * ' + var_d + ' * 2.9225 + 0.02 * ' + var_d
        return s

    def __str__(self):
        return self.stage_name
    
    def __getstate__(self):
        self_dict = self.__dict__.copy()
        return self_dict
    
    def __del__(self):
        pass
