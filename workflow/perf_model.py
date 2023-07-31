import numpy as np
import matplotlib.pyplot as plt
import math
import time
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
    return a / x + b * np.log(x) / x + c / x**2 + d

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
        self.default_input_size = default_input_size  # MB

        self.cold_params_avg = []  # random variable
        self.read_params_avg = []  # A/d + B, d is the equivalent vCPU allocation
        self.compute_params_avg = []  # A/d - B*log(d)/d + C/d**2 + D
        self.write_params_avg = []  # A/d + B
        self.read_cov_avg = []
        self.compute_cov_avg = []
        self.write_cov_avg = []
        self.can_intra_parallel = [True, True, True]  # stands for read, compute, write
        self.parent_relavent = False  # only use for not allow parallel and related to parent stage

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

        print('Training performance model for %s' % self.stage_name)

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
            print('Read')
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
            print('d error avg:', err1)
            print('kd error avg:', err2)
            print('d abs mean error:', s_err1)
            print('kd abs mean error:', s_err2)
            print('d mean error:', m_err1)
            print('kd mean error:', m_err2)
            print('--------------------------------')

            # Compute
            print('Compute')
            popt1, pcov1 = scipy_opt.curve_fit(comp_func, d, y_c)
            y_ = comp_func(d, popt1[0], popt1[1], popt1[2], popt1[3])
            err1 = (y_ - y_c) / y_c
            popt2, pcov2 = scipy_opt.curve_fit(comp_func, kd, y_c)
            print('comp params', popt2)
            y_ = comp_func(kd, popt2[0], popt2[1], popt2[2], popt2[3])
            err2 = (y_ - y_c) / y_c
            # Choose the better one
            s_err1 = np.mean(np.abs(err1))
            s_err2 = np.mean(np.abs(err2))
            m_err1 = np.mean(err1)
            m_err2 = np.mean(err2)
            if s_err1 < s_err2:
                self.can_intra_parallel[1] = False
                self.compute_params_avg = popt1
                self.compute_cov_avg = pcov1
            else:
                self.can_intra_parallel[1] = True
                self.compute_params_avg = popt2
                self.compute_cov_avg = pcov2
            print('d error avg:', err1)
            print('kd error avg:', err2)
            print('d abs mean error:', s_err1)
            print('kd abs mean error:', s_err2)
            print('d mean error:', m_err1)
            print('kd mean error:', m_err2)
            print('--------------------------------')
            
            # Write
            print('Write')
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
            print('d error avg:', err1)
            print('kd error avg:', err2)
            print('d abs mean error:', s_err1)
            print('kd abs mean error:', s_err2)
            print('d mean error:', m_err1)
            print('kd mean error:', m_err2)
            print('--------------------------------')
            print('Intra parallel:', self.can_intra_parallel)
            print('--------------------------------')
        else:
            # k is the vCPU allocation
            # k_d means the read time may be related to the parent stage's number of functions
            k = np.array([eq_vcpu_alloc(mem, 1) for mem, num_func in config_pairs] * num_epochs)
            k_d = np.array([eq_vcpu_alloc(mem, 1) / num_func for mem, num_func in config_pairs] * num_epochs)

            # Use non-linear least squares to fit the parameters for average time
            # Read
            print('Read')
            popt1, pcov1 = scipy_opt.curve_fit(io_func, k, y_r)
            y_ = io_func(k, popt1[0], popt1[1])
            err1 = (y_ - y_r) / y_r
            popt2, pcov2 = scipy_opt.curve_fit(io_func, k_d, y_r)
            y_ = io_func(k_d, popt2[0], popt2[1])
            err2 = (y_ - y_r) / y_r
            # # Choose the better one
            s_err1 = np.mean(np.abs(err1))  # abs mean error
            s_err2 = np.mean(np.abs(err2))
            m_err1 = np.mean(err1)
            m_err2 = np.mean(err2)
            if s_err1 < s_err2:
                self.parent_relavent = False
                self.read_params_avg = popt1
                self.read_cov_avg = pcov1
            else:
                self.parent_relavent = True
                self.read_params_avg = popt2
                self.read_cov_avg = pcov2
            print('k error avg:', err1)
            print('k_d error avg:', err2)
            print('k abs mean error:', s_err1)
            print('k_d abs mean error:', s_err2)
            print('k mean error:', m_err1)
            print('k_d mean error:', m_err2)
            print('--------------------------------')
            print('Parent relavent:', self.parent_relavent)
            print('--------------------------------')

            # Compute, directly use k to fit
            print('Compute')
            popt1, pcov1 = scipy_opt.curve_fit(comp_func, k, y_c)
            print('comp params', popt2)
            y_ = comp_func(k, popt1[0], popt1[1], popt1[2], popt1[3])
            err1 = (y_ - y_c) / y_c
            s_err1 = np.mean(np.abs(err1))  # abs mean error
            m_err1 = np.mean(err1)
            self.compute_params_avg = popt1
            self.compute_cov_avg = pcov1
            print('k error avg:', err1)
            print('k abs mean error:', s_err1)
            print('k mean error:', m_err1)
            print('--------------------------------')

            # Write, directly use k to fit
            print('Write')
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
            print('k error avg:', err1)
            print('k abs mean error:', s_err1)
            print('k mean error:', m_err1)
            print('--------------------------------')
        print('Training finished')
        print('\n\n')

    # private method, for test
    def __visualize(self, profile_path) -> None:
        assert isinstance(profile_path, str) and profile_path.endswith('.json')
        profile = None
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
        y_s = y_s[1:][:,:,0].reshape(-1)  # Only consider warm start
        self.cold_params_avg = y_s

        y_r = np.array(stage_profile['read'])[1:][:,:,0].reshape(-1)  # Only use the average time data
        y_c = np.array(stage_profile['compute'])[1:][:,:,0].reshape(-1)
        y_w = np.array(stage_profile['write'])[1:][:,:,0].reshape(-1)

        d = np.array([num_func for mem, num_func in config_pairs]*num_epochs)
        kd = np.array([eq_vcpu_alloc(mem, num_func) for mem, num_func in config_pairs]*num_epochs)
        popt1, pcov1 = scipy_opt.curve_fit(io_func, kd, y_r)
        popt2, pcov2 = scipy_opt.curve_fit(comp_func, d, y_c)
        popt3, pcov3 = scipy_opt.curve_fit(io_func, kd, y_w)
        sigma1 = np.sqrt(np.diag(pcov1))
        sigma2 = np.sqrt(np.diag(pcov2))
        sigma3 = np.sqrt(np.diag(pcov3))
        print('Read:', popt1, sigma1)
        print('Compute:', popt2, sigma2)
        print('Write:', popt3, sigma3)

        num_samples = 2000

        ps1 = np.random.multivariate_normal(popt1, pcov1, num_samples)
        ps2 = np.random.multivariate_normal(popt2, pcov2, num_samples)
        ps3 = np.random.multivariate_normal(popt3, pcov3, num_samples)
        y_1_l = []
        y_2_l = []
        y_3_l = []
        ps1_max = np.max(ps1, axis=0)
        ps1_min = np.min(ps1, axis=0)
        ps2_max = np.max(ps2, axis=0)
        ps2_min = np.min(ps2, axis=0)
        ps3_max = np.max(ps3, axis=0)
        ps3_min = np.min(ps3, axis=0)

        x = np.linspace(0.9, 32, 100)

        f = 0.1

        y_1 = io_func(x, popt1[0], popt1[1])
        y_1_max = io_func(x, popt1[0]+f*sigma1[0], popt1[1]+f*sigma1[1])
        y_1_min = io_func(x, popt1[0]-f*sigma1[0], popt1[1]-f*sigma1[1])
        for i in range(num_samples):
            y_1_l.append(io_func(x, ps1[i][0], ps1[i][1]))
        y_1_l = np.array(y_1_l)
        y_1_l_max = io_func(x, ps1_max[0], ps1_max[1])
        y_1_l_min = io_func(x, ps1_min[0], ps1_min[1])
        
        y_2 = comp_func(x, popt2[0], popt2[1], popt2[2], popt2[3])
        y_2_max = comp_func(x, popt2[0]+f*sigma2[0], popt2[1]+f*sigma2[1], popt2[2]+f*sigma2[2], popt2[3]+f*sigma2[3])
        y_2_min = comp_func(x, popt2[0]-f*sigma2[0], popt2[1]-f*sigma2[1], popt2[2]-f*sigma2[2], popt2[3]-f*sigma2[3])
        for i in range(num_samples):
            y_2_l.append(comp_func(x, ps2[i][0], ps2[i][1], ps2[i][2], ps2[i][3]))
        y_2_l = np.array(y_2_l)
        y_2_l_max = comp_func(x, ps2_max[0], ps2_max[1], ps2_max[2], ps2_max[3])
        y_2_l_min = comp_func(x, ps2_min[0], ps2_min[1], ps2_min[2], ps2_min[3])
        
        y_3 = io_func(x, popt3[0], popt3[1])
        y_3_max = io_func(x, popt3[0]+f*sigma3[0], popt3[1]+f*sigma3[1])
        y_3_min = io_func(x, popt3[0]-f*sigma3[0], popt3[1]-f*sigma3[1])
        for i in range(num_samples):
            y_3_l.append(io_func(x, ps3[i][0], ps3[i][1]))
        y_3_l = np.array(y_3_l)
        y_3_l_max = io_func(x, ps3_max[0], ps3_max[1])
        y_3_l_min = io_func(x, ps3_min[0], ps3_min[1])
        
        y_p = y_1 + y_2 + y_3 + np.mean(y_s, axis=0)
        y_p_max = y_1_max + y_2_max + y_3_max + np.mean(y_s, axis=0)
        y_p_min = y_1_min + y_2_min + y_3_min + np.mean(y_s, axis=0)
        y_l = y_1_l + y_2_l + y_3_l + np.mean(y_s, axis=0)
        y_l_max = y_1_l_max + y_2_l_max + y_3_l_max + np.mean(y_s, axis=0)
        y_l_min = y_1_l_min + y_2_l_min + y_3_l_min + np.mean(y_s, axis=0)

        y_1_d = io_func(kd, popt1[0], popt1[1])
        y_2_d = comp_func(d, popt2[0], popt2[1], popt2[2], popt2[3])
        y_3_d = io_func(kd, popt3[0], popt3[1])

        y_d = y_1_d + y_2_d + y_3_d + y_s
        y_ = y_r + y_c + y_w + y_s
        err = (y_d - y_) / y_
        print('Error:', err)
        print('Mean Abs Error:', np.mean(np.abs(err)))
        print('Mean Error:', np.mean(err))

        font_size = 20
        plt.rc('font',**{'size': font_size})
        fig_size = (10, 6)
        fig, axes = plt.subplots(nrows=1, ncols=1, sharey=False, figsize=fig_size)

        mode = 'time' # 'error' or 'time'
        if mode == 'error':
            axes.scatter(kd, err*100)
            axes.set_ylabel('Error (%)')
        else:
            s = axes.scatter(kd, y_, zorder=3, label='samples')
            # axes.scatter(kd, y_d)
            l0, = axes.plot(x, y_p, 'r', label='pred_mean')
            for i in range(num_samples):
                l1, = axes.plot(x, y_l[i], 'darkorange', label='pred w/ cov', alpha=0.1, zorder=1)
            l2, = axes.plot(x, y_l_max, 'royalblue', label='pred_max', zorder=2)
            fig.legend(handles=[s, l0, l1], ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.0), fontsize=font_size)
            # plt.plot(x, y_p_max, 'royalblue')
            # plt.plot(x, y_p_min, 'royalblue')
            # plt.scatter(kd, y_s)
            axes.set_ylabel('Time (s)')
        axes.set_xlabel('k*d (eq vCPU)')
        plt.savefig('tmp.png')
    
    def visualize(self, profile_path) -> None:
        self.__visualize(profile_path)

    def predict(self, mem, num_func, mode='latency', parent_d=None, input_size=None) -> float:
        # input_size uses MB as unit
        assert (isinstance(mem, int) or isinstance(mem, float)) and mem >= 128 and mem <= 10240
        assert isinstance(num_func, int) and num_func > 0
        assert mode in ['latency', 'cost']
        if input_size is not None:
            assert (isinstance(input_size, int) or isinstance(input_size, float)) and input_size > 0
        if parent_d is not None:
            assert isinstance(parent_d, int) and parent_d > 0

        x = [num_func, num_func, num_func]
        if self.allow_parallel:
            for i in range(3):
                if self.can_intra_parallel[i]:
                    x[i] = eq_vcpu_alloc(mem, num_func)
        else:
            for i in range(3):
                x[i] = eq_vcpu_alloc(mem, 1)
            if self.parent_relavent:
                x[0] = x[0] / parent_d
        pred = 0

        # For scheduling delay and cold start, just a random sample
        pred += io_func(x[0], self.read_params_avg[0], self.read_params_avg[1])
        pred += np.mean(comp_func(x[1], self.compute_params_avg[0], self.compute_params_avg[1], 
                        self.compute_params_avg[2], self.compute_params_avg[3]))
        pred += np.mean(io_func(x[2], self.write_params_avg[0], self.write_params_avg[1]))
        if input_size is not None:
            pred *= input_size / self.default_input_size
        if mode == 'latency':
            pred += np.random.choice(self.cold_params_avg)
            return pred
        else:
            return pred * num_func * mem / 1024 * 0.0000000167 + 0.2 * num_func / 1000000

    def params(self) -> dict:
        res = {}
        res['cold'] = np.mean(self.cold_params_avg)
        res['read'] = self.read_params_avg.tolist()
        res['compute'] = self.compute_params_avg.tolist()
        res['write'] = self.write_params_avg.tolist()
        return res

    def sample_offline(self, num_samples):
        assert isinstance(num_samples, int) and num_samples > 0
        # Sample for num_samples times
        res = {'cold': [], 'read': [], 'compute': [], 'write': []}
        seed_val = int(time.time())
        rng = np.random.default_rng(seed=seed_val)
        res['cold'] = rng.choice(self.cold_params_avg, num_samples).reshape(-1, 1).tolist()
        res['read'] = rng.multivariate_normal(self.read_params_avg, self.read_cov_avg, 
                                              num_samples).tolist()
        res['compute'] = rng.multivariate_normal(self.compute_params_avg, 
                                                 self.compute_cov_avg, 
                                                 num_samples).tolist()
        res['write'] = rng.multivariate_normal(self.write_params_avg, 
                                               self.write_cov_avg,
                                               num_samples).tolist()
        return res

    def generate_func_code(self, idx, mode) -> (str, int):
        assert isinstance(idx, int) and idx >= 0
        assert mode in ['latency', 'cost']

        func_str = ''
        if mode == 'latency':
            pass
        else:
            pass
        
        return func_str, idx

    def __str__(self):
        return self.stage_name
    
    def __getstate__(self):
        self_dict = self.__dict__.copy()
        return self_dict
    
    def __del__(self):
        pass


if __name__ == '__main__':
    m = StagePerfModel('stage1')
    # m.visualize('../profiles/ML-Pipeline_profile.json')
    m.train('../profiles/ML-Pipeline_profile.json')
    # print(m.sample_offline(3))