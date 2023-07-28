from utils.solver import PCPSolver
import numpy as np
import matplotlib.pyplot as plt
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
        self.default_input_size = default_input_size  # MB or GB

        self.cold_params_avg = []  # random variable
        self.read_params_avg = []  # A/d + B, d is the equivalent vCPU allocation
        self.compute_params_avg = []  # A/d - B*log(d)/d + C/d**2 + D
        self.write_params_avg = []  # A/d + B
        # self.cold_params_max = [] 
        # self.read_params_max = []
        # self.compute_params_max = []  
        # self.write_params_max = []
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

        print('Training performance model for stage %s' % self.stage_name)

        # For scheduling delay and cold start, just a random sample
        self.cold_params_avg = np.array(stage_profile['cold'])[:, :, 0].reshape(-1)
        # self.cold_params_max = np.array(stage_profile['cold'])[:, :, 1].reshape(-1)

        y_r = np.array(stage_profile['read'])
        y_c = np.array(stage_profile['compute'])
        y_w = np.array(stage_profile['write'])
        num_epochs = y_r.shape[0] 
        assert num_epochs % 2 == 1  # num_epochs should be an odd number

        if self.allow_parallel:
            # kd is the equivalent vCPU allocation, d is the number of functions
            d = np.array([num_func for mem, num_func in config_pairs] * num_epochs)
            kd = np.array([eq_vcpu_alloc(mem, num_func) for mem, num_func in config_pairs] * num_epochs)
            print('d:', d)
            print('kd:', kd)
            votes = [0 for _ in range(3)]  # 0: read, 1: compute, 2: write
            param_avg_1 = [None for _ in range(num_epochs)]
            param_avg_2 = [None for _ in range(num_epochs)]
            param_max_1 = [None for _ in range(num_epochs)]
            param_max_2 = [None for _ in range(num_epochs)]

            for i in range(num_epochs):
                print('Epoch %d' % i)
                # Use non-linear least square to fit the parameters
                # _1 is d, _2 is kd
                err1 = [None for _ in range(2)]
                err2 = [None for _ in range(2)]
                popt_avg_1 = [None for _ in range(3)]
                popt_avg_2 = [None for _ in range(3)]
                popt_max_1 = [None for _ in range(3)]
                popt_max_2 = [None for _ in range(3)]

                # Read
                print('Read')
                # Fit for average
                popt_avg_1[0], _ = scipy_opt.curve_fit(io_func, d, y_r[i][:, 0])
                y_ = io_func(d, popt_avg_1[0][0], popt_avg_1[0][1])
                err1[0] = (y_ - y_r[i][:, 0]) / y_r[i][:, 0]
                popt_avg_2[0], _ = scipy_opt.curve_fit(io_func, kd, y_r[i][:, 0])
                y_ = io_func(kd, popt_avg_2[0][0], popt_avg_2[0][1])
                err2[0] = (y_ - y_r[i][:, 0]) / y_r[i][:, 0]
                # # Fit for max
                # popt_max_1[0], _ = scipy_opt.curve_fit(io_func, d, y_r[i][:, 1])
                # y_ = io_func(d, popt_max_1[0][0], popt_max_1[0][1])
                # err1[1] = (y_ - y_r[i][:, 1]) / y_r[i][:, 1]
                # popt_max_2[0], _ = scipy_opt.curve_fit(io_func, kd, y_r[i][:, 1])
                # y_ = io_func(kd, popt_max_2[0][0], popt_max_2[0][1])
                # err2[1] = (y_ - y_r[i][:, 1]) / y_r[i][:, 1]
                # # Choose the better one
                # s_err1 = np.sum(np.abs(err1[0])) + np.sum(np.abs(err1[1]))
                # s_err2 = np.sum(np.abs(err2[0])) + np.sum(np.abs(err2[1]))
                s_err1 = np.sum(np.abs(err1[0]))
                s_err2 = np.sum(np.abs(err2[0]))
                if s_err1 < s_err2:
                    self.can_intra_parallel[0] = False
                    votes[0] += 1
                print('d error avg:', err1[0])
                print('kd error avg:', err2[0])
                print('d error mean:', s_err1 / len(err1[0]))
                print('kd error mean:', s_err2 / len(err2[0]))
                print('--------------------------------')

                # Compute
                print('Compute')
                # Fit for average
                popt_avg_1[1], _ = scipy_opt.curve_fit(comp_func, d, y_c[i][:, 0])
                y_ = comp_func(d, popt_avg_1[1][0], popt_avg_1[1][1], popt_avg_1[1][2], popt_avg_1[1][3])
                err1[0] = (y_ - y_c[i][:, 0]) / y_c[i][:, 0]
                popt_avg_2[1], _ = scipy_opt.curve_fit(comp_func, kd, y_c[i][:, 0])
                y_ = comp_func(kd, popt_avg_2[1][0], popt_avg_2[1][1], popt_avg_2[1][2], popt_avg_2[1][3])
                err2[0] = (y_ - y_c[i][:, 0]) / y_c[i][:, 0]
                # Fit for max
                popt_max_1[1], _ = scipy_opt.curve_fit(comp_func, d, y_c[i][:, 1])
                y_ = comp_func(d, popt_max_1[1][0], popt_max_1[1][1], popt_max_1[1][2], popt_max_1[1][3])
                err1[1] = (y_ - y_c[i][:, 1]) / y_c[i][:, 1]
                popt_max_2[1], _ = scipy_opt.curve_fit(comp_func, kd, y_c[i][:, 1])
                y_ = comp_func(kd, popt_max_2[1][0], popt_max_2[1][1], popt_max_2[1][2], popt_max_2[1][3])
                err2[1] = (y_ - y_c[i][:, 1]) / y_c[i][:, 1]
                # Choose the better one
                # s_err1 = np.sum(np.abs(err1[0])) + np.sum(np.abs(err1[1]))
                # s_err2 = np.sum(np.abs(err2[0])) + np.sum(np.abs(err2[1]))
                s_err1 = np.sum(np.abs(err1[0]))
                s_err2 = np.sum(np.abs(err2[0]))
                if s_err1 < s_err2:
                    # self.compute_params_avg.append(popt_avg_1)
                    # self.compute_params_max.append(popt_max_1)
                    # self.can_intra_parallel[1] = False
                    votes[1] += 1
                # else:
                #     self.compute_params_avg.append(popt_avg_2)
                #     self.compute_params_max.append(popt_max_2)
                #     self.can_intra_parallel[1] = True
                print('d error avg:', err1[0])
                print('kd error avg:', err2[0])
                print('d error mean:', s_err1 / len(err1[0]))
                print('kd error mean:', s_err2 / len(err2[0]))
                print('--------------------------------')
                
                # Write
                print('Write')
                # Fit for average
                popt_avg_1[2], _ = scipy_opt.curve_fit(io_func, d, y_w[i][:, 0])
                y_ = io_func(d, popt_avg_1[2][0], popt_avg_1[2][1])
                err1[0] = (y_ - y_w[i][:, 0]) / y_w[i][:, 0]
                popt_avg_2[2], _ = scipy_opt.curve_fit(io_func, kd, y_w[i][:, 0])
                y_ = io_func(kd, popt_avg_2[2][0], popt_avg_2[2][1])
                err2[0] = (y_ - y_w[i][:, 0]) / y_w[i][:, 0]
                # Fit for max
                popt_max_1[2], _ = scipy_opt.curve_fit(io_func, d, y_w[i][:, 1])
                y_ = io_func(d, popt_max_1[2][0], popt_max_1[2][1])
                err1[1] = (y_ - y_w[i][:, 1]) / y_w[i][:, 1]
                popt_max_2[2], _ = scipy_opt.curve_fit(io_func, kd, y_w[i][:, 1])
                y_ = io_func(kd, popt_max_2[2][0], popt_max_2[2][1])
                err2[1] = (y_ - y_w[i][:, 1]) / y_w[i][:, 1]
                # Choose the better one
                # s_err1 = np.sum(np.abs(err1[0])) + np.sum(np.abs(err1[1]))
                # s_err2 = np.sum(np.abs(err2[0])) + np.sum(np.abs(err2[1]))
                s_err1 = np.sum(np.abs(err1[0]))
                s_err2 = np.sum(np.abs(err2[0]))
                if s_err1 < s_err2:
                    # self.write_params_avg.append(popt_avg_1)
                    # self.write_params_max.append(popt_max_1)
                    # self.can_intra_parallel[2] = False
                    votes[2] += 1
                # else:
                #     self.write_params_avg.append(popt_avg_2)
                #     self.write_params_max.append(popt_max_2)
                #     self.can_intra_parallel[2] = True
                print('d error avg:', err1[0])
                print('kd error avg:', err2[0])
                print('d error mean:', s_err1 / len(err1[0]))
                print('kd error mean:', s_err2 / len(err2[0]))
                print('--------------------------------')

                param_avg_1[i] = popt_avg_1.copy()
                param_avg_2[i] = popt_avg_2.copy()
                param_max_1[i] = popt_max_1.copy()
                param_max_2[i] = popt_max_2.copy()

            print('Votes:', votes)
            if votes[0] >= num_epochs // 2 + 1:
                self.can_intra_parallel[0] = False
                for j in range(num_epochs):
                    self.read_params_avg.append(param_avg_1[j][0])
                    self.read_params_max.append(param_max_1[j][0])
            else:
                self.can_intra_parallel[0] = True
                for j in range(num_epochs):
                    self.read_params_avg.append(param_avg_2[j][0])
                    self.read_params_max.append(param_max_2[j][0])
            if votes[1] >= num_epochs // 2 + 1:
                self.can_intra_parallel[1] = False
                for j in range(num_epochs):
                    self.compute_params_avg.append(param_avg_1[j][1])
                    self.compute_params_max.append(param_max_1[j][1])
            else:
                self.can_intra_parallel[1] = True
                for j in range(num_epochs):
                    self.compute_params_avg.append(param_avg_2[j][1])
                    self.compute_params_max.append(param_max_2[j][1])
            if votes[2] >= num_epochs // 2 + 1:
                self.can_intra_parallel[2] = False
                for j in range(num_epochs):
                    self.write_params_avg.append(param_avg_1[j][2])
                    self.write_params_max.append(param_max_1[j][2])
            else:
                self.can_intra_parallel[2] = True
                for j in range(num_epochs):
                    self.write_params_avg.append(param_avg_2[j][2])
                    self.write_params_max.append(param_max_2[j][2])
        else:
            # k is the vCPU allocation
            # k_d means the read time may be related to the parent stage's number of functions
            k = np.array([eq_vcpu_alloc(mem, 1) for mem, num_func in config_pairs])
            k_d = np.array([eq_vcpu_alloc(mem, 1) / num_func for mem, num_func in config_pairs])
            print('k:', k)
            print('k_d:', k_d)
            votes = 0
            for i in range(num_epochs):
                # Use non-linear least square to fit the parameters
                # _1 is k, _2 is k_d
                err1 = [None for _ in range(2)]
                err2 = [None for _ in range(2)]

                # Read
                print('Read')
                # Fit for average
                popt_avg_1, _ = scipy_opt.curve_fit(io_func, k, y_r[i][:, 0])
                y_ = io_func(d, popt_avg_1[0], popt_avg_1[1])
                err1[0] = (y_ - y_r[i][:, 0]) / y_r[i][:, 0]
                popt_avg_2, _ = scipy_opt.curve_fit(io_func, k_d, y_r[i][:, 0])
                y_ = io_func(kd, popt_avg_2[0], popt_avg_2[1])
                err2[0] = (y_ - y_r[i][:, 0]) / y_r[i][:, 0]
                # Fit for max
                popt_max_1, _ = scipy_opt.curve_fit(io_func, k, y_r[i][:, 1])
                y_ = io_func(d, popt_max_1[0], popt_max_1[1])
                err1[1] = (y_ - y_r[i][:, 1]) / y_r[i][:, 1]
                popt_max_2, _ = scipy_opt.curve_fit(io_func, k_d, y_r[i][:, 1])
                y_ = io_func(kd, popt_max_2[0], popt_max_2[1])
                err2[1] = (y_ - y_r[i][:, 1]) / y_r[i][:, 1]
                # Choose the better one
                s_err1 = np.sum(np.abs(err1[0])) + np.sum(np.abs(err1[1]))
                s_err2 = np.sum(np.abs(err2[0])) + np.sum(np.abs(err2[1]))
                if s_err1 < s_err2:
                    self.read_params_avg.append(popt_avg_1)
                    self.read_params_max.append(popt_max_1)
                    self.parent_relavent = False
                else:
                    self.read_params_avg.append(popt_avg_2)
                    self.read_params_max.append(popt_max_2)
                    self.parent_relavent = True
                print('k error avg:', err1[0])
                print('k_d error avg:', err2[0])
                print('k error max:', err1[1])
                print('k_d error max:', err2[1])
                print('k error sum:', s_err1)
                print('k_d error sum:', s_err2)
                print('--------------------------------')

                # Compute, directly use k to fit
                print('Compute')
                # Fit for average
                popt_avg_1, _ = scipy_opt.curve_fit(comp_func, k, y_c[i][:, 0])
                y_ = comp_func(d, popt_avg_1[0], popt_avg_1[1], popt_avg_1[2], popt_avg_1[3])
                err1[0] = (y_ - y_c[i][:, 0]) / y_c[i][:, 0]
                popt_max_1, _ = scipy_opt.curve_fit(comp_func, k, y_c[i][:, 1])
                y_ = comp_func(d, popt_max_1[0], popt_max_1[1], popt_max_1[2], popt_max_1[3])
                err2[0] = (y_ - y_c[i][:, 1]) / y_c[i][:, 1]
                self.compute_params_avg.append(popt_avg_1)
                self.compute_params_max.append(popt_max_1)
                print('k error avg:', err1[0])
                print('k error max:', err1[1])
                print('--------------------------------')

                # Write, directly use k to fit
                print('Write')
                # Fit for average
                popt_avg_1, _ = scipy_opt.curve_fit(io_func, k, y_w[i][:, 0])
                # y_ = io_func(d, popt_avg_1[0], popt_avg_1[1])
                # err1[0] = (y_ - y_w[i][:, 0]) / y_w[i][:, 0]
                popt_max_1, _ = scipy_opt.curve_fit(io_func, k, y_w[i][:, 1])
                # y_ = io_func(d, popt_max_1[0], popt_max_1[1])
                # err2[0] = (y_ - y_w[i][:, 1]) / y_w[i][:, 1]
                self.write_params_avg.append(popt_avg_1)
                self.write_params_max.append(popt_max_1)
                # print('k error avg:', err1[0])
                # print('k error max:', err1[1])
                # print('--------------------------------')

        self.read_params_avg = np.array(self.read_params_avg).T
        self.read_params_max = np.array(self.read_params_max).T
        self.compute_params_avg = np.array(self.compute_params_avg).T
        self.compute_params_max = np.array(self.compute_params_max).T
        self.write_params_avg = np.array(self.write_params_avg).T
        self.write_params_max = np.array(self.write_params_max).T

    # private method, for test
    def visualize(self, profile_path) -> None:
        assert isinstance(profile_path, str) and profile_path.endswith('.json')
        profile = None
        with open(profile_path, 'r') as f:
            profile = json.load(f)
        assert isinstance(profile, dict) and self.stage_name in profile
        stage_profile = profile[self.stage_name]
        assert isinstance(stage_profile, dict) and 'cold' in stage_profile and \
            'read' in stage_profile and 'compute' in stage_profile and \
            'write' in stage_profile
        
        y_s = np.array(stage_profile['cold'])[:,:,0]
        num_epochs = y_s.shape[0]
        y_s = y_s.reshape(-1)
        y_r = np.array(stage_profile['read'])[:,:,0].reshape(-1)
        y_c = np.array(stage_profile['compute'])[:,:,0].reshape(-1)
        y_w = np.array(stage_profile['write'])[:,:,0].reshape(-1)
        d = np.array([num_func for mem, num_func in config_pairs]*num_epochs)
        kd = np.array([eq_vcpu_alloc(mem, num_func) for mem, num_func in config_pairs]*num_epochs)
        popt1, pcov1 = scipy_opt.curve_fit(io_func, kd, y_r)
        popt2, pcov2 = scipy_opt.curve_fit(comp_func, d, y_c)
        popt3, pcov3 = scipy_opt.curve_fit(io_func, kd, y_w)

        x = np.linspace(0.5, 30, 100)

        y_1 = io_func(x, popt1[0], popt1[1])
        y_2 = comp_func(x, popt2[0], popt2[1], popt2[2], popt2[3])
        y_3 = io_func(x, popt3[0], popt3[1])
        y_p = y_1 + y_2 + y_3 + np.mean(y_s, axis=0)

        y_1_d = io_func(kd, popt1[0], popt1[1])
        y_2_d = comp_func(d, popt2[0], popt2[1], popt2[2], popt2[3])
        y_3_d = io_func(kd, popt3[0], popt3[1])

        y_d = y_1_d + y_2_d + y_3_d + y_s
        y_ = y_r + y_c + y_w + y_s
        err = (y_d - y_) / y_
        print('Error:', err)
        print('Mean Abs Error:', np.mean(np.abs(err)))
        print('Mean Error:', np.mean(err))

        mode = 'time' # 'error' or 'time'
        if mode == 'error':
            plt.scatter(kd, err*100)
        else:
            plt.scatter(kd, y_)
            plt.scatter(kd, y_d)
            plt.plot(x, y_p, 'r')
            # plt.scatter(kd, y_p)
        plt.ylabel(mode)
        plt.savefig('tmp.png')

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
            pred += np.mean(io_func(d, self.read_params_avg[0], self.read_params_avg[1]))
            pred += np.mean(comp_func(d, self.compute_params_avg[0], self.compute_params_avg[1], 
                           self.compute_params_avg[2], self.compute_params_avg[3]))
            pred += np.mean(io_func(d, self.write_params_avg[0], self.write_params_avg[1]))
            if input_size is not None:
                pred *= input_size / self.default_input_size
            return pred * num_func * mem / 1024 * 0.0000000167 + 0.2 * num_func / 1000000

    def generate_func_code(self, idx, mode) -> (str, int):
        assert isinstance(idx, int) and idx >= 0
        assert mode in ['latency', 'cost']

        func_str = ''
        if mode == 'latency':
            pass
        else:
            pass
        
        return func_str, idx

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


if __name__ == '__main__':
    m = StagePerfModel('stage2')
    m.visualize('../profiles/ML-Pipeline_profile.json')