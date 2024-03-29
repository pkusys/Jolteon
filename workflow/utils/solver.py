import math
import json
import numpy as np
import scipy.optimize as scipy_opt
from scipy.optimize import NonlinearConstraint
from deprecation import deprecated
from .basic_class import MyQueue, MyProcess
from multiprocessing import Manager, Queue

'''
PCP: Probabilistic (Chance) Constrained Programming
The PCPSolver uses the sample approximation approach to solve the PCP problem as described in 
the paper "A Sample Approximation Approach for Optimization with Probabilistic Constraints".
'''
class PCPSolver:
    def __init__(self, num_X, objective, constraint, bound, 
                 obj_params, cons_params, 
                 constraint_2=None,
                 risk=0.05, approx_risk=0, confidence_error=0.001, 
                 ftol=1, 
                 k_configs=[0.5, 1, 1.5, 2, 2.5, 3, 4],
                 d_configs=[1, 4, 8, 16, 32],
                 bound_type='latency',
                 need_probe=None,
                 probe_depth=4, 
                 solver_info={'optlib': 'scipy', 'method': 'SLSQP'}):
        assert isinstance(num_X, int) and num_X > 0
        assert callable(objective) and callable(constraint)
        if constraint_2 is not None:
            assert callable(constraint_2)
        # We consider bound as one positive number (SLO or budget)
        assert (isinstance(bound, float) or isinstance(bound, int)) and bound > 0
        assert isinstance(obj_params, list) and isinstance(cons_params, list)

        assert isinstance(risk, float) and risk > 0 and risk < 1
        assert (isinstance(approx_risk, int) and approx_risk == 0 or approx_risk == 1) or \
            (isinstance(approx_risk, float) and approx_risk > 0 and approx_risk < 1)
        assert isinstance(confidence_error, float) and confidence_error > 0 and \
            confidence_error < 1
        assert ftol > 0
        assert isinstance(solver_info, dict) and 'optlib' in solver_info and \
            'method' in solver_info

        self.num_X = num_X
        self.objective = objective
        self.constraint = constraint
        self.constraint_2 = constraint_2
        self.bound = bound
        self.obj_params = obj_params
        self.cons_params = cons_params

        # User-defined risk level (epsilon) for constraint satisfaction (e.g., 0.01 or 0.05)
        self.risk = risk 
        # Approximated risk level (alpha), a parameter for the sample approximation problem
        self.approx_risk = approx_risk
        # Confidence error (delta) for the lower bound property of the ground-truth optimal 
        # objective value or the feasibility of the solution or both, 
        # depending on the relationship between epsilon and alpha, default to 0.01
        self.confidence_error = confidence_error

        # Used for solving
        self.ftol = ftol
        # Used for probing
        self.k_configs = k_configs
        self.d_configs = d_configs

        self.bound_type = bound_type
        self.need_probe = need_probe
        self.probe_depth = probe_depth

        # Solver information for the sample approximation problem
        self.solver_info = solver_info

    @staticmethod
    def sample_size(num_stages, risk, approx_risk, confidence_error) -> int:
        # {0.5, 1, 1.5, 2, 3, 4} as the intra-function resource space, so the size is 8
        # {4, 8, 16, 32} as the parallelism space, so the size is 4
        search_space_size = (7 * 4)**(num_stages // 2)  # num_X / 2 stages
        min_abs_tol = 1e-2
        if math.isclose(risk, approx_risk, abs_tol=min_abs_tol):
            return math.ceil(0.5 / min_abs_tol**2 * math.log((1 + search_space_size) / confidence_error))
        elif risk < approx_risk:
            return math.ceil(0.5 / (approx_risk - risk)**2 * (-math.log(confidence_error)))
        else: 
            return math.ceil(0.5 / (risk - approx_risk)**2 * math.log(search_space_size / confidence_error))

    '''
    Solve the sample approximation problem
    @param init_vals: initial values for the optimization variables, single value or list or numpy array
    @param x_bound: optional bounds for each optimization variable, single tuple or list of tuples, 
                    default to (0.5, None)
    @return: a dictionary containing the solution status, the optimal objective value, 
            the constraint values, and the optimal solution
    '''
    def solve(self, init_vals=None, x_bound=None) -> dict:
        assert self.solver_info['optlib'] == 'scipy'
        assert self.solver_info['method'] == 'SLSQP'
            
        x0 = np.ones(self.num_X) * 2  # initial guess
        if init_vals is not None:
            if isinstance(init_vals, int) or isinstance(init_vals, float):
                x0 = np.ones(self.num_X) * init_vals
            elif isinstance(init_vals, list) and len(init_vals) == self.num_X:
                # ssert all(isinstaance(x, int) or isinstance(x, float) for x in init_vals)
                x0 = np.array(init_vals)
            elif isinstance(init_vals, np.ndarray) and init_vals.shape == (self.num_X, ):
                x0 = init_vals

        X_bounds = [(0.5, None) for _ in range(self.num_X)] # optional bounds for each x
        if x_bound is not None:
            if isinstance(x_bound, tuple) and len(x_bound) == 2:
                X_bounds = [x_bound for _ in range(self.num_X)]
            elif isinstance(x_bound, list) and len(x_bound) == self.num_X:
                X_bounds = x_bound
            elif isinstance(x_bound, list) and len(x_bound) == 2:
                # [0] is for parallelism, [1] is for intra-function resource
                X_bounds = []
                for _ in range(self.num_X // 2):
                    X_bounds.append(x_bound[0])
                    X_bounds.append(x_bound[1])

        obj_params = np.array(self.obj_params)
        cons_params = np.array(self.cons_params).T
        nonlinear_constraints = NonlinearConstraint(lambda x: self.constraint(x, cons_params, self.bound), -np.inf, 0)
        if self.constraint_2 is not None:
            b = self.bound if self.bound_type == 'latency' else 0
            nonlinear_constraints_2 = NonlinearConstraint(lambda x: self.constraint_2(x, obj_params, b), -np.inf, 0)
            nonlinear_constraints = [nonlinear_constraints_2, nonlinear_constraints]
        
        res = scipy_opt.minimize(lambda x: self.objective(x, obj_params), x0, 
                                    method=self.solver_info['method'],
                                    bounds=X_bounds, 
                                    constraints=nonlinear_constraints,
                                    options={'ftol': self.ftol, 'disp': False})
        
        solve_res = {}
        solve_res['status'] = res.success
        solve_res['obj_val'] = res.fun
        solve_res['cons_val'] = self.constraint(res.x, cons_params, self.bound)
        solve_res['x'] = res.x

        return solve_res

    '''
    Solve the sample approximation problem iteratively to tolerate a certain number of 
    constraint violations
    '''
    def iter_solve(self, init_vals=None, x_bound=None):
        while True:
            res = self.solve(init_vals=init_vals, x_bound=x_bound)
            if res['status']:
                break
            else:
                cons_val = np.array(res['cons_val'])
                ratio_not_satisfied = np.sum(cons_val > self.ftol) / len(cons_val)
                if ratio_not_satisfied < self.risk:
                    break
                else:
                    print('bound:', self.bound, 'ratio:', ratio_not_satisfied)
                    self.bound += self.ftol * 4

        return res

    def probe(self, d_init, k_init):
        # assume init is within the feasible region
        d_pos = []
        k_pos = []
        d_config = np.array(self.d_configs)
        k_config = np.array(self.k_configs)
        for d in d_init:
            mask = (d_config == d)
            if np.any(mask):
                j = np.where(mask)[0][0]
                d_pos.append(j)
        for k in k_init:
            mask = (k_config == k)
            if np.any(mask):
                j = np.where(mask)[0][0]
                k_pos.append(j)
        
        d_pos = np.array(d_pos)
        k_pos = np.array(k_pos)
        x_pos = np.zeros(self.num_X, dtype=int)
        x_pos[0::2] = d_pos
        x_pos[1::2] = k_pos

        cons_params = np.array(self.cons_params).T

        searched = set()

        need_pos = []
        if self.need_probe is not None:
            need_pos = [i for i in range(self.num_X) if self.need_probe[i]]

        def bfs(x_pos, max_depth=4):
            q = MyQueue()
            searched.add(tuple(x_pos))
            q.push([x_pos, 0])

            best_pos = x_pos.copy()
            best_x = np.zeros(self.num_X)
            best_x[0::2] = d_config[best_pos[0::2]]
            best_x[1::2] = k_config[best_pos[1::2]]
            best_obj = self.objective(best_x, self.obj_params)
            best_cons = self.constraint(best_x, self.obj_params, self.bound)

            steps = [-1, 1]

            while len(q) > 0:
                p = q.pop()

                x = np.zeros(self.num_X)
                x[0::2] = d_config[p[0][0::2]]
                x[1::2] = k_config[p[0][1::2]]
                cons = self.constraint(x, cons_params, self.bound)
                cons = np.percentile(cons, 100 * (1 - self.risk))
                obj = self.objective(x, self.obj_params)
                # print('x:', x, 'obj:', obj, 'cons:', cons)

                if best_cons < 0:  # tight bound
                    if cons < 0 and cons > best_cons and obj < best_obj:
                        best_pos = p[0].copy()
                        best_obj = obj
                        best_cons = cons
                else:  # find a feasible solution first
                    if cons < best_cons:
                        best_pos = p[0].copy()
                        best_obj = obj
                        best_cons = cons

                if p[1] < max_depth:
                    for t in range(self.num_X):
                        if len(need_pos) > 0 and t not in need_pos:
                            continue
                        if t % 2 == 0:  # d
                            config = d_config
                        else:  # k
                            config = k_config
                        for s in steps:
                            new_x_pos = p[0].copy()
                            new_x_pos[t] += s
                            if new_x_pos[t] < 0 or new_x_pos[t] >= len(config) or (t % 2 == 0 and new_x_pos[t] == 0):
                                continue
                            if tuple(new_x_pos) in searched:
                                continue
                            searched.add(tuple(new_x_pos))
                            q.push([new_x_pos, p[1] + 1])

            return best_pos

        x = np.zeros(self.num_X)
        x[0::2] = d_config[x_pos[0::2]]
        x[1::2] = k_config[x_pos[1::2]]
        cons = self.constraint(x, cons_params, self.bound)
        cons = np.percentile(cons, 100 * (1 - self.risk))
        feasible = cons < 0
        old_x_pos = x_pos.copy()
        while not feasible:  # find a feasible solution first
            x_pos = bfs(x_pos, self.probe_depth)
            if x_pos.tolist() == old_x_pos.tolist():  # no improvement
                break
            old_x_pos = x_pos.copy()
            x = np.zeros(self.num_X)
            x[0::2] = d_config[x_pos[0::2]]
            x[1::2] = k_config[x_pos[1::2]]
            cons = self.constraint(x, self.obj_params, self.bound)
            cons = self.constraint(x, cons_params, self.bound)
            cons = np.percentile(cons, 100 * (1 - self.risk))
            feasible = cons < 0
        
        # find the best solution
        best_pos = bfs(x_pos, self.probe_depth)
        best_d = d_config[best_pos[0::2]].tolist()
        best_k = k_config[best_pos[1::2]].tolist()

        return best_d, best_k

    def get_vals(self, d, k, tile=95):
        assert len(d) == self.num_X // 2
        assert len(k) == self.num_X // 2
        x = np.zeros(self.num_X)
        x[0::2] = d
        x[1::2] = k

        cons_params = np.array(self.cons_params).T

        obj = self.objective(x, cons_params)
        obj = np.percentile(obj, tile)
        cons = self.constraint(x, cons_params, self.bound) + self.bound
        cons = np.percentile(cons, tile)
        
        if self.bound_type == 'latency':
            return cons, obj
        else:
            return obj, cons

    @deprecated
    def probe_parallel(self, d_init, k_init):
        # assume init is within the feasible region
        d_pos = []
        k_pos = []
        d_config = np.array(self.d_configs)
        k_config = np.array(self.k_configs)
        for d in d_init:
            mask = (d_config == d)
            if np.any(mask):
                j = np.where(mask)[0][0]
                d_pos.append(j)
        for k in k_init:
            mask = (k_config == k)
            if np.any(mask):
                j = np.where(mask)[0][0]
                k_pos.append(j)
        
        d_pos = np.array(d_pos)
        k_pos = np.array(k_pos)
        x_pos = np.zeros(self.num_X, dtype=int)
        x_pos[0::2] = d_pos
        x_pos[1::2] = k_pos

        cons_params = np.array(self.cons_params).T

        searched = Manager().dict()

        need_pos = []
        if self.need_probe is not None:
            need_pos = [i for i in range(self.num_X) if self.need_probe[i]]

        def bfs(x_pos, max_depth=8):
            q = MyQueue()
            # searched.add(tuple(x_pos))
            searched[tuple(x_pos)] = 1
            q.push([x_pos, 0])

            best_pos = x_pos.copy()
            best_x = np.zeros(self.num_X)
            best_x[0::2] = d_config[best_pos[0::2]]
            best_x[1::2] = k_config[best_pos[1::2]]
            best_obj = self.objective(best_x, self.obj_params)
            best_cons = self.constraint(best_x, self.obj_params, self.bound)

            steps = [-1, 1]

            while len(q) > 0:
                p = q.pop()

                x = np.zeros(self.num_X)
                x[0::2] = d_config[p[0][0::2]]
                x[1::2] = k_config[p[0][1::2]]
                cons = self.constraint(x, cons_params, self.bound)
                cons = np.percentile(cons, 100 * (1 - self.risk))
                obj = self.objective(x, self.obj_params)

                if best_cons < 0:  # tight bound
                    if cons < 0 and cons > best_cons and obj < best_obj:
                        best_pos = p[0].copy()
                        best_obj = obj
                        best_cons = cons
                else:  # find a feasible solution first
                    if cons < best_cons:
                        best_pos = p[0].copy()
                        best_obj = obj
                        best_cons = cons

                if p[1] < max_depth:
                    for t in range(self.num_X):
                        if len(need_pos) > 0 and t not in need_pos:
                            continue
                        if t % 2 == 0:  # d
                            config = d_config
                        else:  # k
                            config = k_config
                        for s in steps:
                            new_x_pos = p[0].copy()
                            new_x_pos[t] += s
                            if new_x_pos[t] < 0 or new_x_pos[t] >= len(config) or (t % 2 == 0 and new_x_pos[t] == 0):
                                continue
                            if tuple(new_x_pos) in searched:
                                continue
                            searched[tuple(new_x_pos)] = 1
                            q.push([new_x_pos, p[1] + 1])

            return best_pos, best_obj, best_cons

        threads = []

        pq = Queue()
        for i in range(len(need_pos)):
            new_x_pos = x_pos.copy()
            new_x_pos[need_pos[i]] += 1
            if (need_pos[i] % 2 == 0 and new_x_pos[need_pos[i]] < len(d_config)):
                threads.append(MyProcess(target=bfs, args=new_x_pos, queue=pq))
            if (need_pos[i] % 2 == 1 and new_x_pos[need_pos[i]] < len(k_config)):
                threads.append(MyProcess(target=bfs, args=new_x_pos, queue=pq))
            new_x_pos = x_pos.copy()
            new_x_pos[need_pos[i]] -= 1
            if new_x_pos[need_pos[i]] >= 0 or (new_x_pos[need_pos[i]] % 2 == 0 and new_x_pos[need_pos[i]] > 0):
                threads.append(MyProcess(target=bfs, args=new_x_pos))
        for t in threads:
            t.start()
        # get the results of all threads
        best_pos = x_pos.copy()
        best_x = np.zeros(self.num_X)
        best_x[0::2] = d_config[best_pos[0::2]]
        best_x[1::2] = k_config[best_pos[1::2]]
        best_obj = self.objective(best_x, self.obj_params)
        best_cons = self.constraint(best_x, self.obj_params, self.bound)
        for t in threads:
            t.join()
        while not pq.empty():
            pos, obj, cons = t._result
            if cons > best_cons:
                best_pos = pos
                best_obj = obj
                best_cons = cons
        best_d = d_config[best_pos[0::2]].tolist()
        best_k = k_config[best_pos[1::2]].tolist()

        return best_d, best_k

    '''
    The generic constraint satisfaction is deprecated due to the nondeterministic behavior of
    existing solvers for the logical control flow in the constraint function (e.g, if-else and 
    for-loop are not supported, numpy.where is undefined behavior in scipy.optimize.minimize)
    '''
    @deprecated
    def sample_constraint_satifaction(self, param_samples, x, num_samples):
        assert isinstance(param_samples, np.ndarray) and len(param_samples.shape) == 2 and \
            num_samples == param_samples.shape[0]
        num_satisfied = np.sum(np.where(self.constraint(param_samples, x) <= self.bound, 1, 0))
        return num_satisfied - num_samples * (1 - self.approx_risk)


if __name__ == '__main__':
    num_stages = 8
    risk = 0.05
    approx_risk = 0
    confidence_error = 0.01
    sample_size = PCPSolver.sample_size(num_stages, risk, approx_risk, confidence_error)
    print('sample size:', sample_size)