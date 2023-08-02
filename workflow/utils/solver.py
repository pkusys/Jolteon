import math
import json
import numpy as np
import scipy.optimize as scipy_opt
from scipy.optimize import NonlinearConstraint
import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory

'''
PCP: Probabilistic (Chance) Constrained Programming
The PCPSolver uses the sample approximation approach to solve the PCP problem as described in 
the paper "A Sample Approximation Approach for Optimization with Probabilistic Constraints".
'''
class PCPSolver:
    def __init__(self, num_X, objective, constraint,
                 bound, obj_param_path, cons_param_path, 
                 constraint_2=None,
                 risk=0.01, approx_risk=0, confidence_error=0.01, 
                 solver_info={'optlib': 'scipy', 'method': 'SLSQP'}) -> None:
        assert isinstance(num_X, int) and num_X > 0
        assert callable(objective) and callable(constraint)
        if constraint_2 is not None:
            assert callable(constraint_2)
        # We consider bound as one positive number (SLO or budget)
        assert (isinstance(bound, float) or isinstance(bound, int)) and bound > 0
        assert isinstance(obj_param_path, str) and obj_param_path.endswith('.json')
        assert isinstance(cons_param_path, str) and cons_param_path.endswith('.json')

        assert isinstance(risk, float) and risk > 0 and risk < 1
        assert (isinstance(approx_risk, int) and approx_risk == 0 or approx_risk == 1) or \
            (isinstance(approx_risk, float) and approx_risk > 0 and approx_risk < 1)
        assert isinstance(confidence_error, float) and confidence_error > 0 and \
            confidence_error < 1
        assert isinstance(solver_info, dict) and 'optlib' in solver_info and \
            'method' in solver_info

        self.num_X = num_X
        self.objective = objective
        self.constraint = constraint
        self.constraint_2 = constraint_2
        self.bound = bound

        # User-defined risk level (epsilon) for constraint satisfaction (e.g., 0.01 or 0.05)
        self.risk = risk 
        # Approximated risk level (alpha), a parameter for the sample approximation problem
        self.approx_risk = approx_risk
        # Confidence error (delta) for the lower bound property of the ground-truth optimal 
        # objective value or the feasibility of the solution or both, 
        # depending on the relationship between epsilon and alpha, default to 0.01
        self.confidence_error = confidence_error
        # Solver information for the sample approximation problem
        # TODO: add gurobi, use scipy as the default solver
        self.solver_info = solver_info
        # Sample size
        # self.num_samples = self.sample_size()
        self.num_samples = 3  # for testing
        self.num_fused_samples = self.num_samples  
        # TODO: implement the fusion of samples

    def sample_size(self) -> int:
        # Orion picks {min, 1024, 1792, max}, so the size is 4
        search_space_size = (4 * 6)**(self.num_X // 2)  # num_X / 2 stages, 4 parallelisms, 6 resource spaces (1024, 2048, 4096, 1792, 3584, 7168)
        # 10G/128M is the resource space size, 900 is the parallelism space size
        min_abs_tol = 1e-2
        if math.isclose(self.risk, self.approx_risk, abs_tol=min_abs_tol):
            return math.ceil(0.5 / min_abs_tol**2 * math.log((1 + search_space_size) / self.confidence_error))
        elif self.risk < self.approx_risk:
            return math.ceil(0.5 / (self.approx_risk - self.risk)**2 * (-math.log(self.confidence_error)))
        else: 
            return math.ceil(0.5 / (self.risk - self.approx_risk)**2 * math.log(search_space_size / self.confidence_error))

    def get_sample_size(self) -> int:
        return self.num_samples

    def get_fused_sample_size(self) -> int:
        return self.num_fused_samples

    def load_params(self, param_path):
        with open(param_path, 'r') as f:
            params = json.load(f)
        params = np.array(params)
        return params
    
    '''
    The generic constraint satisfaction is deprecated due to the nondeterministic behavior of
    existing solvers for the logical control flow in the constraint function (e.g, if-else and 
    for-loop are not supported, numpy.where is undeterministis in scipy.optimize.minimize)
    '''
    def sample_constraint_satifaction(self, param_samples, x):
        assert isinstance(param_samples, np.ndarray) and len(param_samples.shape) == 2 and \
            self.num_samples == param_samples.shape[0] 
        num_satisfied = np.sum(np.where(self.constraint(param_samples, x) <= self.bound, 1, 0))
        return num_satisfied - self.num_samples * (1 - self.approx_risk)

    def solve(self):
        if self.solver_info['optlib'] == 'scipy':
            assert self.solver_info['method'] in ['SLSQP', 'COBYLA', 'trust-constr', 'L-BFGS-B', 'TNC']
            
            x0 = np.ones(self.num_X)
            # TODO: assign reasonable initial values for x
            obj_params = self.load_params(self.obj_param_path)
            cons_params = self.load_params(self.cons_param_path)
            nonlinear_constraints = NonlinearConstraint(lambda x: self.constraint(x, cons_params, self.bound), -np.inf, 0)
            x_bounds = [(0, None) for _ in range(self.num_X)] # optional
            x_bounds = None
            res = scipy_opt.minimize(lambda x: self.objective(x, obj_params), x0, 
                                     method=self.solver_info['method'], bounds=x_bounds, 
                                     constraints=nonlinear_constraints)
            print('Optimization result:')
            print('Status:', res.success)
            print('Objective value:', res.fun)
            print('Decision variables:', res.x)
        elif self.solver_info['optlib'] == 'pyomo':
            assert self.solver_info['method'] in ['gurobi', 'ipopt', 'snopt', 'knitro']
            # self.solver = SolverFactory(self.solver_info['method'])


# if __name__ == '__main__':
#     svr = PCPSolver(4, test_func, lambda x: x[0] + x[1] - 1, 1, '.json', '.json', 
#                     risk=0.05, approx_risk=0, confidence_error=0.05)
#     print(svr.get_sample_size())