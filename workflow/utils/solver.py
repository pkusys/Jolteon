import math
import json
import numpy as np
import scipy.optimize as scipy_opt
from scipy.optimize import NonlinearConstraint
from deprecation import deprecated
# import pyomo.environ as pyo
# from pyomo.environ import *
# from pyomo.opt import SolverFactory

'''
PCP: Probabilistic (Chance) Constrained Programming
The PCPSolver uses the sample approximation approach to solve the PCP problem as described in 
the paper "A Sample Approximation Approach for Optimization with Probabilistic Constraints".
'''
class PCPSolver:
    def __init__(self, num_X, objective, constraint,
                 bound, obj_params, cons_params, 
                 constraint_2=None,
                 risk=0.05, approx_risk=0, confidence_error=0.001, 
                 solver_info={'optlib': 'scipy', 'method': 'SLSQP'}) -> None:
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
        # Solver information for the sample approximation problem
        self.solver_info = solver_info

    @staticmethod
    def sample_size(num_stages, risk, approx_risk, confidence_error) -> int:
        # {1024, 1792, 2048, 4096} as the intra-function resource space, so the size if 4
        # {1, 2, 4, 8, 16, 32} as the parallelism space, so the size is 6
        search_space_size = (4 * 6)**(num_stages // 2)  # num_X / 2 stages
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
                    default to (0.75, None)
    @return: a dictionary containing the solution status, the optimal objective value, 
            the constraint values, and the optimal solution
    '''
    def solve(self, init_vals=None, x_bound=None) -> dict:
        if self.solver_info['optlib'] == 'scipy':
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

            X_bounds = [(0.75, None) for _ in range(self.num_X)] # optional bounds for each x
            if x_bound is not None:
                if isinstance(x_bound, tuple) and len(x_bound) ==2:
                    X_bounds = [x_bound for _ in range(self.num_X)]
                elif isinstance(x_bound, list) and len(x_bound) == self.num_X:
                    X_bounds = x_bound

            obj_params = np.array(self.obj_params)
            cons_params = np.array(self.cons_params).T
            nonlinear_constraints = NonlinearConstraint(lambda x: self.constraint(x, cons_params, self.bound), -np.inf, 0)
            if self.constraint_2 is not None:
                nonlinear_constraints_2 = NonlinearConstraint(lambda x: self.constraint_2(x, obj_params), -np.inf, 0)
                nonlinear_constraints = [nonlinear_constraints, nonlinear_constraints_2]
            
            res = scipy_opt.minimize(lambda x: self.objective(x, obj_params), x0, 
                                     method=self.solver_info['method'], bounds=X_bounds, 
                                     constraints=nonlinear_constraints)
            
            solve_res = {}
            solve_res['status'] = res.success
            solve_res['obj_val'] = res.fun
            solve_res['cons_val'] = self.constraint(res.x, cons_params, self.bound)
            solve_res['x'] = res.x

            return solve_res
        else:
            raise NotImplementedError
    
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


# if __name__ == "__main__":
#     risk = 0.05
#     approx_risk = 0.0
#     confidence_error = 0.001
#     search_space_size = (4 * 6)**4
#     num_samples = math.ceil(0.5 / (risk - approx_risk)**2 * math.log(search_space_size / confidence_error))
#     print(num_samples)