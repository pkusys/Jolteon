# Test different optimizers, used for debugging

import numpy as np
import scipy.optimize as scipy_opt
# from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
from pyomo.environ import *

############################# PYOMO #############################
'''
# Number of samples for approximation
num_samples = 100

# Generate random samples for the constraint
samples = np.random.rand(num_samples, 4)

# Create a Pyomo ConcreteModel
model = ConcreteModel()

# Define decision variables
model.I = RangeSet(1, 3)
model.J = RangeSet(1, num_samples)
model.x = Var(model.I, domain=NonNegativeReals)

# Define the objective function using a function call
def objective_function(x):
    return x[1] + 2*x[2] + 3/x[3]

model.objective = Objective(expr=objective_function(model.x), sense=minimize)

a_init = {(i, j): samples[j-1][i-1] for i in range(1, 5) for j in model.J}
model.a = Param(RangeSet(1, 4), model.J, initialize=a_init)

# Function for computing the constraint expression
def constraint_rule(m, j):
    return m.a[1, j]*m.x[1] + m.a[2, j]/m.x[2] + m.a[3, j]*pyomo.environ.log(m.x[3]) <= m.a[4, j]

model.constraint = Constraint(model.J, rule=constraint_rule)

# The chance constraint
# model.chance_constraint = Constraint(expr=summation(model.y) >= (1 - 0.05)*num_samples)

# Solve the optimization problem
solver = SolverFactory('ipopt')  # Cannot use ipopt, installation error
results = solver.solve(model)

# Print the results
print('Optimization result:')
print('Status:', results.solver.status)
print('Objective value:', value(model.objective))
print('Decision variables:')
for v in model.x:
    print(f'{v}: {value(model.x[v])}')
'''

############################# SCIPY #############################

# Number of samples for approximation
num_samples = 100

# Generate random samples for the constraint
# samples = np.random.rand(num_samples, 4)
samples = np.array([[0.1, 0.2, 0.3, 0.4],
                    [0.2, 0.3, 0.4, 0.5],
                    [0.3, 0.4, 0.5, 0.6],
                    [0.4, 0.5, 0.6, 0.7]])
samples = samples.T

# Objective function
def objective(x):
    return x[0] + 2*x[1] + 3/x[2]

# Define the constraint
def constraint(x, sample):
    return sample[0]*x[0] + sample[1]/x[1] - sample[2]*np.log(x[2]) - sample[3]

# Initialize decision variables
x0 = np.array([1.0, 1.0, 1.0])

# Nonlinear constraints
nonlinear_constraints = []
# for sample in samples.T:
#     nonlinear_constraints.append(NonlinearConstraint(lambda x: constraint(x, sample), -np.inf, 0))
nonlinear_constraints.append(NonlinearConstraint(lambda x: constraint(x, samples.T), -np.inf, 0))

x_bounds = [(5, None), (5, None), (5, None)]

# Solve the optimization problem
res = scipy_opt.minimize(objective, x0, method='SLSQP', bounds=x_bounds, constraints=nonlinear_constraints)

# Print the results
print('Optimization result:')
print('Status:', res.success)
print('Objective value:', res.fun)
print('Decision variables:', res.x)
