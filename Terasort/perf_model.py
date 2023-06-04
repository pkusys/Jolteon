import scipy as sp
import numpy as np
import math
from scipy.optimize import leastsq

import matplotlib
from matplotlib import pyplot as plt


# def func(p, x):
#     A, B, C, D = p
#     return A / x + B * np.log2(x) / x + C * np.log2(x) + D

# def error(p, x, y):
#     return func(p, x) - y

def jct_func(p, x1, x2):
    A1, B1, C1, D1, A2, B2, C2, D2, E = p
    return A1 / x1 + B1 * np.log2(x1) / x1 + C1 * np.log2(x1) + D1 * x1 + \
        A2 / x2 + B2 * np.log2(x2) / x2 + C2 * np.log2(x2) + D2 * x2 + E

def jct_error(p, x1, x2, y):
    return jct_func(p, x1, x2) - y

def cost_func(p, x1, x2):
    A, B1, C1, D1, B2, C2, D2, E = p
    return A + B1 * np.log2(x1) + C1 * np.log2(x1) * x1 + D1 * x1 + \
        B2 * np.log2(x2) + C2 * np.log2(x2) * x2 + D2 * x2 + E * x1 * x2

def cost_error(p, x1, x2, y):
    return cost_func(p, x1, x2) - y

X1 = np.array([16, 32, 64, 128, 
               16, 32, 64, 128, 256, 
               16, 32, 64, 128, 256, 
               16, 32, 64, 128, 256, 
               16, 32, 64, 128])

X2 = np.array([16, 16, 16, 16, 
               32, 32, 32, 32, 32, 
               64, 64, 64, 64, 64, 
               128, 128, 128, 128, 128, 
               256, 256, 256, 256])

JCT = np.array([58.08, 42.40, 46.69, 63.52, 
                42.50, 33.03, 26.90, 34.20, 46.14, 
                34.77, 29.09, 21.92, 26.14, 34.24, 
                39.04, 26.98, 24.03, 25.10, 39.33, 
                49.37, 40.78, 38.14, 38.97])
JCT = JCT * 1000

COST = np.array([0.035694, 0.036401, 0.044074, 0.064353, 
                 0.030663, 0.037110, 0.042052, 0.066541, 0.103807, 
                 0.029365, 0.036843, 0.053156, 0.082648, 0.142785, 
                 0.035102, 0.047266, 0.067903, 0.116004, 0.260699, 
                 0.043998, 0.070705, 0.095836, 0.193230])
COST = COST * 1e5
P0 = np.array([1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000])
# P0 = np.array([1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000])
Para = leastsq(jct_error, P0, args=(X1, X2, JCT))

# print("Fitting Parameters:", Para[0])
print("A1=", Para[0][0], "B1=", Para[0][1], "C1=", Para[0][2], "D1=", Para[0][3],
      "A2=", Para[0][4], "B2=", Para[0][5], "C2=", Para[0][6], "D2=", Para[0][7], "E=", Para[0][8])
# print actual y and predicted y
print(JCT)
print(jct_func(Para[0], X1, X2))
print("error=", jct_error(Para[0], X1, X2, JCT))
print("percentage error=", jct_error(Para[0], X1, X2, JCT) / JCT * 100)