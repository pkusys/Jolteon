import scipy as sp
import numpy as np
import math
from scipy.optimize import leastsq

def func(p, x):
    A, B, C, D = p
    return A / x + B * np.log2(x) / x + C * np.log2(x) + D

def error(p, x, y):
    return func(p, x) - y

X = np.array([16, 32, 64, 128, 256])
Y = np.array([1881.289062, 3228.828125, 5163.984375, 9691.59375, 19370.25])
P0 = np.array([1, 1, 1, 1])
Para = leastsq(error, P0, args=(X, Y))

# print("Fitting Parameters:", Para[0])
print("A=", Para[0][0], "B=", Para[0][1], "C=", Para[0][2], "D=", Para[0][3])
# print actual y and predicted y
print(Y)
print(func(Para[0], X))
print("error=", error(Para[0], X, Y))
print("percentage error=", error(Para[0], X, Y) / Y * 100)