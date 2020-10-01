import numpy as np
from dapper import *

def dxdt(x):
    # SIR and C
    beta = 1  # Infection rate
    gamma = .1

    S = x[0]
    I = x[1]
    R = x[2]
    C = x[3]
    N = S + I + R

    d = np.zeros(4)
    d[0] = -beta * C * S * I / N # S dot
    d[1] = beta * C * S * I / N - gamma * I# I dot
    d[2] = gamma * I

    d[3] = 0 # constant C
    return d

def Dyn(E, t0, dt):
    def step(x0):
        return rk4(lambda t, x: dxdt(x), x0, t0, dt)

    # Ensemble case
    for n in range(E.shape[1]):
        E[:, n] = step(E[:, n])

    return E