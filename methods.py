'''
Methods for multiscale program
'''

import numpy as np
import scipy as scp
import scipy.integrate as sint
import matplotlib.pyplot as plt
import agent_journal

#######################################################################################################################
'''Methods for generating initial data'''
#######################################################################################################################

''' Here we can keep the generate S and I.  We should not need to use R as we can assume this to be zero at the start
for all of our runs.  Else, we should add generate E as well. D will never be necessary to generate.'''

def generate_S(n=3, scale=1):
    arr = np.ones((n))
    ex = np.array([float(10) ** (j * scale) for j in range(1, n + 1)])
    return arr * ex

def generate_I(n=3, initial=1):
    arr = np.array([initial * float(j) for j in range(n)])
    return arr

def generate_R(n=3):
    arr = np.zeros((n))
    return arr.astype(float)

def initial_counts(n=3, scale=1, initial=1):
    S = generate_S(n, scale)
    I = generate_I(n, initial)
    R = generate_R(n)

    S = S - I

    return np.array([S, I, R])
#######################################################################################################################
#######################################################################################################################


#######################################################################################################################
'''SIR Function'''
#######################################################################################################################
def f(t, y, n, beta, gamma, C):
    S = y[:n]
    I = y[n:2 * n]
    R = y[2 * n:3 * n]

    sdot = compute_sdot(S, I, R, beta, C)
    idot = compute_idot(S, I, R, beta, C, gamma)
    rdot = compute_rdot(I, gamma)

    return np.array([sdot, idot, rdot]).flatten()

def f_multiscale(t, y, n, beta, gamma, t_resample, maxstep):

    # Build SIR solution vector
    S = y[:n]
    I = y[n:2 * n]
    R = y[2 * n:3 * n]

    # Call agent based model to get C for university
    if t % t_resample < maxstep*10**-3: # Within tolerance of 0 but smaller than the max step
        C = agent_journal.getcontacts

    sdot = compute_sdot(S, I, R, beta, C)
    idot = compute_idot(S, I, R, beta, C, gamma)
    rdot = compute_rdot(I, gamma)

    return np.array([sdot, idot, rdot]).flatten()

# Derivatives for SIR
def compute_sdot(S, I, R, beta, C):
    '''
    S,I ,R, are n by 1 vectors, beta is a constant,
    C is an n by n matrix
    '''

    inv_pop = np.divide(np.ones_like(S + I + R, dtype=float), S + I + R)

    sdot = [-beta * S[i] * np.sum(np.multiply(C[i],
                                              np.multiply(I, inv_pop)))
            for i in range(len(S))]
    return np.array(sdot)

def compute_idot(S, I, R, beta, C, gamma):
    sdot = -1 * compute_sdot(S, I, R, beta, C)
    return sdot - gamma * I

def compute_rdot(I, gamma):
    return gamma * I

#######################################################################################################################
#######################################################################################################################


#######################################################################################################################
# Plotting
#######################################################################################################################