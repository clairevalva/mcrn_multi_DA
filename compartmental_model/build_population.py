'''
Methods for multiscale program
'''

import numpy as np

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


