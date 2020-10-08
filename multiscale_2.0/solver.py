'''
Solves the diffusion problem on the time varying network.

For a fixed interval draw a new network (e.g., each day).  Then, allow diffusion to occur at a small
 machine timestep.

'''

import numpy as np


def dIdt(t, I, L, threshold = .25):
    
    I = np.reshape(I, (1,10))
    Idot = np.zeros_like(I)
    
    # print(I.shape)
    for j in range(len(I)):
        if I[j,0] >= 1:
            Idot[j,0] = 0
        else:
             mat = np.matmul(L,I)
             Idot[j,0] = mat[j]
    return Idot


