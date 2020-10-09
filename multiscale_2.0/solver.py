'''
Solves the diffusion problem on the time varying network.

For a fixed interval draw a new network (e.g., each day).  Then, allow diffusion to occur at a small
 machine timestep.

'''
import numpy as np
import scipy.integrate as scint

def dIdt(t, I, max_step, L, beta):
    threshold = .25
    Idot = np.zeros_like(I)

    for j in range(len(I)):
        if I[j] >= 1-10**(-6):
            Idot[j] = 0
        elif I[j] >= threshold:
            Idot[j] = (1-I[j])/max_step
        else:
            mat = L @ I
            Idot[j] = -beta * mat[j]
    return Idot


def run_model(I_0, total_time, step_size, beta, L):
    interval = [0, total_time]

    I = scint.solve_ivp(dIdt, interval, I_0[:,0], method='RK45', max_step=step_size, min_step=step_size,
                            args=(step_size, L, beta))
    return I
