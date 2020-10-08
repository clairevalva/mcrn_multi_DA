'''
Solves the diffusion problem on the time varying network.

For a fixed interval draw a new network (e.g., each day).  Then, allow diffusion to occur at a small
 machine timestep.
'''

def dIdt(I, L, threshold):
    threshold = .25
    Idot = [L*I + 1 for i in range(len(I)) if I[i] > threshold]
    return Idot

