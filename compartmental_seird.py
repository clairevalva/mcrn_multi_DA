import numpy as np

def seird_f(t, y, n, beta, gamma, lam, kappa,  C, perq):
    '''
    beta = infection rate
    gamma = inverse latent time before infection onset
    lam = recovery rate (short for lambda)
    kappa = death rate
    C = contact matrix
    '''

    # Partition solution vector into the SEIRD portions
    S = y[:n]
    E = y[n:2 * n]
    Q = y[2 * n:3 * n]
    I = y[3 * n:4 * n]
    R = y[4 * n:5 * n]
    D = y[5 * n:6 * n] # D Is not relevant for the dynamics really

    # 1/(total size) of each compartment (not counting D)
    inv_pop = np.divide(np.ones_like(S + E + I + R, dtype=float), S + E + I + R)

    Sdot = compute_Sdot(S, I, inv_pop, beta, C)
    Edot = compute_Edot(S, E, I, inv_pop, beta, gamma, C)
    Qdot = compute_Qdot(E, Q, gamma, lam, kappa, perq)
    Idot = compute_Idot(E, I, gamma, lam, kappa, perq)
    Rdot = compute_Rdot(I, Q, lam)
    Ddot = compute_Ddot(I, Q, kappa)

    return np.array([Sdot, Edot, Qdot, Idot, Rdot, Ddot]).flatten()

# Derivatives for SEIRD
def compute_Sdot(S, I, inv_pop, beta, C):
    '''
    S,I,inv_pop, are n by 1 vectors, beta is a constant,
    C is an n by n matrix
    '''

    Sdot = [-beta * S[i] * np.sum(np.multiply(C[i], np.multiply(I, inv_pop))) for i in range(len(S))]

    return np.array(Sdot)

def compute_Qdot(E, Q, gamma, lam, kappa, perq):
    t1 = gamma*perq
    
    return E*t1 - lam*kappa*Q
    

def compute_Edot(S, E, I, inv_pop, beta, gamma, C):

    Sdot = -1 * compute_Sdot(S, I, inv_pop, beta, C)
    return Sdot - gamma * E

def compute_Idot(E, I, gamma, lam, kappa, perq):

    return (1 - perq)*gamma * E - (lam + kappa) * I

def compute_Rdot(I, Q, lam):

    return lam * (I + Q)

def compute_Ddot(I, Q, kappa):

    return kappa * (I + Q)