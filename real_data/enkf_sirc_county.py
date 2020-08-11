import numpy.random as npr
from dapper import *
import get_case_data_county

# Dimensions
M = 4  # model
p = 1  # observations

# Fetch observation data
obs_remove = 50 # remove first few data points of observations
yy = get_case_data_county.fetch()[obs_remove:]

# Model covariance
Q_chol = np.zeros((M, M))
Q_chol[3, 3] = 0.015 # add uncertainty in C parameter
Q = Q_chol @ Q_chol.T

# Initial conditions
mu0 = np.array([357000 - yy[1, 0], .05 * yy[1, 0], .95 * yy[1, 0], .20]) # initial conditions SEQIRD and C

# Initial covariance
P0_chol = np.diag([10, 10, 10, 0.01])
P0 = P0_chol @ P0_chol.T

# Assimilation parameters
dt = 0.01  # integrational time step
dkObs = 100  # number of steps between observations
dtObs = dkObs * dt  # time between observations
KObs = np.size(yy)-1  # total number of observations
K = dkObs * (KObs + 1)  # total number of time steps
ensemble_members = 20

# Observation error covariance
R_chol = 40 * np.eye(p)
R = R_chol @ R_chol.T

# Init
xxhat = np.zeros((K + 1, M))
xxhat[0, :] = mu0

def estimate_mean_and_cov(E):
    M, N = E.shape

    x_bar = np.sum(E, axis=1) / N
    B_bar = np.zeros((M, M))
    for n in range(N):
        xc = (E[:, n] - x_bar)[:, None]  # x_centered
        B_bar += xc @ xc.T
        # B_bar += np.outer(xc,xc)
    B_bar /= (N - 1)

    return x_bar, B_bar

def estimate_cross_cov(Ex, Ey):
    N = Ex.shape[1]
    #assert N == Ey.shape[1]
    X = Ex - np.mean(Ex, axis=1, keepdims=True)
    Y = Ey - np.mean(Ey, axis=1, keepdims=True)
    CC = X @ Y.T / (N - 1)
    return CC

def dxdt(x):
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

    if E.ndim == 1:
        # Truth (single state vector) case
        E = step(E)
    else:
        # Ensemble case
        for n in range(E.shape[1]):
            E[:, n] = step(E[:, n])

    return E

def Obs(E, t):
    # Return I + R
    arr = np.reshape(E[1] + E[2], (ensemble_members, 1))
    return np.transpose(arr)

def divide_1st_by_2nd(B, A):
    return nla.solve(A.T, B.T).T

def my_EnKF(N):
    E = mu0[:, None] + P0_chol @ npr.randn(M, N)
    for k in range(1, K + 1):
        # Forecast
        t = k * dt
        E = Dyn(E, t - dt, dt)
        E += Q_chol @ npr.randn(M, N)
        # Check for negative C values
        for j in range(1, ensemble_members):
            if E[3, j] < 0:
                E[3, j] = 0

        if not k % dkObs:
            # Analysis
            y = yy[k // dkObs - 1]  # current obs
            Eo = Obs(E, t) # observed ensemble
            # Compute ensemble moments
            BH = estimate_cross_cov(E, Eo)
            HBH = estimate_mean_and_cov(Eo)[1]
            # Compute Kalman Gain
            KG = divide_1st_by_2nd(BH, HBH + R)
            # Generate perturbations
            Perturb = R_chol @ npr.randn(p, N)
            # Update ensemble with Kalman gain
            E += KG @ (y[:, None] - Perturb - Eo)
            # Check for very small values of C
            for j in range(1, ensemble_members):
                if E[3, j] < 0:
                    E[3, j] = 0

        xxhat[k] = np.mean(E, axis=1)

        # Prevent decrease in I + R
        if xxhat[k, 1] + xxhat[k, 2] < xxhat[k-1, 1] + xxhat[k-1, 2]:
            xxhat[k, 1] = xxhat[k-1, 1]
            xxhat[k, 2] = xxhat[k-1, 2]



# Run assimilation
my_EnKF(ensemble_members)

# Plot
plt.clf()
plt.figure(figsize=(16, 9), dpi=1200)
fig, axs = plt.subplots(2, 1, True)
axs[0].plot(dt * np.arange(K + 1), xxhat[:, 1] + xxhat[:, 2], 'b', label="Estimate of I+R")
axs[0].plot(dtObs * np.arange(1, KObs + 2), yy[:], 'g*', label="Observations")
axs[1].plot(dt * np.arange(K + 1), xxhat[:, 3], 'b', label="Estimate of C")
axs[0].legend()
axs[1].legend()
plt.xlabel("Time (t)")
plt.savefig("assimilated_data/SIRC_assimilation_county.png")

np.save("assimilated_data/SIRC_beta=1_county.npy", xxhat)
