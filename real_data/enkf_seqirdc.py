import numpy.random as npr
from dapper import *
import get_case_data

# Model
x0 = np.array([168000-200, 60, 30, 40, 1, 1, 1, 0.002, .5]) # Initial condition for SEQIRD C kappa Q_percent
M = len(x0) # ndim
Q_chol = np.zeros((M, M)) # model noise
#Q_chol = .1 * np.diag([.5, .5, .5, .5, .5, .5, .01, 0.00005, .005])
Q = Q_chol @ Q_chol.T

# Observations
yy = np.transpose(get_case_data.fetch()) # get observation data
yy = yy[18:, :] #remove first few observations
p = len(yy[1, :])  # ndim obs
#R_chol = .01 * np.eye(p) # observation noise
R_chol = 100 * np.diag([1, 0.01])
R = R_chol @ R_chol.T

# Assimilation parameters
ensemble_members = 30
dt = 0.005  # integrational time step
dkObs = 200  # number of steps between observations
dtObs = dkObs * dt  # time between observations
KObs = len(yy[:, 1])-1  # total number of observations
K = dkObs * (KObs + 1)  # total number of time steps
#P0_chol = 0.1 * np.eye(M)
P0_chol = .1 * np.diag([10, 1, 1, 1, 1, .5, .01, 0.00005, 0.01])
P0 = P0_chol @ P0_chol.T



# Init analysis
xxhat = np.zeros((K + 1, M))
xxhat[0, :] = x0



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
    beta = 1 #infection rate
    gamma = 0.1 #0.07  # Inverse of latent time to infection (chosen from paper)
    lam = 0.1  # Recovery rate (chosen from paper) (choose between 0.07 to 0.5)
    #kappa = 0.002  #0.002# Death rate (found from (US Deaths/US Infections))
    S = x[0]
    E = x[1]
    Q = x[2]
    I = x[3]
    R = x[4]
    D = x[5]
    C = x[6]
    kappa = x[7]
    Q_percent = x[8]
    N = 168000
    d = np.zeros(M)
    d[0] = -beta * C * S * I / N
    d[1] = beta * C * S * I / N - gamma * E
    d[2] = gamma * Q_percent * E - (lam + kappa) * Q
    d[3] = gamma * (1-Q_percent) * E - (lam + kappa) * I
    d[4] = lam * (I + Q)
    d[5] = kappa * (I + Q)
    d[6] = 0
    d[7] = 0
    d[8] = 0
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
    # add in Q = E[2] if you want to observe those who quarantine
    arr = np.reshape([E[2] + E[3] + E[4], E[5]], (ensemble_members, 2))
    return np.transpose(arr)

def divide_1st_by_2nd(B, A):
    return nla.solve(A.T, B.T).T

def my_EnKF(N):
    E = x0[:, None] + P0_chol @ npr.randn(M, N)
    for k in range(1, K + 1):
        # Forecast
        t = k * dt
        E = Dyn(E, t - dt, dt)
        E += Q_chol @ npr.randn(M, N)
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
        xxhat[k] = np.mean(E, axis=1)

# Run assimilation
my_EnKF(ensemble_members)

print(xxhat[-1,:])

# Plot
plt.clf()
plt.figure(figsize=(16, 9), dpi=1200)
fig, axs = plt.subplots(5, 1, True)
axs[0].plot(dt * np.arange(K + 1), xxhat[:, 2] + xxhat[:, 3] + xxhat[:, 4], 'b', label="Estimate of I+R")
axs[0].plot(dtObs * np.arange(1, KObs + 2), yy[:, 0], 'g*', label="Infection Observations")
axs[1].plot(dt * np.arange(K + 1), xxhat[:, 5], 'b', label="Estimate of D")
axs[1].plot(dtObs * np.arange(1, KObs + 2), yy[:, 1], 'g*', label="Death Observations")
axs[2].plot(dt * np.arange(K + 1), xxhat[:, -3], 'b', label="Estimate of C")
axs[3].plot(dt * np.arange(K + 1), xxhat[:, -2], 'b', label="Estimate of kappa")
axs[4].plot(dt * np.arange(K + 1), xxhat[:, -1], 'b', label="Estimate of Q_percent")
axs[0].legend()
axs[1].legend()
axs[2].legend()
axs[3].legend()
axs[4].legend()

axs[0].set_ylim(0,700)
axs[1].set_ylim(0,25)
plt.xlabel("Time (t)")
plt.savefig("SEQIRDC_assimilation.png")

np.save("SEQIRDC_beta=1.npy", xxhat)