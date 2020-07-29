import numpy.random as npr
from dapper import *
import get_case_data

M = 4  # ndim
p = 1  # ndim obs
Q_chol = np.zeros((M, M))
Q = Q_chol @ Q_chol.T
mu0 = np.array([168000, 1, 0, 2]) # initial conditions
P0_chol = np.eye(4)
P0 = P0_chol @ P0_chol.T

# Assimilation parameters
dt = 0.01  # integrational time step
dkObs = 100  # number of steps between observations
dtObs = dkObs * dt  # time between observations
KObs = 134  # total number of observations
K = dkObs * (KObs + 1)  # total number of time steps
ensemble_members = 50

R_chol = .5 * np.eye(p)
R = R_chol @ R_chol.T

# Init
# xx = np.zeros((K + 1, M)) # synthetic truth
yy = get_case_data.fetch() # get observation data
# xx[0] = mu0 + P0_chol @ npr.randn(M)
xxhat = np.zeros((K + 1, M))


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
    beta = 1  # Infection rate (chosen from paper)
    lam = 0.1  # Recovery rate (chosen from paper) (choose between 0.07 to 0.5)
    kappa = 0.002  # Death rate (chosen from paper)
    S = x[0]
    I = x[1]
    R = x[2]
    C = x[3]
    N = S + I + R
    d = np.zeros(4)
    d[0] = -beta * C * S * I / N
    d[1] = beta * C * S * I / N - (lam + kappa) * I
    d[2] = (lam + kappa) * I
    d[3] = 0
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


# Loop for generating synthetic truth
# for k in range(1, K + 1):
#     xx[k] = Dyn(xx[k - 1], (k - 1) * dt, dt)
#     xx[k] += Q_chol @ npr.randn(M)
    # if not k % dkObs:
    #     kObs = k // dkObs - 1
    #     yy[kObs] = Obs(xx[k], np.nan) + R_chol @ npr.randn(p)

# Useful linear algebra: compute B/A
def divide_1st_by_2nd(B, A):
    return nla.solve(A.T, B.T).T

def my_EnKF(N):
    E = mu0[:, None] + P0_chol @ npr.randn(M, N)
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

# Plot
fig, axs = plt.subplots(4, 1, True)
for m in range(1):
    # axs[m].plot(dt * np.arange(K + 1), xx[:, m], 'k', label="Truth")
    axs[m].plot(dt * np.arange(K + 1), xxhat[:, 3], 'b', label="Estimate")
    if m < p:
        axs[m].plot(dtObs * np.arange(1, KObs + 2), yy[:, m], 'g*')
    axs[m].set_ylabel("Dim %d" % m)
axs[0].legend()
plt.xlabel("Time (t)")
plt.savefig("test.png")

#np.save("c_values_beta=1.npy", xxhat)
