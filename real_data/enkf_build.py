import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as nla
import numpy.random as npr
import scipy.integrate

M = 3 # ndim
p = 3 # ndim obs
Q_chol = np.zeros((M, M))
Q = Q_chol @ Q_chol.T
mu0     = np.array([1.509, -1.531, 25.46])
P0_chol = np.eye(3)
P0      = P0_chol @ P0_chol.T

# Assimilation parameters
dt    = 0.01           # integrational time step
dkObs = 25             # number of steps between observations
dtObs = dkObs*dt       # time between observations
KObs  = 60             # total number of observations
K     = dkObs*(KObs+1) # total number of time steps


R_chol = np.sqrt(2)*np.eye(p)
R      = R_chol @ R_chol.T

# Init
xx    = np.zeros((K+1   ,M))
yy    = np.zeros((KObs+1,p))
xx[0] = mu0 + P0_chol @ npr.randn(M)
xxhat = np.zeros((K+1,M))

def Dyn(E, t0, dt):
    def step(x0):
        return scipy.integrate.RK45(lambda t, x: dxdt(x), x0, t0, dt)

    if E.ndim == 1:
        # Truth (single state vector) case
        E = step(E)
    else:
        # Ensemble case
        for n in range(E.shape[1]):
            E[:, n] = step(E[:, n])

    return E

def Obs(E, t):
    if E.ndim == 1: return E[:p]
    else:           return E[:p,:]

# Loop
for k in range(1,K+1):
    xx[k]  = Dyn(xx[k-1], (k-1)*dt, dt)
    xx[k] += Q_chol @ npr.randn(M)
    if not k%dkObs:
        kObs = k//dkObs-1
        yy[kObs] = Obs(xx[k],np.nan) + R_chol @ npr.randn(p)

# METHODS
def dxdt(x):
    sig  = 10.0
    rho  = 28.0
    beta = 8.0/3
    x,y,z = x
    d     = np.np.zeros(3)
    d[0]  = sig*(y - x)
    d[1]  = rho*x - y - x*z
    d[2]  = x*y - beta*z
    return d


# Useful linear algebra: compute B/A
def divide_1st_by_2nd(B,A):
    return nla.solve(A.T,B.T).T

def my_EnKF(N):
    E = mu0[:,None] + P0_chol @ npr.randn((M,N))
    for k in range(1,K+1):
        # Forecast
        t   = k*dt
        E   = Dyn(E,t-dt,dt)
        E  += Q_chol @ npr.randn((M,N))
        if not k%dkObs:
            # Analysis
            y        = yy[k//dkObs-1] # current obs
            Eo       = Obs(E,t)
            BH       = np.estimate_cross_cov(E,Eo)
            HBH      = np.estimate_mean_and_cov(Eo)[1]
            Perturb  = R_chol @ npr.randn((p,N))
            KG       = divide_1st_by_2nd(BH, HBH+R)
            E       += KG @ (y[:,None] - Perturb - Eo)
        xxhat[k] = np.mean(E,axis=1)

# Run assimilation
my_EnKF(10)

# Plot
fig, axs = plt.subplots(3,1,True)
for m in range(3):
    axs[m].plot(dt*np.arange(K+1), xx   [:,m], 'k', label="Truth")
    axs[m].plot(dt*np.arange(K+1), xxhat[:,m], 'b', label="Estimate")
    if m<p:
        axs[m].plot(dtObs*np.arange(1,KObs+2),yy[:,m],'g*')
    axs[m].set_ylabel("Dim %d"%m)
axs[0].legend()
plt.xlabel("Time (t)")



