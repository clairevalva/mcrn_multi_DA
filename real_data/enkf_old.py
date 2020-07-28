# Ensemble Kalman Filter for Data Assimilation to recover C

import filterpy.kalman as kf
#import compartmental_model/compartmental_model
import numpy as np
import get_case_data

uni_city_coupling = float(1.0)
C = 1
beta = 0.69 # Infection rate (chosen from paper)
lam = 0.1 # Recovery rate (chosen from paper) (choose between 0.07 to 0.5)
kappa = 0.002 # Death rate (found from (US Deaths/US Infections))


# x will be Sdot vector, observation will be change in S
# Initial Data
N = 168000 #city size
initial_infected = 0
x = np.array([N, 0.0, 0.0, C]) #Susceptible, Infected, Recovered, C



def fx(x, dt):
    S = x[0]
    I = x[1]
    C = x[3]
    F = np.array([[-beta * C * S * I / N, -beta * C * S / N, 0, -beta * S * I / N],
                  [beta * C * I / N, beta * C * S / N - (lam + kappa), 0, beta * S * I / N],
                  [0, lam+kappa, 0, 0],
                  [0, 0, 0, 0]])

    return np.dot(F, x)

def hx(x):
   return np.array([x[1]+x[2]])

# Filter info
dt = 1 # 1 day
dim_z = 1 # 1 dimensional observations
N = 10 # ensemble members
P = np.eye(4) * 100.
f = kf.EnsembleKalmanFilter(x=x, P=P, dim_z=1, dt=dt, N=N,
         hx=hx, fx=fx)

std_noise = 3.
f.R *= std_noise**2
#f.Q = Q_discrete_white_noise(2, dt, .01)
f.Q *= std_noise**2

observations = get_case_data.fetch()
while True:
    z = 1#observations[n, 0]
    f.predict()
    f.update(np.asarray([z]))