import numpy as np
import scipy as scp
import scipy.integrate as sint
import matplotlib.pyplot as plt
import methods
import seaborn as sns
import scale_cm
import compartmental_seird

# Parameters
n = 2 # Number of compartments
beta = 0.69 # Infection rate (chosen from paper)
gamma = 0.07 # Inverse of latent time to infection (chosen from paper)
lam = 0.1 # Recovery rate (chosen from paper) (choose between 0.07 to 0.5)
kappa = 0.002 # Death rate (found from (US Deaths/US Infections))
time_interval = [0, 100]
C = [[1, .5], [.5, 1]]

# Initial conditions
S = [33500, 168000-33499]
E = [0, 0]
I = [0, 1]
R = [0, 0]
D = [0, 0]
y_0 = np.array([S, E, I, R, D]).flatten() # Create initial vector for solver

# Solver parameters
maxstep = 0.1


# Run Multiscale Model
solution = sint.solve_ivp(compartmental_seird.seird_f, time_interval, y_0, max_step=maxstep,
                          args=(n, beta, gamma, lam, kappa,  C))

# Plot for a test
# Reshape solution
shaped = np.array([np.reshape(solution.y, (5, n, np.size(solution.t)))])
# Plot I for compartment 0 + 1
for i in range(0, 5):
    plt.scatter(solution.t, np.add(shaped[0, i, 0, :], shaped[0, i, 1, :]), label="City " + str(i))

plt.legend()
plt.show()