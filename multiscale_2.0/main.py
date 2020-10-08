# import multiscale_model
import numpy as np
# import sys
import scipy.integrate as scint
import initialize_population
import solver
import matplotlib.pyplot as plt


max_step = 0.01
threshold = 0.25
I_0 = np.zeros((initialize_population.population))
I_0[0] = 1

interval = [0,10]


L = initialize_population.L


# interval is just time start/end
solved = scint.solve_ivp(solver.dIdt, interval, I_0, max_step=max_step, args = (L, threshold))
           
print(solved.t.shape)
print(solved.y.shape)

plt.clf()
for i in range(10):
    plt.plot(solved.t, solved.y[i], label = str(i))
plt.legend()
plt.show()  