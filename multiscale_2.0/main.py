# import multiscale_model
import numpy as np
# import sys
import scipy.integrate as scint
import initialize_population
import build_network
import solver
import matplotlib.pyplot as plt
import networkx as nx
import solver

# Model info
total_time = 10

# Schedule

# Network information
population = 100
seed = 1
network = build_network.make(population, seed)
L = nx.laplacian_matrix(network)# sparse Laplacian matrix

# Initial conditions
I_0 = np.zeros(population) #should be zeros
I_0 = np.reshape(I_0, (population, 1))
I_0[0] = 1

# Disease spread information
beta = .3
threshold = 0.25

# Integrator settings
step_size = 0.01

#######################################################################################################################
# MAIN #
#######################################################################################################################

#Run the solver
I = solver.run_model(I_0, total_time, step_size, beta, L)

# Plot the infection curve for each node
plt.clf()
for i in range(population):
    plt.plot(I.t, I.y[i])
plt.show()

# Plot the total infected
# total_infected = np.zeros_like(I.y[1])
# for i in range(population):
#     total_infected = total_infected + I.y[i]
# plt.plot(I.t, total_infected, label="Total infected")
# plt.show()