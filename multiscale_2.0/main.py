import numpy as np
import initialize_population
import matplotlib.pyplot as plt
import solver

# Model info
num_weeks = 2
infectious_time = 4 #number of days a person stays infected

# Schedule

# Network information
population = 500
L = initialize_population.build_L(population)

# Initial conditions
I_0 = np.zeros(population) #should be zeros
I_0[0] = 1

# Disease spread information
beta = .0002 * population
threshold = 0.25

# Integrator settings
step_size = 0.1

#######################################################################################################################
# MAIN #
#######################################################################################################################

#scheduler.get_L(L, 6)

#Run the solver
t, I = solver.run_model(population, I_0, num_weeks, step_size, beta, L, infectious_time)


#######################################################################################################################
# Plotting #
#######################################################################################################################

fig, axs = plt.subplots(2, sharex=True)
# # Plot the infection curve for each node
for i in range(population):
    axs[0].plot(t, I[i])
# axs[0].xlabel("Time (days)")
# axs[0].ylabel("Infection Percentage")

# plt.clf()
# for i in range(population):
#     plt.plot(t, I[i, :])
# plt.xlabel("Time (days)")
# plt.ylabel("Infection Percentage")
# plt.show()

# Plot the total infected
total_infected = np.zeros_like(I[1])
for i in range(population):
    total_infected = total_infected + np.round(I[i])
axs[1].plot(t, total_infected)
plt.xlabel("Time (days)")
plt.show()

#######################################################################################################################
# EXTRAS #
#######################################################################################################################

# Draw a graph
# nx.draw(network_trivial, with_labels=True, font_weight='bold')
# plt.show()