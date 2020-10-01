import multiscale_model
import numpy as np
import sys

# Agent model information
class_periods = int(sys.argv[1])
class_size = int(sys.argv[2])

# Compartmental model parameters
beta = .3 # Infection rate (chosen somewhat arbitrarily)
gamma = 0.07 # Inverse of latent time to infection (chosen from paper)
lam = 0.1 # Recovery rate (chosen from paper) (choose between 0.07 to 0.5)
kappa = 0.002 # Death rate (found from (US Deaths/US Infections))
Q_percent = float(sys.argv[4]) # Quarantine percentage, set to 0 for no quarantining

# Assimilated contact values
C22 = np.mean(np.load("real_data/assimilated_data/SIRC_beta=1_city.npy")[:, 3])
C33 = np.mean(np.load("real_data/assimilated_data/SIRC_beta=1_county.npy")[:, 3])

# Build C matrix
uni_city_coupling = float(sys.argv[3])
C = np.array([[1, (1 / beta) * uni_city_coupling * C22],
              [(1 / beta) * uni_city_coupling * C22, (1 / beta) * C22]]) # Unscaled contact matrix

major_size = 500 # false if no major separation, size of major otherwise

agent_ensN = 10 # size of agent model ensemble

''' 
    schedule_types can be "none" (no daily schedule),
    "day_stagger", or "week_stagger"
'''
stype = str(sys.argv[5]) 

# Initial conditions
# CSU, Fort Collins, Larimer County population sizes
uni_size = int(33.5*(10**3)) # 33.5k
city_size = int(168*(10**3)) # 168k
county_size = int(357000)  # not used yet
compartment_sizes = [uni_size, city_size - uni_size]
n = len(compartment_sizes)
initial_infected = 1 # Always starts in the largest compartment!


# Run information
num_weeks = int(sys.argv[6])

# Run and plot
Cs, solutions, variances = multiscale_model.run(initial_infected, num_weeks, class_periods, class_size,
                                                n, beta, gamma, lam, kappa, C, Q_percent, compartment_sizes, schedule_type = stype, majors = major_size, agent_ens = agent_ensN)

savenamesol = "multi_runs/multi_Q=" + str(Q_percent) + "_csize=" + str(class_size) + "_per=" + str(class_periods) + "_msize" + str(major_size) + "_sch=" + str(stype) + "_coupling=" + str(uni_city_coupling) + ".npy"
#
savenameC = "multi_runs/C_multi_Q=" + str(Q_percent) + "_csize=" + str(class_size) + "_per=" + str(class_periods) + "_msize" + str(major_size) + "_sch=" + str(stype) + "_coupling=" + str(uni_city_coupling) + ".npy"

# savenamesol = "multi_runs/no_school.npy"
# savenameC = "multi_runs/no_school.npy"
np.save(savenameC, Cs)
np.save(savenamesol, solutions)