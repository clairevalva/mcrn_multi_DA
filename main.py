'''
Build our results from here
'''
import multiscale_model
import plot
import numpy as np
import sys


# Agent model information
class_periods = int(sys.argv[1])
class_size = int(sys.argv[2])

# http ://www.apple.com/covid19/mobility we could use this mobility data to get an idea for C[1,1]

# Compartmental model parameters
beta = .3 #0.69 # Infection rate (chosen from paper)
C22 = np.mean(np.load("real_data/c_values_beta=1.npy")[:,3])
uni_city_coupling = float(sys.argv[3])
C = np.array([[1, (1 / beta) * uni_city_coupling * C22],
              [(1 / beta) * uni_city_coupling * C22, (1 / beta) * C22]]) # Unscaled contact matrix
gamma = 0.07 # Inverse of latent time to infection (chosen from paper)
lam = 0.1 # Recovery rate (chosen from paper) (choose between 0.07 to 0.5)
kappa = 0.002 # Death rate (found from (US Deaths/US Infections))
Q_percent = float(sys.argv[4]) # Quarantine percentage, set to 0 for no quarantining
major_size = 500 # false if no major separation, size of major otherwise

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
compartment_sizes = [uni_size, city_size]
n = len(compartment_sizes)
initial_infected = 1 # Always starts in the largest compartment!


# Run information
num_weeks = int(sys.argv[6])

# Run and plot
Cs, solutions = multiscale_model.run(initial_infected, num_weeks, class_periods, class_size,
                     n, beta, gamma, lam, kappa,  C, Q_percent, compartment_sizes, schedule_type = stype, majors = major_size)

savenamesol = "multi_runs/multi_Q=" + str(Q_percent) + "_csize=" + str(class_size) + "_per=" + str(class_periods) + "_msize" + str(major_size) + "_sch=" + str(stype) + "_coupling=" + str(coupling) + ".npy"

savenameC = "multi_runs/C_multi_Q=" + str(Q_percent) + "_csize=" + str(class_size) + "_per=" + str(class_periods) + "_msize" + str(major_size) + "_sch=" + str(stype) + "_coupling=" + str(coupling) + ".npy"

np.save(savenameC, Cs)
np.save(savenamesol, solutions)

