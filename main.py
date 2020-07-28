'''
Build our results from here
'''
import multiscale_model
import plot
import numpy as np


# Agent model information
class_periods = 3
class_size = 10

# http ://www.apple.com/covid19/mobility we could use this mobility data to get an idea for C[1,1]

# Compartmental model parameters
uni_city_coupling = float(1.0)
C = np.array([[1, uni_city_coupling],
              [uni_city_coupling, 1]]) # Unscaled contact matrix
beta = 1 #0.69 # Infection rate (chosen from paper)
gamma = 0.07 # Inverse of latent time to infection (chosen from paper)
lam = 0.1 # Recovery rate (chosen from paper) (choose between 0.07 to 0.5)
kappa = 0.002 # Death rate (found from (US Deaths/US Infections))
Q_percent = 1/5 # Quarantine percentage, set to 0 for no quarantining

''' 
    schedule_types can be "none" (no daily schedule),
    "day_stagger", or "week_stagger"
'''
stype = "week_stagger"  

# Initial conditions
# CSU, Fort Collins, Larimer County population sizes
uni_size = int(10000) # 33.5k
city_size = int(100000) # 168k
county_size = int(357000)  # not used yet
compartment_sizes = [uni_size, city_size]
n = len(compartment_sizes)
initial_infected = 1 # Always starts in the largest compartment!


# Run information
num_weeks = 2

# Run and plot
multiscale_model.run(initial_infected, num_weeks, class_periods, class_size,
 n, beta, gamma, lam, kappa,  C, Q_percent, compartment_sizes, schedule_types = stype)

plot.draw(class_periods, class_size)