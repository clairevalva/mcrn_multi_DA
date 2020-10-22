import numpy as np
import tree_behavior
import matplotlib.pyplot as plt
import initialize_population
import solver
#import solver


# Model info
num_infected = 3
total_pop = 500
infect_seed = False # can prescribe a seed for random, otherwise False, the randomization here may be unecessary
num_weeks = 2
infectious_time = 4 #number of days a person stays infected
p_infected = .2

# Schedule
tree, initial_infected = tree_behavior.initialize_tree(num_infected, total_pop, seed = infect_seed, return_initial = True)

# Network information
population = 500
L = initialize_population.build_L(population)


# some psuedocode because I do not understand how L is structured


    
