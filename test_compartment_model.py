# Imports
import numpy as np
import scipy as scp
import scipy.integrate as sint
import matplotlib.pyplot as plt
import methods
import seaborn as sns
import csv
import pandas as pd
from datetime import datetime, date
import requests
import scale_cm

'''
TODO: Allow for time varying C in the SIR model
'''

# Data
start_date = '03-09-2020'
end_date = '07-22-2020'
date_format = "%m-%d-%Y"
begin = datetime.strptime(start_date, date_format)
finish = datetime.strptime(end_date, date_format)
delta_days = finish - begin
total_days = delta_days.total_seconds()/86400 #convert from seconds to days.

# Parameters
n = 2
gamma = 0.125
beta = .15

sizes = [10 + 10*xx for xx in range(10)]
C11_names = ["contact_matrices/cm_periods=5_size=" + str(size) + ".npy" for size in sizes]
cross_diag = 0.5

t_range = [0, total_days] # Choose unit time for ease, change later with dates
# t_interventions = [x for x in range(t_range)] # Resample from the agent model every so often, have to change

maxstep = 0.01 # Maximum step size for the integrator

C11s = scale_cm.compute_c11(C11_names)
Cs = np.array([[[c, cross_diag],[cross_diag,1]] for c in C11s])

# Initial conditions
#S = [33500,168000-33499]
S = [33500, 168000]
I = [0,1]
R = [0,0]
y_0 = np.array([S,I,R]).flatten()

for step in range(len(Cs)):
    # Run Multiscale Model
    solution = sint.solve_ivp(methods.f, t_range, y_0, max_step = maxstep, args=(n, beta, gamma, Cs[step]))
    np.save("SIRs/class_size=" + str(sizes[step]) + ".npy", solution)


