import numpy as np
import scipy as scp
import scipy.integrate as sint
import matplotlib.pyplot as plt
import methods
import seaborn as sns
import scale_cm
import compartmental_seird
import quar_agents
import agent_journal
import random

# Parameters
n = 2 # Number of compartments
beta = 0.69 # Infection rate (chosen from paper)
gamma = 0.07 # Inverse of latent time to infection (chosen from paper)
lam = 0.1 # Recovery rate (chosen from paper) (choose between 0.07 to 0.5)
kappa = 0.002 # Death rate (found from (US Deaths/US Infections))
perq = 1/3 # people are horrible (quarEEntine rate)
time_interval = [0, 100]
C = np.array([[1, .5], [.5, 1]])
class_periods = 5
class_size = 50


time_stops = [xx*10 for xx in range(6)]
time_stops.append(time_interval[-1])

# Initial conditions
S = [33500, 168000-33499]
S[0] = 3000
E = [0, 0]
Q = [0, 0]
I = [0, 1]
R = [0, 0]
D = [0, 0]
y_0 = np.array([S, E, Q, I, R, D]).flatten() # Create initial vector for solver

# Solver parameters
maxstep = 0.1
solutions = []
c11s = []

model = agent_journal.UnivModel(S[0],5,S[0], class_periods = class_periods, class_size = class_size)

for yy in range(class_periods):
    model.step()

contacts = agent_journal.contactnumbers(model, returnarr = False)
scaling = contacts[-1]/20  
C[0,0] = 20

c11s.append(20)

for xx in range(len(time_stops) - 1):
    interval = [time_stops[xx], time_stops[xx + 1]]
   
    # Run Multiscale Model
    solution = sint.solve_ivp(compartmental_seird.seird_f, interval, y_0, max_step=maxstep,
                          args=(n, beta, gamma, lam, kappa,  C, perq))
    solutions.append(solution)
    
    print("solution number: " + str(xx))
    # Plot for a test
    shaped = np.array([np.reshape(solution.y, (6, n, np.size(solution.t)))])
    y0 = shaped[0,:,:,-1].flatten()
    
    uniQ = shaped[0,2,0,-1]
    uniD = shaped[0,-1,0,-1]
    
    removeN = int(np.floor(np.sum(uniQ + uniD)))
    removels = quar_agents.remove_ls(removeN, model.tick, S[0])
    
    print("model number: " + str(xx + 1))
    model = agent_journal.UnivModel(S[0],5,S[0], class_periods = class_periods, class_size = class_size)
    model.step(toremove = removels)
    for yy in range(class_periods - 1):
        model.step()
    
    contacts = agent_journal.contactnumbers(model, returnarr = False)
    c11 = contacts[-1]/scaling 
    C[0,0] = c11
    
    c11s.append(c11)

    
    
np.save("testc11s.npy", c11s)
np.save("testsolutions.npy", solutions)