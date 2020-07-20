# Imports
import numpy as np
import scipy as scp
import scipy.integrate as sint
import matplotlib.pyplot as plt
import methods
import seaborn as sns

'''
TODO: Allow for time varying C in the SIR model
'''

# Parameters
n = 2
gamma = 0.5
beta = 4
C = [[.5, 0],[0,.5]]
#C = [[10,.5],[.5,1]]
t_range = [0, 50]
t_intervention = 6

# Initial conditions
#S = [33500,168000-33499]
S = [33500, 168000]
I = [0,1]
R = [0,0]
y_0 = np.array([S,I,R]).flatten()

# Main
# Run without intervention
pleasework = sint.solve_ivp(methods.f, t_range, y_0, max_step = 0.01, args=(n, beta, gamma, C))

# Run with intervention
#pleasework = sint.solve_ivp(methods.f_intervention, t_range, y_0, max_step = 0.01, args=(n, beta, gamma, C,t_intervention))


# Reshape solution
shaped = np.reshape(pleasework.y, (3,n,len(pleasework.t)))

#comp1 = shaped[:,0,:]

# Plot
plt.clf()
sns.set_context("poster")
plt.scatter(pleasework.t,np.add(shaped[1,0,:], shaped[1,1,:]),label = "I, City")
plt.scatter(pleasework.t,shaped[1,0,:],label = "I, University")
plt.legend()
plt.show()

#sums = np.sum(comp1,axis = 0)

# Plot
compartment = 1
plt.clf()
plt.scatter(pleasework.t,shaped[0,compartment,:],label = "S")
plt.scatter(pleasework.t,shaped[1,compartment,:],label = "I")
plt.scatter(pleasework.t,shaped[2,compartment,:],label = "R")
plt.legend()
plt.show()

# Plot
compartment = 0
plt.clf()
plt.scatter(pleasework.t,shaped[0,compartment,:],label = "S")
plt.scatter(pleasework.t,shaped[1,compartment,:],label = "I")
plt.scatter(pleasework.t,shaped[2,compartment,:],label = "R")
plt.legend()
plt.show()

