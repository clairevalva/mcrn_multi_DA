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

#######################################################################################################################
'''Fetch Fort Collins Data'''
#######################################################################################################################

# Make initial cases in Fort Collins = 0

# build = {'Date': [0], 'Cases': [0]}
# fc_cases = pd.DataFrame(data=build)

#print(fc_cases)

larimer_cases = pd.read_csv('LC-COVID-casesdata.csv', sep=',', parse_dates=['ReportedDate'])

date_list = pd.date_range(start = '03-09-2020', end = datetime.today())

new_fc_cases = pd.DataFrame({'Cases':[0]*len(date_list)},index = date_list)

#print(fc_cases)

for i in larimer_cases.index:
    date = larimer_cases.at[i,'ReportedDate']
    if larimer_cases.at[i, 'City'] == 'Fort Collins':
        new_fc_cases.at[date, 'Cases'] = new_fc_cases.at[date, 'Cases']+1

print(new_fc_cases)


# with open('LC-COVID-casesdata.csv', newline='') as csvfile:
#     reader = csv.reader(csvfile, delimiter=',', quotechar='|')
#     for row in reader:
#         df = pd.DataFrame({'Date': [row[1]], 'Cases': [0]})
#         print(df)
#         fc_cases.append(df)
#         #print(row[1])
#         #print(', '.join(row))
#         if "Fort Collins" in row:
#             x = 1
#             #fc_cases = fc_cases + 1
# print(fc_cases)

#######################################################################################################################

# Plot Model City vs. Real Data (Fort Collins, CO)
# plt.clf()
# sns.set_context("poster")
# plt.scatter(pleasework.t,np.add(shaped[1,0,:], shaped[1,1,:]),label = "I, City")
# plt.legend()
# plt.show()

# Plot City and University
# plt.clf()
# sns.set_context("poster")
# plt.scatter(pleasework.t,np.add(shaped[1,0,:], shaped[1,1,:]),label = "I, City")
# plt.scatter(pleasework.t,shaped[1,0,:],label = "I, University")
# plt.legend()
# plt.show()

#sums = np.sum(comp1,axis = 0)

# # Plot
# compartment = 1
# plt.clf()
# plt.scatter(pleasework.t,shaped[0,compartment,:],label = "S")
# plt.scatter(pleasework.t,shaped[1,compartment,:],label = "I")
# plt.scatter(pleasework.t,shaped[2,compartment,:],label = "R")
# plt.legend()
# plt.show()
#
# # Plot
# compartment = 0
# plt.clf()
# plt.scatter(pleasework.t,shaped[0,compartment,:],label = "S")
# plt.scatter(pleasework.t,shaped[1,compartment,:],label = "I")
# plt.scatter(pleasework.t,shaped[2,compartment,:],label = "R")
# plt.legend()
# plt.show()

