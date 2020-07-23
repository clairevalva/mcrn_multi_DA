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
C = [[1, 0],[0,1]]
t_range = [0, total_days] # Choose unit time for ease, change later with dates
t_resample = 1 # Resample from the agent model every so often (1 -> 1 day resample)
maxstep = 0.01 # Maximum step size for the integrator

# Initial conditions
#S = [33500,168000-33499]
S = [33500, 168000]
I = [0,1]
R = [0,0]
y_0 = np.array([S,I,R]).flatten()

# Run Multiscale Model
solution = sint.solve_ivp(methods.f_multiscale, t_range, y_0, max_step = maxstep, args=(n, beta, gamma, t_resample, maxstep))

# Reshape solution
shaped = np.reshape(solution.y, (3,n,len(solution.t)))

#comp1 = shaped[:,0,:]

#######################################################################################################################
'''Fetch Fort Collins Data'''
#######################################################################################################################

# Fetch from URL
# url_cases = 'https://apps.larimer.org/api/covid/?t=1595446631841&gid=1219297132&csv=cases'
# url_deaths = 'https://larimer-county-data-lake.s3-us-west-2.amazonaws.com/Public/covid/covid_deaths.csv'
# r = requests.get(url, allow_redirects=True)

# Create a list of dates through today
#date_list = pd.date_range(start = '03-09-2020', end = datetime.today())
date_list = pd.date_range(start = start_date, end = end_date) # Chosen for a specific date

# Import the Larimer County confirmed/probable Cases
larimer_cases = pd.read_csv('LC-COVID-casesdata.csv', sep=',', parse_dates=['ReportedDate'])

# Pull out the Fort Collins cases
fc_cases = pd.DataFrame({'New Cases':[0]*len(date_list)},index = date_list)
for i in larimer_cases.index:
    date = larimer_cases.at[i,'ReportedDate']
    if larimer_cases.at[i, 'City'] == 'Fort Collins':
        fc_cases.at[date, 'New Cases'] = fc_cases.at[date, 'New Cases']+1

fc_cases['Total Cases'] = fc_cases['New Cases'].cumsum()

# Import the Larimer County deaths
larimer_deaths = pd.read_csv('LC-COVID-deathsdata.csv', sep=',', parse_dates=['death_date'])
#print(larimer_deaths)
#Pull out the Fort Collins confirmed/probable deaths
fc_deaths = pd.DataFrame({'New Deaths':[0]*len(date_list)},index = date_list)
for i in larimer_deaths.index:
    date = larimer_deaths.at[i,'death_date']
    if larimer_deaths.at[i, 'city'] == 'Fort Collins':
        fc_deaths.at[date, 'New Deaths'] = fc_deaths.at[date, 'New Deaths']+1

fc_deaths['Total Deaths'] = fc_deaths['New Deaths'].cumsum()

fc_data = pd.concat([fc_cases, fc_deaths], axis=1, sort=False)

# Plot Fort Collins cases
# fc_data.plot()
# plt.show()

#print(np.shape(shaped))
#######################################################################################################################

# Plot Model City vs. Real Data (Fort Collins, CO)
plt.clf()
sns.set_context("poster")
plt.scatter(solution.t,np.add(shaped[1,0,:], shaped[1,1,:]),label = "I, City")
plt.legend()
plt.show()

# Plot City and University
# plt.clf()
# sns.set_context("poster")
# plt.scatter(solution.t,np.add(shaped[1,0,:], shaped[1,1,:]),label = "I, City")
# plt.scatter(solution.t,shaped[1,0,:],label = "I, University")
# plt.legend()
# plt.show()

#sums = np.sum(comp1,axis = 0)

# # Plot
# compartment = 1
# plt.clf()
# plt.scatter(solution.t,shaped[0,compartment,:],label = "S")
# plt.scatter(solution.t,shaped[1,compartment,:],label = "I")
# plt.scatter(solution.t,shaped[2,compartment,:],label = "R")
# plt.legend()
# plt.show()
#
# # Plot
# compartment = 0
# plt.clf()
# plt.scatter(solution.t,shaped[0,compartment,:],label = "S")
# plt.scatter(solution.t,shaped[1,compartment,:],label = "I")
# plt.scatter(solution.t,shaped[2,compartment,:],label = "R")
# plt.legend()
# plt.show()

