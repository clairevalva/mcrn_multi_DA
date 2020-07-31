from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#######################################################################################################################
# Fetch Larimer case/death data.
#######################################################################################################################
# Data

def fetch():
    start_date = '03-09-2020'
    end_date = '07-22-2020'
    date_format = "%m-%d-%Y"
    begin = datetime.strptime(start_date, date_format)
    finish = datetime.strptime(end_date, date_format)
    delta_days = finish - begin
    total_days = delta_days.total_seconds()/86400 #convert from seconds to days.

    # Create a list of dates through today
    date_list = pd.date_range(start=start_date, end=end_date) # Chosen for a specific date
    date_nums = [x for x in range(len(date_list))]

    # Import the Larimer County confirmed/probable Cases
    larimer_cases = pd.read_csv('raw_data/LC-COVID-casesdata.csv', sep=',', parse_dates=['ReportedDate'])

    # Pull out the Fort Collins cases
    fc_cases = pd.DataFrame({'new_cases':[0]*len(date_list)}, index=date_list)
    for i in larimer_cases.index:
        date = larimer_cases.at[i, 'ReportedDate']
        if larimer_cases.at[i, 'City'] == 'Fort Collins':
            fc_cases.at[date, 'new_cases'] = fc_cases.at[date, 'new_cases']+1

    fc_cases['total_cases'] = fc_cases['new_cases'].cumsum()
    fc_cases_datenums = fc_cases.reset_index()
    fc_cases_datenums = fc_cases_datenums.drop(axis=1, columns='index')
    fc_cases_datenums = fc_cases_datenums.drop(axis=1, columns='new_cases')
    fc_cases_array = fc_cases_datenums.to_numpy()

    # Import the Larimer County deaths
    larimer_deaths = pd.read_csv('raw_data/LC-COVID-deathsdata.csv', sep=',', parse_dates=['death_date'])
    #print(larimer_deaths)
    #Pull out the Fort Collins confirmed/probable deaths
    fc_deaths = pd.DataFrame({'new_deaths':[0]*len(date_list)},index = date_list)
    for i in larimer_deaths.index:
        date = larimer_deaths.at[i,'death_date']
        if larimer_deaths.at[i, 'city'] == 'Fort Collins':
            fc_deaths.at[date, 'new_deaths'] = fc_deaths.at[date, 'new_deaths']+1

    fc_deaths['total_deaths'] = fc_deaths['new_deaths'].cumsum()
    fc_deaths_datenums = fc_deaths.reset_index()
    fc_deaths_datenums = fc_deaths_datenums.drop(axis=1, columns='index')
    fc_deaths_datenums = fc_deaths_datenums.drop(axis=1, columns='new_deaths')
    fc_deaths_array = fc_deaths_datenums.to_numpy()

    fc_data_array = np.concatenate((fc_cases_array, fc_deaths_array))
    fc_data_array = fc_data_array.reshape(2, 136)
    return fc_data_array[:,:-1]
