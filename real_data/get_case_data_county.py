import pandas as pd

#######################################################################################################################
# Fetch Larimer County case/death data.
#######################################################################################################################

def fetch():
    start_date = '03-09-2020'
    end_date = '08-11-2020'

    # Create a list of dates through today
    date_list = pd.date_range(start=start_date, end=end_date) # Chosen for a specific date

    # Import the Larimer County confirmed/probable Cases
    larimer_cases = pd.read_csv('raw_data/LC-COVID-casesdata.csv', sep=',', parse_dates=['ReportedDate'])

    # Pull out the Larimer cases
    cases = pd.DataFrame({'new_cases': [0] * len(date_list)}, index=date_list)
    for i in larimer_cases.index:
        date = larimer_cases.at[i, 'ReportedDate']
        cases.at[date, 'new_cases'] = cases.at[date, 'new_cases']+1

    cases['total_cases'] = cases['new_cases'].cumsum()
    cases_datenums = cases.reset_index()
    cases_datenums = cases_datenums.drop(axis=1, columns='index')
    cases_datenums = cases_datenums.drop(axis=1, columns='new_cases')
    cases_array = cases_datenums.to_numpy()
    return cases_array[:-1]
