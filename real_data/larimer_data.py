from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
#######################################################################################################################
# Fetch Larimer case/death data.
#######################################################################################################################
# Data
start_date = '03-09-2020'
end_date = '07-22-2020'
date_format = "%m-%d-%Y"
begin = datetime.strptime(start_date, date_format)
finish = datetime.strptime(end_date, date_format)
delta_days = finish - begin
total_days = delta_days.total_seconds()/86400 #convert from seconds to days.

# Fetch from URL
# url_cases = 'https://apps.larimer.org/api/covid/?t=1595446631841&gid=1219297132&csv=cases'
# url_deaths = 'https://larimer-county-data-lake.s3-us-west-2.amazonaws.com/Public/covid/covid_deaths.csv'
# r = requests.get(url, allow_redirects=True)

# Create a list of dates through today
#date_list = pd.date_range(start = '03-09-2020', end = datetime.today())
date_list = pd.date_range(start=start_date, end=end_date) # Chosen for a specific date

# Import the Larimer County confirmed/probable Cases
larimer_cases = pd.read_csv('LC-COVID-casesdata.csv', sep=',', parse_dates=['ReportedDate'])

# Pull out the Fort Collins cases
fc_cases = pd.DataFrame({'new_cases':[0]*len(date_list)},index = date_list)
for i in larimer_cases.index:
    date = larimer_cases.at[i, 'ReportedDate']
    if larimer_cases.at[i, 'City'] == 'Fort Collins':
        fc_cases.at[date, 'new_cases'] = fc_cases.at[date, 'new_cases']+1

fc_cases['total_cases'] = fc_cases['new_cases'].cumsum()

# Import the Larimer County deaths
larimer_deaths = pd.read_csv('LC-COVID-deathsdata.csv', sep=',', parse_dates=['death_date'])
#print(larimer_deaths)
#Pull out the Fort Collins confirmed/probable deaths
fc_deaths = pd.DataFrame({'new_deaths':[0]*len(date_list)},index = date_list)
for i in larimer_deaths.index:
    date = larimer_deaths.at[i,'death_date']
    if larimer_deaths.at[i, 'city'] == 'Fort Collins':
        fc_deaths.at[date, 'new_deaths'] = fc_deaths.at[date, 'new_deaths']+1

fc_deaths['total_deaths'] = fc_deaths['new_deaths'].cumsum()

fc_data = pd.concat([fc_cases, fc_deaths], axis=1, sort=False)
# print(fc_data)
# Plot Fort Collins cases
# fc_data.plot()
# plt.show()

#print(np.shape(shaped))
#######################################################################################################################


#######################################################################################################################
# Fetch Larimer mobility data
#######################################################################################################################
# Data (NEEDS DIFFERENT DATE FORMAT
start_date = '2020-03-09'
end_date = '2020-07-22'
date_format = "%Y-%m-%d"
begin = datetime.strptime(start_date, date_format)
finish = datetime.strptime(end_date, date_format)
delta_days = finish - begin
total_days = delta_days.total_seconds()/86400 #convert from seconds to days.

apple_mobility = pd.read_csv('applemobilitytrends-2020-07-25.csv', sep=',')
#print(apple_mobility.at[2591, 'region']) #verify larimer county
date_list = pd.date_range(start=start_date, end=end_date)
larimer_mobility = pd.DataFrame({'mobility_percentage': [0]*len(date_list)}, index=date_list)
for datenum in range(len(date_list)):
    #print(date_list.date[datenum])
    larimer_mobility.at[date_list.date[datenum], 'mobility_percentage'] = \
        apple_mobility.at[2591, date_list.date[datenum].strftime('%Y-%m-%d')]
print(larimer_mobility)
larimer_mobility.plot()
plt.show()
#######################################################################################################################
