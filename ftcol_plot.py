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

'''
# Plot Model City vs. Real Data (Fort Collins, CO)
plt.clf()
sns.set_context("poster")
plt.scatter(solution.t,np.add(shaped[1,0,:], shaped[1,1,:]),label = "I, City")
plt.legend()
plt.show()
'''

