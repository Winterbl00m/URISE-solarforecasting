import pandas as pd

# Given a time at which data is supposed to be predicted within test_times,
# find 24-hours worth of aggregated data prior to the time

timestamp = train_times[-1]
index_of_time = df[df['Time'] == timestamp].index[0]
initial_index = index_of_time - (24*4)
histpower1 = psds.loc[initial_index:index_of_time]
histpower = histpower1.loc['Sum of Power']
Total_hist_power = pd.DataFrame(histpower, index=['Sum of Power'])