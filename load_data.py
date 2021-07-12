import pandas as pd
import numpy as np

filename = 'C:/Users/aaris/Downloads/15minute_data_austin.csv'

psds = pd.read_csv(filename)
col_list = ['local_15min', 'air1', 'clotheswasher1', 'dishwasher1', 'furnace1', 'refrigerator1']
psds = psds[col_list]
# psds = psds.loc[35032:69679]
psds = psds.loc[35032:35047]
psds = psds.rename(columns={"local_15min": "Time"})
psds['Time'] = pd.to_datetime(psds['Time'])
psds.set_index('Time').resample('1T')
psds = psds.sort_values(by='Time', ascending=True)
psds.insert(loc=6, column="Sum of Power", value=psds.sum(axis=1))
psds.fillna(0, inplace=True)
print(psds)
# psds.to_csv('preprocessingoutputfinal.csv', index=False)

# 1/1/2018 (1:00 pm) - 12/31/2018 (11:45 pm)
# [163375:171273]

weatherfile = 'C:/Users/aaris/Downloads/ev_and_weather/ev_and_weather/weather.csv'

wds = pd.read_csv(weatherfile, parse_dates=['localhour'])
weather_columns = ['localhour', 'temperature']
wds = wds[weather_columns]
wds = wds.loc[163375:163379]
wpsds = pd.DataFrame(np.repeat(wds.values, 4, axis=0))
wpsds.columns = wds.columns
pd.to_datetime(wpsds['localhour'])
# print(wpsds)
wpsds.set_index('localhour').resample('15T')
# print(wpsds)
# wpsds.resample('15Min')
# wpsds = wpsds.sort_values(by='localhour', ascending=True)
# wpsds = wpsds.asfreq('15min')
# print(wpsds)
# wpsds.to_csv('loaddataweathertest.csv', index=False)
# psds['temperature'] = wpsds['temperature'].values
# psds.to_csv('loaddataweathertest1.csv', index=False)

