import pandas as pd
import numpy as np

filename = 'C:/Users/aaris/Downloads/15minute_data_austin.csv'

psds = pd.read_csv(filename)
col_list = ['local_15min', 'air1', 'clotheswasher1', 'dishwasher1', 'furnace1', 'refrigerator1', 'solar', 'grid']
psds = psds[col_list]
psds = psds.loc[35032:69679]
psds = psds.rename(columns={"local_15min": "Time"})
psds['solar'] *= -1
psds['Time'] = psds['Time'].astype(str).str[:-6]
# removes timezone from Time
psds['Time'] = pd.to_datetime(psds['Time'], errors='coerce')
# to_datetime turns values into datetimeindex, required to merge with weather data
psds = psds.sort_values(by='Time', ascending=True)
# sorts values in ascending order of Time - data was not completely sorted originally
psds.fillna(0, inplace=True)
psds.insert(loc=8, column="Sum of Power", value=psds.drop(['solar', 'grid'], axis=1).sum(axis=1))

# 1/1/2018 (1:00 pm) - 12/31/2018 (11:45 pm)
# [163375:171273]

weatherfile = 'C:/Users/aaris/Downloads/ev_and_weather/ev_and_weather/weather.csv'

wds = pd.read_csv(weatherfile)
weather_columns = ['localhour', 'temperature']
wds = wds[weather_columns]
wds = wds.loc[163375:171273]
wds.set_index('localhour')
wds = wds.rename(columns={"localhour": "Time"})
wds['Time'] = pd.to_datetime(wds['Time'], errors='coerce')

combined = psds.merge(wds, how='outer')
combined = combined.fillna(method='ffill')
combined.to_csv('solar_load_weatherdata.csv', index=False)