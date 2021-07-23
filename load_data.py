import pandas as pd
import numpy as np
from datetime import datetime, timedelta
# import os

# os.path
filename = 'C:/Users/aaris/Downloads/15minute_data_austin.csv'
dataid = 1642

# read file
psds = pd.read_csv(filename)

# locates all rows for household
psds = psds[psds['dataid'] == dataid]

# locates columns for time, specific appliances, and total power (grid)
col_list = ['local_15min', 'air1', 'clotheswasher1', 'dishwasher1', 'furnace1', 'refrigerator1', 'solar', 'grid']
psds = psds[col_list]

psds = psds.rename(columns={"local_15min": "Time"})
psds['solar'] *= -1

# removes timezone from Time
psds['Time'] = psds['Time'].astype(str).str[:-6]

# to_datetime turns values into datetimeindex, required to merge with weather data
psds['Time'] = pd.to_datetime(psds['Time'], errors='coerce')

# sorts values in ascending order of Time - data was not completely sorted originally
psds = psds.sort_values(by='Time', ascending=True)

# fill empty cells with 0
psds.fillna(0, inplace=True)

# create sum of power for simpler aggregated data
psds.insert(loc=8, column="Sum of Power", value=psds.drop('grid', axis=1).sum(axis=1))


inittime = psds['Time'].min()
fintime = psds['Time'].max()

# 1/1/2018 (1:00 pm) - 12/31/2018 (11:45 pm)
# [163375:171273]

def hour_rounder(t):
    # Rounds to nearest hour by adding a timedelta hour if minute >= 30
    return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)
               +timedelta(hours=t.minute//30))

initialtime = hour_rounder(inittime)
finaltime = hour_rounder(fintime)

x = pd.to_datetime(initialtime)
print(initialtime)
y = pd.to_datetime(finaltime)
print(finaltime)

metadatafile = 'C:/Users/aaris/Downloads/metadata.csv'

mta = pd.read_csv(metadatafile)
mtacolumns = ['dataid', 'city', 'state']
mta = mta[mtacolumns]
mta = mta.loc[mta['dataid'] == str(dataid)]
state = mta['state']

def find_latitude(state):
    if (state == 'Texas').any():
        latitude = 30.292432

    if (state == 'Colorado').any():
        latitude = 40.027278

    if (state == 'California').any():
        latitude = 32.778033
    
    return latitude

weatherfile = 'C:/Users/aaris/Downloads/ev_and_weather/ev_and_weather/weather.csv'

wds = pd.read_csv(weatherfile)
weather_columns = ['localhour', 'latitude', 'temperature', 'summary']
wds = wds[weather_columns]
wds = wds.rename(columns={"localhour": "Time"})
wds['Time'] = pd.to_datetime(wds['Time'], errors='coerce')
wds = wds.loc[wds['latitude'] == find_latitude(state)]
wds = wds.drop('latitude', axis=1)
wds = wds.sort_values(by='Time', ascending=True)

print(wds.dtypes)

wds = wds[(wds['Time'] >= initialtime) & (wds['Time'] < finaltime)]

combined = psds.merge(wds, how='outer')
combined = combined.fillna(method='ffill')
combined = combined.sort_values(by='Time', ascending=True)
combined.to_csv('solar_load_weatherdata.csv', index=False)