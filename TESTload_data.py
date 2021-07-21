import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim

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


initialtime = psds['Time'].min()
print(initialtime)
finaltime = psds['Time'].max()
print(finaltime)
# 1/1/2018 (1:00 pm) - 12/31/2018 (11:45 pm)
# [163375:171273]

metadatafile = 'C:/Users/aaris/Downloads/metadata.csv'

mta = pd.read_csv(metadatafile)
mtacolumns = ['dataid', 'city', 'state']
mta = mta[mtacolumns]
mta = mta.loc[mta['dataid'] == str(dataid)]
city = mta['city']

geolocator = Nominatim()

print(geolocator.geocode(city))


weatherfile = 'C:/Users/aaris/Downloads/ev_and_weather/ev_and_weather/weather.csv'

wds = pd.read_csv(weatherfile)
weather_columns = ['localhour', 'temperature']
wds = wds[weather_columns]
wds = wds.rename(columns={"localhour": "Time"})
wds['Time'] = pd.to_datetime(wds['Time'], errors='coerce')
wds = wds.set_index('Time')

wds = wds[initialtime:finaltime]

combined = psds.merge(wds, how='outer')
combined = combined.fillna(method='ffill')
combined = combined.sort_values(by='Time', ascending=True)
combined.to_csv('2TESTsolar_load_weatherdata.csv', index=False)