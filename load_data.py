import pandas as pd
import numpy as np

filename = 'C:/Users/aaris/Downloads/15minute_data_austin.csv'

psds = pd.read_csv(filename)
col_list = ['local_15min', 'air1', 'clotheswasher1', 'dishwasher1', 'furnace1', 'refrigerator1']
psds = psds[col_list]
psds = psds.loc[35032:69679]
# psds = psds.loc[35032:35047]
# dt.tz_localize works on [35032:35047] but not on entire data set
psds = psds.rename(columns={"local_15min": "Time"})
psds['Time'] = pd.to_datetime(psds['Time'])
# to_datetime turns values into datetimeindex, required to remove timezone
psds['Time'] = psds['Time'].dt.tz_localize(None)
# tz_localize(None) removes timezone from Time
psds.insert(loc=6, column="Sum of Power", value=psds.sum(axis=1))
psds.fillna(0, inplace=True)
# psds.to_csv('preprocessingoutputfinal.csv', index=False)

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
combined.to_csv('loadweatherdata.csv', index=False)