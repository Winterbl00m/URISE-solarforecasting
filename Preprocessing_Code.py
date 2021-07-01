# preprocessing code for pecan street disaggregation
# operates in Python 3.8.5 ('base': conda)

import pandas as pd

psds = pd.read_csv('C:/Users/aaris/Downloads/15minute_data_austin.csv')
col_list = ['local_15min', 'air1', 'clotheswasher1', 'dishwasher1', 'furnace1', 'refrigerator1']
psds = psds[col_list]
psds = psds.loc[35032:69679]
psds = psds.rename(columns={"local_15min": "Time"})
psds.insert(loc=6, column="Sum of Power", value=psds.sum(axis=1))
psds.fillna(0, inplace=True)
psds.to_csv('preprocessingoutputfinal.csv', index=False)

# 35032-69679