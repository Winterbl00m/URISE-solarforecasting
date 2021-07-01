# preprocessing code for pecan street disaggregation
# operates in Python 3.8.5 ('base': conda)

import pandas as pd

# import os
# path = 'C:/Users/aaris/Downloads/RNN-Preprocessing'
# dirs = os.listdir(path)
# for file in dirs:
#     print(file)

psds = pd.read_csv('C:/Users/aaris/Downloads/15minute_data_austin.csv')
a = psds.loc[35032:35035, 'local_15min']
b = psds.loc[35032:35035, 'air1']
c = psds.loc[35032:35035, 'clotheswasher1']
d = psds.loc[35032:35035, 'dishwasher1']
e = psds.loc[35032:35035, 'furnace1']
f = psds.loc[35032:35035, 'refrigerator1']
data = pd.DataFrame([[a], [b], [c], [d], [e], [f]])
df = pd.DataFrame({"Time":[a],
                 "air1":[b],
                 "clotheswasher1":[c],
                 "dishwasher1":[d],
                 "furnace1":[e],
                 "refigerator1":[f]})

df.to_csv('output1.csv', index=False)
df.insert(loc=6, column="Sum of Power", value=df.sum(axis=1))
df.to_csv('output.csv', index=False)

# 69679
# print(psds.loc[psds['air1'] < 0])
# psds.loc[['local_15min', 'air1', 'clotheswasher1', 'dishwasher1', 'furnace1', 'refrigerator1']]
# psds.loc['1642':'2335':'2361':'2818':'3456', 'local_15min']
# psds.loc['1642':'2335':'2361':'2818':'3456', 'air1']
# psds.loc['1642':'2335':'2361':'2818':'3456', 'clotheswasher1']
# psds.loc['1642':'2335':'2361':'2818':'3456', 'dishwasher1']
# psds.loc['1642':'2335':'2361':'2818':'3456', 'furnace1']
# psds.loc['1642':'2335':'2361':'2818':'3456', 'refrigerator1']