import pandas as pd

df = pd.DataFrame([[1, 2], [4, 5], [7, 8]],
     index=['cobra', 'cobra', 'sidewinder'],
     columns=['max_speed', 'shield'])

Total = df['max_speed'].loc()
print(Total)
print(df)

df.loc['cobra', 'max_speed']


import os

path = 'C:/Users/aaris/Downloads/RNN-Preprocessing'
dirs = os.listdir(path)
for file in dirs:
    print(file)


