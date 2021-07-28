# Importing the libraries
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import random 
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, concatenate
import matplotlib.pyplot as plt
from LSTM_model import create_LSTM_model
from LSTM_model import create_dataset

df = pd.read_csv('solar_load_weatherdata.csv')

list_of_outputs = ['air1', 'clotheswasher1', 'dishwasher1', 'furnace1', 'refrigerator1', 'solar']

# Load saved model weights
model = create_LSTM_model(list_of_outputs)

model.load_weights('./model.ckpt')

# Predict on data
def pred_index(df, initialindex, finalindex):
    index_lst = []
    for index in df.index:
        index_lst.append(index)
    predict_indexes = index_lst[initialindex:finalindex]
    
    return predict_indexes

in1 = 29000
in2 = 29100

x, y, z = create_dataset(df, indexes = pred_index(df, in1, in2), list_of_outputs = list_of_outputs)
prediction = model.predict([x, y])

# Locate time on which data is predicted
predictiontime = df['Time'].loc[in1:(in2-1)]

# Locate actual energy data
actualair = df['air1'].loc[in1:(in2-1)]
actualclotheswasher = df['clotheswasher1'].loc[in1:(in2-1)]
actualdishwasher = df['dishwasher1'].loc[in1:(in2-1)]
actualfurnace = df['furnace1'].loc[in1:(in2-1)]
actualrefrigerator = df['refrigerator1'].loc[in1:(in2-1)]
actualsolar = df['solar'].loc[in1:(in2-1)]


# Plot predicted vs. actual
plt.figure()

plt.plot(predictiontime, actualair, color = 'green', label='actual')
plt.plot(predictiontime, prediction[0], color = 'blue', label='predicted')

plt.title('air1 Prediction')
plt.ylabel('EGauge Energy')
plt.xlabel('Time')
plt.legend()

plt.show()


plt.figure()

plt.plot(predictiontime, actualclotheswasher, color = 'green', label='actual')
plt.plot(predictiontime, prediction[1], color = 'blue', label='predicted')

plt.title('clotheswasher1 Prediction')
plt.ylabel('EGauge Energy')
plt.xlabel('Time')
plt.legend()

plt.show()


plt.figure()

plt.plot(predictiontime, actualdishwasher, color = 'green', label='actual')
plt.plot(predictiontime, prediction[2], color = 'blue', label='predicted')

plt.title('dishwasher1 Prediction')
plt.ylabel('EGauge Energy')
plt.xlabel('Time')
plt.legend()

plt.show()


plt.figure()

plt.plot(predictiontime, actualfurnace, color = 'green', label='actual')
plt.plot(predictiontime, prediction[3], color = 'blue', label='predicted')

plt.title('furnace1 Prediction')
plt.ylabel('EGauge Energy')
plt.xlabel('Time')
plt.legend()

plt.show()


plt.figure()

plt.plot(predictiontime, actualrefrigerator, color = 'green', label='actual')
plt.plot(predictiontime, prediction[4], color = 'blue', label='predicted')

plt.title('refrigerator1 Prediction')
plt.ylabel('EGauge Energy')
plt.xlabel('Time')
plt.legend()

plt.show()


plt.figure()

plt.plot(predictiontime, actualsolar, color = 'green', label='actual')
plt.plot(predictiontime, prediction[5], color = 'blue', label='predicted')

plt.title('solar Prediction')
plt.ylabel('EGauge Energy')
plt.xlabel('Time')
plt.legend()

plt.show()


