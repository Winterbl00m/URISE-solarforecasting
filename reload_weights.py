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
from main import create_LSTM_model
from main import create_dataset

df = pd.read_csv('solar_load_weatherdata.csv')

# Load saved model weights
model = create_LSTM_model()

model.load_weights('./model.ckpt')

# Predict on specified data
# predict_indexes = [0:10]
# y = create_dataset(df, indexes = predict_indexes)
# prediction = model.predict(y)
# print(prediction)