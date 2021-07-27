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

df = pd.read_csv('solar_load_weatherdata.csv')

NUM_SAMPLES = 97

def create_dataset(df, indexes):
    """
    Creates a tf Dataset

    Inputs
    df: a panda's dataframe
    indexes: a list of indexes to be included in the dataset

    Returns
    pandas dataframes with the correct inputs and outputs 
    """

    column_lst = []
    for column_name in range(NUM_SAMPLES):
        column_lst.append(str(column_name))

    #Create panadas dataframe for input and output
    power_input_df = pd.DataFrame(columns = column_lst) #power time series data
    temp_input_df = pd.DataFrame(columns = column_lst) #temperaturn time series data
    output_df = pd.DataFrame() 

    for index in indexes[0:100]:
        initial_index = index - (NUM_SAMPLES-1)

        #input rows
        power_input_row = df.loc[initial_index:index]['grid'].tolist()
        temp_input_row = df.loc[initial_index:index]['temperature'].tolist()

        #Checks to make sure that there is complete data
        if (len(power_input_row) == len(temp_input_row) == NUM_SAMPLES):

            #add input rows to input dataframes
            power_input_df.loc[len(power_input_df)] = power_input_row
            temp_input_df.loc[len(temp_input_df)] = temp_input_row

            #adds ouput data to output dataframe
            output_row = df.loc[[index]]
            output_df = output_df.append(output_row, ignore_index=True)

            power_input_df = np.array(power_input_df)
            temp_input_df = np.array(temp_input_df)
            output_df = np.array(output_df)

    #Cleans output dataframe (not necessary but improves runtime)
    # output_df.pop('Time')
    # output_df.pop('Sum of Power')
    # output_df.pop('temperature')

    #return the two input dataframes and the output dataframe
    return power_input_df, temp_input_df, output_df 

list_of_outputs = ['air1', 'clotheswasher1', 'dishwasher1', 'furnace1', 'refrigerator1', 'solar']

# Load saved model weights
model = create_LSTM_model(list_of_outputs)

model.load_weights('./model.ckpt')

# Predict on specified data
predict_indexes = [2, 3, 4, 5, 6, 7, 8, 9, 10]
y = create_dataset(df, indexes = predict_indexes)
prediction = model.predict(y, verbose=0)
print(prediction)