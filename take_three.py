# Importing the libraries
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import random 
import numpy as np
import matplotlib.pyplot as plt
import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Input

NUM_SAMPLES = 97

def split_data(df, train_frac, val_frac):
    """
    Splits the data into three list of times 
    
    Input
    df: a panda's dataframe
    train_frac: fraction of data for training(float between 0 and 1)
    val_frac: fraction of data for validation(float between 0 and 1)

    Returns
    train_indexes: the indexes for the training dataset
    val_indexes: the indexs for the validation dataset
    test_indexes: the indexes for the testing dataset
    """
    index_lst = []
    for index in df.index:
        index_lst.append(index)

    #Number of timestamps for each dataset
    train_num = int(len(index_lst) * train_frac)
    val_num = int(len(index_lst) * val_frac + train_num)
 
    #Shuffles the list of times and splits it into val, test, and train
    random.shuffle(index_lst)
    train_indexes = index_lst[:train_num]
    val_indexes = index_lst[train_num:val_num]
    test_indexes = index_lst[val_num:]

    return train_indexes, val_indexes, test_indexes


def create_dataset(df, indexes):
    """
    Creates a tf Dataset

    Inputs
    df: a panda's dataframe
    indexes: a list of indexes to be included in the dataset

    Returns
    dataset: a tf dataset with the correct inputs and outputs 
    """
    column_lst = []
    for column_name in range(NUM_SAMPLES):
        column_lst.append(str(column_name))


    #Create panadas dataframe for input and output
    output_df = pd.DataFrame()
    input_df = pd.DataFrame(columns = column_lst)

    for index in indexes[0:100]:
        initial_index = index - (NUM_SAMPLES-1)

        input_row = df.loc[initial_index:index]['Sum of Power'].tolist()

        if len(input_row) == NUM_SAMPLES :
            input_df.loc[len(input_df)] = input_row

            output_row = df.loc[[index]]
            output_df = output_df.append(output_row, ignore_index=True)

        

    
    output_df.pop('Time')
    output_df.pop('Sum of Power')

    # dataset = tf.data.Dataset.from_tensor_slices((input_df.values, output_df.values))

    return input_df, output_df 


def create_model():
    """
    Adapted from https://github.com/katanaml/sample-apps/blob/master/04/multi-output-model.ipynb
    Returns
    model : a tf model with one or more output layers
    
    """
    # Input Layer
    input_layer = Input(shape=(NUM_SAMPLES,))
    # input_size = some number
    # somethingcool = tf.reshape(input_layer, [int(input_size/NUM_SAMPLES), 1, NUM_SAMPLES])
    reshape_input = tf.reshape(input_layer, [10, 1, NUM_SAMPLES])
    # Hidden Layer(s)
    RNN_layer = LSTM(units=128)(reshape_input)
    first_hidden_layer = Dense(units='128', activation='relu')(RNN_layer)

    #Output Layer(s)
    y1_output = Dense(units='1', name='air1')(first_hidden_layer)
    y2_output = Dense(units='1', name='clotheswasher1')(first_hidden_layer)
    y3_output = Dense(units='1', name='dishwasher1')(first_hidden_layer)
    y4_output = Dense(units='1', name='furnace1')(first_hidden_layer)
    y5_output = Dense(units='1', name='refrigerator1')(first_hidden_layer)
    #create tf model object
    model = Model(inputs=input_layer,outputs=[y1_output, y2_output, y3_output, y4_output, y5_output])
    print(model.summary())

    # Specify the optimizer, and compile the model with loss functions for both outputs
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss={'air1': 'mse',
                  'clotheswasher1': 'mse',
                  'dishwasher1' : 'mse',
                  'furnace1' : 'mse',
                  'refrigerator1' : 'mse'},

                  metrics={'air1': tf.keras.metrics.RootMeanSquaredError(),
                  'clotheswasher1': tf.keras.metrics.RootMeanSquaredError(),
                  'dishwasher1' : tf.keras.metrics.RootMeanSquaredError(),
                  'furnace1' : tf.keras.metrics.RootMeanSquaredError(),
                  'refrigerator1' : tf.keras.metrics.RootMeanSquaredError()} )

    return model


#reads data from the preprocessed csv file
df = pd.read_csv('preprocessingoutputfinal.csv')

train_frac = .6
val_frac = .2

#Splits the data into train, val, and test
train_indexes, val_indexes, test_indexes = split_data(df, train_frac, val_frac)

#Creates the Datasets
train_x, train_y = create_dataset(df, indexes = train_indexes)
val_x, val_y = create_dataset(df, indexes = val_indexes)

# list_of_outputs = ['air1', 'clotheswasher1', 'dishwasher1', 'furnace1', 'refrigerator1']

#Turns output data from a dataframe to five arrays
train_y = train_y.pop('air1') , train_y.pop('clotheswasher1'), train_y.pop('dishwasher1'), train_y.pop('furnace1'), train_y.pop('refrigerator1')
val_y = val_y.pop('air1') , val_y.pop('clotheswasher1'), val_y.pop('dishwasher1'), val_y.pop('furnace1'), val_y.pop('refrigerator1')

#Create Model
model = create_model()

# Train the model for 100 epochs
history = model.fit(train_x, train_y,
                    epochs=100, batch_size=10, validation_data=(val_x,val_y))