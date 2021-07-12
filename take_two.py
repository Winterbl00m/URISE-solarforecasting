import tensorflow as tf
from tensorflow import keras
import pandas as pd
import random 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

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
    for column_name in range(4*24+1):
        column_lst.append(str(column_name))


    #Create panadas dataframe for input and output
    output_df = pd.DataFrame()
    input_df = pd.DataFrame(columns = column_lst)

    

    for index in indexes[0:100]:
        initial_index = index - (24*4)

        input_row = df.loc[initial_index:index]['Sum of Power'].tolist()

        if len(input_row) == 97 :
            input_df.loc[len(input_df)] = input_row

            output_row = df.loc[[index]]
            output_df = output_df.append(output_row, ignore_index=True)
        

    
    output_df.pop('Time')
    output_df.pop('Sum of Power')

    output_df = output_df.pop('air1')
    
    dataset = tf.data.Dataset.from_tensor_slices((input_df.values, output_df.values))

    return dataset 


def create_model():
    """
    Adapted from https://github.com/katanaml/sample-apps/blob/master/04/multi-output-model.ipynb
    """
    # Define model layers.
    input_layer = Input(shape=(97,))

    first_dense = Dense(units='128', activation='relu')(input_layer)
    # Y1 output will be fed from the first dense
    y1_output = Dense(units='1', name='refrigerator')(first_dense)

    model = Model(inputs=input_layer,outputs=[y1_output])

    return model



df = pd.read_csv('preprocessingoutputfinal.csv')
train_frac = .6
val_frac = .2
train_indexes, val_indexes, test_indexes = split_data(df, train_frac, val_frac)

train_dataset = create_dataset(df, indexes = train_indexes)
val_dataset = create_dataset(df, indexes = val_indexes)
# test_dataset = create_dataset(df, timestamps = test_times)

# for element in train_dataset:
#     print(element)

model = create_model()

# Specify the optimizer, and compile the model with loss functions for both outputs
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss={'refrigerator': 'mse'},
              metrics={'refrigerator': tf.keras.metrics.RootMeanSquaredError()})


# Train the model for 200 epochs
history = model.fit(train_dataset,
                    epochs=100, validation_data=val_dataset)