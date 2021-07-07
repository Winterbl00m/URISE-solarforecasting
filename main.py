import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 
import pandas as pd
import random 

filename = 'C:/Users/aaris/Downloads/15minute_data_austin.csv'

def load_data(filename):
    """
    Loads data from csv file and returns a pandas dataframe
    (operates in Python 3.8.5 ('base': conda))

    filename: pecan street csv file
    """
    psds = pd.read_csv(filename)
    col_list = ['local_15min', 'air1', 'clotheswasher1', 'dishwasher1', 'furnace1', 'refrigerator1']
    psds = psds[col_list]
    psds = psds.loc[35032:69679]
    psds = psds.rename(columns={"local_15min": "Time"})
    psds.insert(loc=6, column="Sum of Power", value=psds.sum(axis=1))
    psds.fillna(0, inplace=True)
    psds.to_csv('preprocessingoutputfinal.csv', index=False)

    return psds


def create_datasets(df, train_frac, val_frac):
    """
    TODO
    Creates a tf Datasets

    df: a panda's dataframe
    train_frac: fraction of data for training(float between 0 and 1)
    val_frac: fraction of data for validation(float between 0 and 1)
    """
    
    list_of_times = df['Time'].to_list()

    #Number of timestamps for each dataset
    train_num = int(len(list_of_times) * train_frac)
    val_num = int(len(list_of_times) * val_frac + train_num)
 
    #Shuffles the list of times and splits it into val, test, and train
    random.shuffle(list_of_times)
    train_times = list_of_times[:train_num]
    val_times = list_of_times[train_num:val_num]
    test_times = list_of_times[val_num:]

    #Create panadas dataframe for input and output
    output_pd = pd.DataFrame()
    input_pd = pd.DataFrame()

    for timestamp in train_times:
        index_of_time = df[df['Time'] == timestamp].index[0]
        initial_index = index_of_time - (24*4)

        foo1 = df.loc[initial_index:index_of_time]['Sum of Power']
        foo2 = foo1.transpose()

        foo = df.loc[[index_of_time]]

        output_pd = output_pd.append(foo, ignore_index=True)
        input_pd = input_pd.append(foo2, ignore_index=True)

    
    output_pd.pop('Time')
    output_pd.pop('Sum of Power')
    print("output is")
    print(output_pd.head())
    print("input is")
    print(input_pd.head())
    # target = df.pop('target')
    # dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))



    pass

def create_feature_layer():
    feature_columns = []

    time = tf.feature_column.numeric_column(df['Time'])
    feature_columns.append(time)

    my_feature_layer = layers.DenseFeatures(feature_columns)
    return my_feature_layer

def create_model(learning_rate, my_feature_layer, number_of_outputs):
    """
    Creates an ANN model

    learning_rate: float. The learning rate of the model. Between 0 and 1
    feature_layer: A tf layer.  Aka the input layer. The first layer of the model. 
    """
    #Create Model
    model = keras.Sequential()

    # Add the layer containing the features to the model.
    model.add(my_feature_layer)

    #Hidden Layers(RNN)
    model.add(layers.SimpleRNN(12))

    model.add(layers.SimpleRNN(12))

    #Output layer
    model.add(layers.Dense(number_of_outputs, activation='sigmoid', name='output'))

    #Compile Model
    model.compile(
        optimizer = tf.keras.optimizers.Adam(lr=learning_rate),
        loss = tf.keras.losses.BinaryCrossentropy(),
        metrics = tf.keras.metrics.BinaryAccuracy())

    return model
     


def train_model(model, train_dataset, val_dataset):
    """
    Train the model by feeding it data.
    From https://towardsdatascience.com/building-your-first-neural-network-in-tensorflow-2-tensorflow-for-hackers-part-i-e1e2f1dfe7a0

    model: tf model
    train_dataset = tf training dataset
    val_dataset = tf validation dataset
    """

    history = model.fit(
        train_dataset.repeat(), 
        epochs=10, 
        steps_per_epoch=500,
        validation_data=val_dataset.repeat(), 
        validation_steps=2
    )

df = pd.read_csv('preprocessingoutputfinal.csv')
# print(df.head)
create_datasets(df, train_frac = .6, val_frac = .2)
