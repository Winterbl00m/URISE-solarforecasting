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

    #Cleans output dataframe (not necessary but improves runtime)
    # output_df.pop('Time')
    # output_df.pop('Sum of Power')
    # output_df.pop('temperature')

    #return the two input dataframes and the output dataframe
    return power_input_df, temp_input_df , output_df 


def create_LSTM_model(list_of_outputs):
    """
    Adapted from https://github.com/katanaml/sample-apps/blob/master/04/multi-output-model.ipynb
    and https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/
    Returns
    model : a tf model with one or more output layers
    
    """
    # Model A
    # Input Layer
    input_layerA = Input(shape=(NUM_SAMPLES,))
    size_inputA = tf.size(input_layerA)
    reshape_inputA = tf.reshape(input_layerA, [size_inputA/NUM_SAMPLES, 1, NUM_SAMPLES])
    #Hidden Layers
    x = LSTM(units=128)(reshape_inputA)
    x = Dense(units='32', activation='relu')(x)
    x = Model(inputs=input_layerA, outputs=x)

    # Model B
    # Input Layer
    input_layerB = Input(shape=(NUM_SAMPLES,))
    size_inputB = tf.size(input_layerB)
    reshape_inputB = tf.reshape(input_layerB, [size_inputB/NUM_SAMPLES, 1, NUM_SAMPLES])
    #Hidden Layers
    y = LSTM(units=128)(reshape_inputB)
    y = Dense(units='32', activation='relu')(y)
    y = Model(inputs=input_layerB, outputs=y)

    #Concatination Layer
    combined = concatenate([x.output, y.output]) 

    # Hidden Layer(s)
    z = Dense(units='16', activation='relu')(combined)

    #Output Layer(s)
    output_layer_lst = []
    for output in list_of_outputs:
        output_layer_lst.append(Dense(units='1', name=output)(z))

    #create tf model object
    model = Model(inputs=[x.input, y.input],outputs=output_layer_lst)

    return model


def plot_loss(history):
    # Plot trainig and validation loss over epochs
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)

    plt.figure()

    plt.plot(epochs, loss, color = 'blue', label='Training loss')
    plt.plot(epochs, val_loss, color = 'green', label='Validation loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


def plot_train_rmse(history):
    # Plot training root-mean-squared error over epochs
    rmse = '_root_mean_squared_error'
    plt.figure()
    epochs = range(1, len(history.history[list_of_outputs[0] + rmse]) + 1)

    for output in list_of_outputs:
        output_rmse = history.history[output + rmse]
        plt.plot(epochs, output_rmse, label= output + ' rmse')

    plt.title('Training RMSE over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.legend()

    plt.show()


def plot_val_rmse(history):
    # Plot validation root-mean-squared error over epochs
    rmse = '_root_mean_squared_error'
    plt.figure()
    epochs = range(1, len(history.history['val' + '_' + list_of_outputs[0] + rmse]) + 1)

    for output in list_of_outputs:
        output_rmse = history.history['val' + '_' + output + rmse]
        plt.plot(epochs, output_rmse, label= output + ' rmse')

    plt.title('Validation RMSE over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.legend()

    plt.show()

if __name__ == "__main__":
    #reads data from the preprocessed csv file
    df = pd.read_csv('solar_load_weatherdata.csv')
    list_of_outputs = ['air1', 'clotheswasher1', 'dishwasher1', 'furnace1', 'refrigerator1', 'solar']
    list_of_outputs = ['air1','solar']
    train_frac = .6
    val_frac = .2

    #Splits the data into train, val, and test
    train_indexes, val_indexes, test_indexes = split_data(df, train_frac, val_frac)

    #Creates the Datasets
    train_power, train_temp, train_y = create_dataset(df, indexes = train_indexes)
    val_power, val_temp, val_y = create_dataset(df, indexes = val_indexes)

    #Turns output data from a dataframe to five arrays
    foo1 = []
    foo2 = []
    for output in list_of_outputs:
        foo1.append(train_y.pop(output)) 
        foo2.append(val_y.pop(output))
    train_y = foo1
    val_y = foo2

    # Create Model
    model = create_LSTM_model(list_of_outputs)

    # Specify the optimizer, and compile the model with loss functions for both outputs
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
    loss_dict = {}
    metrics_dict = {}
    for output in list_of_outputs:
        loss_dict[output] = 'mse'
        metrics_dict[output] = tf.keras.metrics.RootMeanSquaredError()
    model.compile(optimizer=optimizer, loss = loss_dict, metrics = metrics_dict)

    # Train the model for 100 epochs
    history = model.fit([train_power, train_temp], train_y,
                        epochs=100, batch_size=10, validation_data=([val_power, val_temp], val_y))

    # Print model summary and export to take_two_modelsummary.txt
    print(model.summary())
    with open('take_two_modelsummary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    plot_loss(history)
    plot_train_rmse(history)
    plot_val_rmse(history)

    # Save model
    model.save_weights('./model.ckpt')
