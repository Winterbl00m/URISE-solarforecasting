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
from tensorflow.keras.utils import plot_model

from tkinter import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

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


def create_dataset(df, indexes, list_of_outputs):
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
        power_input_row = df.loc[initial_index:index]['Sum of Power'].tolist()
        temp_input_row = df.loc[initial_index:index]['temperature'].tolist()

        #Checks to make sure that there is complete data
        if (len(power_input_row) == len(temp_input_row) == NUM_SAMPLES):

            #add input rows to input dataframes
            power_input_df.loc[len(power_input_df)] = power_input_row
            temp_input_df.loc[len(temp_input_df)] = temp_input_row

            #adds ouput data to output dataframe
            output_row = df.loc[[index]]
            output_df = output_df.append(output_row, ignore_index=True)


    #Turns output data from a dataframe to a list of one column dataframes
    output = []
    for item in list_of_outputs:
        output.append(output_df.pop(item)) 

    #return the two input dataframes and the output dataframe
    return power_input_df, temp_input_df , output 


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
    # Plot training and validation loss over epochs
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


def plot_train_rmse(history, list_of_outputs):
    # Plot training root-mean-squared error over epochs
    rmse = 'root_mean_squared_error'
    plt.figure()
    if len(list_of_outputs) > 1:
        epochs = range(1, len(history.history[list_of_outputs[0] + '_' + rmse]) + 1)
        for output in list_of_outputs:
            output_rmse = history.history[output + '_' + rmse]
            plt.plot(epochs, output_rmse, label= output + ' rmse')

    else:
        epochs = range(1, len(history.history[rmse]) + 1)
        for output in list_of_outputs:
            output_rmse = history.history[rmse]
            plt.plot(epochs, output_rmse, label= output + ' rmse')

    plt.title('Training RMSE over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    plt.show()


def plot_val_rmse(history, list_of_outputs):
    # Plot validation root-mean-squared error over epochs
    rmse = 'root_mean_squared_error'
    plt.figure()

    if len(list_of_outputs) > 1:
        epochs = range(1, len(history.history['val' + '_' + list_of_outputs[0] + '_' + rmse]) + 1)
        for output in list_of_outputs:
            output_rmse = history.history['val' + '_' + output + '_' + rmse]
            plt.plot(epochs, output_rmse, label= output + ' rmse')

    else:
        epochs = range(1, len(history.history['val' +  '_' + rmse]) + 1)
        for output in list_of_outputs:
            output_rmse = history.history['val' + '_' + rmse]
            plt.plot(epochs, output_rmse, label= output + ' rmse')


    plt.title('Validation RMSE over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.legend()

    plt.show()


def plot(history, list_of_outputs):
    # the figure that will contain the plot
    fig = Figure(figsize = (5, 5))

    # adding the subplots
    loss_plot = fig.add_subplot(311)
    train_rmse_plot = fig.add_subplot(312)
    val_rmse_plot = fig.add_subplot(313)

    # Plot training and validation loss over epochs
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    rmse = 'root_mean_squared_error'

    # plotting loss
    loss_plot.plot(epochs, loss, color = 'blue', label='Training loss')
    loss_plot.plot(epochs, val_loss, color = 'green', label='Validation loss')
    loss_plot.title.set_text('Training and Validation Loss over Epochs')
    loss_plot.set_xlabel('Epochs')
    loss_plot.set_ylabel('Loss')
    loss_plot.legend()

	# plotting RMSE
    if len(list_of_outputs) > 1:
        for output in list_of_outputs:
            output_rmse = history.history[output + '_' + rmse]
            train_rmse_plot.plot(epochs, output_rmse, label= output + ' rmse')
            val_output_rmse = history.history['val' + '_' + output + '_' + rmse]
            val_rmse_plot.plot(epochs, val_output_rmse, label= output + ' rmse')
    else:
        for output in list_of_outputs:
            output_rmse = history.history[rmse]
            train_rmse_plot.plot(epochs, output_rmse, label= output + ' rmse')
            val_output_rmse = history.history['val' + '_' + rmse]
            val_rmse_plot.plot(epochs, val_output_rmse, label= output + ' rmse')

    train_rmse_plot.title.set_text('Training RMSE over Epochs')
    train_rmse_plot.set_xlabel('Epochs')
    train_rmse_plot.set_ylabel('Loss')
    train_rmse_plot.legend()

    val_rmse_plot.title.set_text('Validation RMSE over Epochs')
    val_rmse_plot.set_xlabel('Epochs')
    val_rmse_plot.set_ylabel('Loss')
    val_rmse_plot.legend()

    # creating the Tkinter canvas
    # containing the Matplotlib figure
    canvas = FigureCanvasTkAgg(fig, master = window)
    canvas.draw()

    # placing the canvas on the Tkinter window
    canvas.get_tk_widget().pack()


if __name__ == "__main__":
    #reads data from the preprocessed csv file
    df = pd.read_csv('solar_load_weatherdata.csv')
    list_of_outputs = df.columns.tolist()
    list_of_outputs.remove('Time')
    list_of_outputs.remove('grid')
    list_of_outputs.remove('Sum of Power')
    list_of_outputs.remove('temperature')
    list_of_outputs.remove('summary')

    for item in list_of_outputs:
        if df[item][0] == None:
            list_of_outputs.remove(item)
    print(list_of_outputs)
    list_of_outputs = ['air1', 'clotheswasher1', 'dishwasher1', 'furnace1', 'refrigerator1', 'solar']
    train_frac = .6
    val_frac = .2

    #Splits the data into train, val, and test
    train_indexes, val_indexes, test_indexes = split_data(df, train_frac, val_frac)

    #Creates the Datasets
    train_power, train_temp, train_y = create_dataset(df, indexes = train_indexes, list_of_outputs = list_of_outputs)
    val_power, val_temp, val_y = create_dataset(df, indexes = val_indexes, list_of_outputs = list_of_outputs)

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

    
    # the main Tkinter window
    window = Tk()

    # setting the title
    window.title('Plotting in Tkinter')

    # dimensions of the main window
    window.geometry("500x1500")

    # button that displays the plot
    plot_button = Button(master = window,
                        command = plot(history, list_of_outputs),
                        height = 2,
                        width = 10,
                        text = "Plot")

    # place the button
    # in main window
    plot_button.pack()

    # run the gui
    window.mainloop()

    # plot_loss(history)
    # plot_train_rmse(history, list_of_outputs)
    # plot_val_rmse(history, list_of_outputs)

    # Save model
    # model.save_weights('./model.ckpt')

    # Plot model inputs and outputs in block form
    # import os
    # os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'

    # plot_model(model, to_file='model.png')

