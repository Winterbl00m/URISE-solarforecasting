# Importing the libraries
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import random 
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, SimpleRNN
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

        input_row = df.loc[initial_index:index]['Sum of Power'].tolist() #+ df.loc[initial_index:index]['temperature'].tolist()

        if len(input_row) == NUM_SAMPLES:
            input_df.loc[len(input_df)] = input_row

            output_row = df.loc[[index]]
            output_df = output_df.append(output_row, ignore_index=True)
  
    output_df.pop('Time')
    output_df.pop('Sum of Power')
    output_df.pop('temperature')

    print(output_df.head())
    return input_df, output_df 


def create_model():
    """
    Adapted from https://github.com/katanaml/sample-apps/blob/master/04/multi-output-model.ipynb
    Returns
    model : a tf model with one or more output layers
    
    """
    # Input Layer
    input_layer = Input(shape=(NUM_SAMPLES,))

    #Hidden Layer(s)
    first_hidden_layer = Dense(units='128', activation='relu')(input_layer)

    #Output Layer(s)
    y1_output = Dense(units='1', name='air1')(first_hidden_layer)
    y2_output = Dense(units='1', name='clotheswasher1')(first_hidden_layer)
    y3_output = Dense(units='1', name='dishwasher1')(first_hidden_layer)
    y4_output = Dense(units='1', name='furnace1')(first_hidden_layer)
    y5_output = Dense(units='1', name='refrigerator1')(first_hidden_layer)
    y6_output = Dense(units='1', name='solar')(first_hidden_layer)

    #create tf model object
    model = Model(inputs=input_layer,outputs=[y1_output, y2_output, y3_output, y4_output, y5_output, y6_output])

    # Specify the optimizer, and compile the model with loss functions for both outputs
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss={'air1': 'mse',
                  'clotheswasher1': 'mse',
                  'dishwasher1' : 'mse',
                  'furnace1' : 'mse',
                  'refrigerator1' : 'mse',
                  'solar' : 'mse'},

                  metrics={'air1': tf.keras.metrics.RootMeanSquaredError(),
                  'clotheswasher1': tf.keras.metrics.RootMeanSquaredError(),
                  'dishwasher1' : tf.keras.metrics.RootMeanSquaredError(),
                  'furnace1' : tf.keras.metrics.RootMeanSquaredError(),
                  'refrigerator1' : tf.keras.metrics.RootMeanSquaredError(),
                  'solar' : tf.keras.metrics.RootMeanSquaredError()} )

    return model


#reads data from the preprocessed csv file
df = pd.read_csv('solar_load_weatherdata.csv')

train_frac = .6
val_frac = .2

#Splits the data into train, val, and test
train_indexes, val_indexes, test_indexes = split_data(df, train_frac, val_frac)

#Creates the Datasets
train_x, train_y = create_dataset(df, indexes = train_indexes)
val_x, val_y = create_dataset(df, indexes = val_indexes)

# list_of_outputs = ['air1', 'clotheswasher1', 'dishwasher1', 'furnace1', 'refrigerator1']

#Turns output data from a dataframe to five arrays
train_y = train_y.pop('air1') , train_y.pop('clotheswasher1'), train_y.pop('dishwasher1'), train_y.pop('furnace1'), train_y.pop('refrigerator1'), train_y.pop('solar')
val_y = val_y.pop('air1') , val_y.pop('clotheswasher1'), val_y.pop('dishwasher1'), val_y.pop('furnace1'), val_y.pop('refrigerator1'), val_y.pop('solar')

#Create Model
model = create_model()

# Train the model for 100 epochs
history = model.fit(train_x, train_y,
                    epochs=100, batch_size=10, validation_data=(val_x,val_y))

# Print model summary and export to take_two_modelsummary.txt
print(model.summary())
with open('take_two_modelsummary.txt', 'w') as f:

    model.summary(print_fn=lambda x: f.write(x + '\n'))

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

# Plot training root-mean-squared error over epochs
air_rmse = history.history['air1_root_mean_squared_error']
clotheswasher_rmse = history.history['clotheswasher1_root_mean_squared_error']
dishwasher_rmse = history.history['dishwasher1_root_mean_squared_error']
furnace_rmse = history.history['furnace1_root_mean_squared_error']
refrigerator_rmse = history.history['refrigerator1_root_mean_squared_error']
solar_rmse = history.history['solar_root_mean_squared_error']

epochs = range(1, len(air_rmse) + 1)

plt.figure()

plt.plot(epochs, air_rmse, color = 'blue', label='Air RMSE')
plt.plot(epochs, clotheswasher_rmse, color = 'green', label='Clotheswaher RMSE')
plt.plot(epochs, dishwasher_rmse, color = 'red', label='Dishwasher RMSE')
plt.plot(epochs, furnace_rmse, color = 'purple', label='Furnace RMSE')
plt.plot(epochs, refrigerator_rmse, color = 'orange', label='Refrigerator RMSE')
plt.plot(epochs, solar_rmse, color = 'gold', label='Solar RMSE')

plt.title('Training RMSE over Epochs')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.legend()

plt.show()

# Plot validation root-mean-squared error over epochs
val_air_rmse = history.history['val_air1_root_mean_squared_error']
val_clotheswasher_rmse = history.history['val_clotheswasher1_root_mean_squared_error']
val_dishwasher_rmse = history.history['val_dishwasher1_root_mean_squared_error']
val_furnace_rmse = history.history['val_furnace1_root_mean_squared_error']
val_refrigerator_rmse = history.history['val_refrigerator1_root_mean_squared_error']
val_solar_rmse = history.history['val_solar_root_mean_squared_error']

epochs = range(1, len(val_air_rmse) + 1)

plt.figure()

plt.plot(epochs, val_air_rmse, color = 'blue', label='Air RMSE')
plt.plot(epochs, val_clotheswasher_rmse, color = 'green', label='Clotheswaher RMSE')
plt.plot(epochs, val_dishwasher_rmse, color = 'red', label='Dishwasher RMSE')
plt.plot(epochs, val_furnace_rmse, color = 'purple', label='Furnace RMSE')
plt.plot(epochs, val_refrigerator_rmse, color = 'orange', label='Refrigerator RMSE')
plt.plot(epochs, val_solar_rmse, color = 'gold', label='Solar RMSE')

plt.title('Validation RMSE over Epochs')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.legend()

plt.show()