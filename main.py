import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 

def create_feature_layer(filename, current_time):
    """
    Creates a tf layer with the feature layer

    filename: the name of the file with the data in it 
    current_time : the time we want to get the features of
    """
    pass

def create_model(learning_rate, feature_layer, N_LABELS):
    """
    Creates an ANN model

    learning_rate: float. The learning rate of the model. Between 0 and 1
    feature_layer: A tf layer.  Aka the input layer. The first layer of the model. 
    """
    #Create Model
    model = keras.Sequential()

    # Add the layer containing the features to the model.
    model.add(feature_layer)

    #Hidden Layers(RNN)
    model.add(layers.SimpleRNN(128))

    model.add(layers.SimpleRNN(128))

    #Output layer
    model.add(layers.Dense(N_LABELS, activation='sigmoid', name='output'))

    #Compile Model
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
        loss='binary_crossentropy',
        metrics=[tf.keras.losses.BinaryCrossentropy()])

    return model
     


def train_model():
    """Train the model by feeding it data."""
    pass 