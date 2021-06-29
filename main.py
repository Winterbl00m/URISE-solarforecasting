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

def create_model(learning_rate, feature_layer):
    """
    Creates an ANN model

    learning_rate: float. The learning rate of the model. Between 0 and 1
    feature_layer: A tf layer.  Aka the input layer. The first layer of the model. 
    """

    # Add the layer containing the features to the model.
    model.add(feature_layer)

    #TODO: Define Hidden Layers

    #TODO: Define output layer

    #TODO: compile model

    #TODO: when done delte pass and uncomment out
    #return model
    pass 


def train_model():
    """Train the model by feeding it data."""
    pass 