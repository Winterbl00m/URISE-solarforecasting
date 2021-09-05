# URISE-solarforecasting

## Getting Started
In order to run our model you will need tensorflow version 2.1.x (we used 2.1.0.) 
which can only be installed older versions of Python(we used python 3.7)

1. ### Installing TensorFlow
    1. To Begin you will need to download Python 3.7 from 
    https://www.python.org/downloads/release/python-370/ 
    if it is not already installed

    2. Then in a terminal you will need to install tensorflow 2.1.0 using pip install
    ```terminal
    pip install tensorflow==2.1.0
    ```
    Ensure that you are installing tensorflow with 3.7's version of pip.
    To check your version of pip you can type 
    ```terminal
    pip --version
    ```
2. ### Download Data
    1. Our model uses data from Pacan Street Dataport so you will need access
    https://dataport.pecanstreet.org/
    2. Download '15minute_data_austin.csv' from https://dataport.pecanstreet.org/academic under Austin 15-min
    3. Download 'metadata.csv' from https://dataport.pecanstreet.org/academic under Metadata Report
    4. Download 'weather.csv' from https://jupyterhub.pecanstreet.org/hub/login within ev_and_weather.zip
    

3. ### Run Model
    1. Run gui.py 
    2. Select which devices you wish to disaggregate
    3. Click Create Model
    4. TODO


## How the model works(If you are curious) 
### What is a Recurrent Neural Network 
### Diagram of Model 
#### Input Layers
#### Reshape
#### LSTM
#### Dense Layers
#### Output Layers
