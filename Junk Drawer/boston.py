# https://towardsdatascience.com/multi-output-model-with-tensorflow-keras-functional-api-875dd89aa7c6 
import pandas as pd
# Importing the Boston Housing dataset
from sklearn.datasets import load_boston


# Loading the Boston Housing dataset
boston = load_boston()

# Initializing the dataframe
data = pd.DataFrame(boston.data)
print(data.head())

#Adding the feature names to the dataframe
data.columns = boston.feature_names

#Adding target variable to dataframe
data['PRICE'] = boston.target
data.head()

# Split the data into train and test with 80 train / 20 test
train,test = train_test_split(data, test_size=0.2, random_state = 1)
train,val = train_test_split(train, test_size=0.2, random_state = 1)

