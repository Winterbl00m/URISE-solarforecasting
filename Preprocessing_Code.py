# preprocessing code for pecan street disaggregation
# operates in Python 3.8.5 ('base': conda)
# leverages code from https://github.com/pipette/Electricity-load-disaggregation.git
# references also include https://github.com/gissemari/Disaggregation-PecanStreet.git

import numpy as np
import pandas as pd
import itertools
from sklearn.cluster import KMeans
# K-means algorithm
from sklearn.metrics import silhouette_score
# K-means analysis through silhouette score (computes mean silhouette coefficient of all samples)
from sklearn.preprocessing import Normalizer
# used to normalize data

# import os

# path = 'C:/Users/aaris/Downloads/RNN-Preprocessing'
# dirs = os.listdir(path)
# for file in dirs:
#     print(file)

pecanstreetdataset = pd.read_csv('C:/Users/aaris/Downloads/RNN-Preprocessing/15minute_data_austin.csv')
print(pecanstreetdataset.loc['air1'])
# pecanstreetdataset.loc[['air1', 'clotheswasher1', 'dishwasher1', 'furnace1', 'refrigerator1']]
# pecanstreetdataset.loc[['house1', 'house2', 'house3', 'house4', 'house5']]

class appliance():
    # defines a class for the appliances
    def __init__(self, name, power_data):
        pdata = Normalizer().fit(power_data)
        normalized_power_data = pdata.transform(power_data)
        # Normalizer rescales the values between 0 and 1 for simpler computing
        self.name =  name
        self.power_data = normalized_power_data

# or use "from sklearn.model_selection import train_test_split"
# which splits arrays or matrices into random training and testing subsets
# use it twice to include a validation set as well

def train_test_split(dataframe, split, second_split = None):
    # splits dataframe into training and validation set
    # param dataframe: total dataframe
    # param split: date on which to split
    # param second_split: option to create second test set
    # return: train, test and optionally second test dataframe
    df = dataframe.fillna(value = 0,inplace = False)
    # dataframe.fillna is derived from pandas and it will fill holes in the data with value
    df['total'] = dataframe.sum(axis = 1)
    # dataframe.sum will return the sum of the values over the requested axis
    if second_split:
        # if a second test set exists (validation set)
        return df[:split], df[split:second_split], df[second_split:]
    else:
        return df[:split], df[split:]

def cluster(x_train, x_test, max_number_clusters):
    # iteratively finds an optimal number of clusters based on silhouette score (K-means clustering with silhouette analysis)
    # param data: N*K numpy array, in case of a 1D array supply a column vector N*1
    # param max_number_clusters: maximum number of clusters (value of K)
    # return: cluster centers
    highest_score = -1
    for i in range(2, max_number_clusters):
        # minimum of 2 clusters
        print("Fitting a K-means model with {} clusters".format(i))
        kmeans = KMeans(n_clusters = i).fit(x_train)
        # param n_clusters: number of clusters to form
        labels = kmeans.predict(x_test)
        print("Calculating silhouette score...")
        s_score = silhouette_score(x_test, labels, sample_size = 10000)
        # silhouette score calculates a value between -1 to 1, where the higher the value, the better the clustering
        # param x_test: array of piecewise distances between samples
        # param labels: predicted labels for each sample
        # param sample_size: the size of the sample to use when calculating the silhouette coefficient
        if s_score > highest_score:
            # if silhouette score is greater than the worst value possible
            highest_score = s_score
            # reset highest_score such that it can find the optimal value of i for the highest value of s_score
            centers = kmeans.cluster_centers_
            # kmeans.cluster_centers_ will return the coordinates of the center of the clusters
        print("Silhouette score with {} clusters:{}".format(i, s_score))
    print("Highest silhouette score of {} achieved with {} clusters\n".format(highest_score, len(centers)))
    return centers

def create_combined_states(df):
    new_df = df.copy()
    columns = new_df.columns
    column_combinations = []
    for i in range(2, len(columns)+1):
        column_combinations = column_combinations + list(itertools.combinations(columns, i))
        # itertools.combinations returns combinations of i length subsequences of elements from the input columns
    for x in column_combinations:
        name = " ".join(list(x))
        # .join concatenates any number of strings into one larger string
        new_df[name] = df[list(x)].sum(axis = 1)
        # dataframe.sum will return the sum of the values over the requested axis
    return new_df