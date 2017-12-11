import pandas as pd
import numpy as np
import math
import datetime
from pathlib import Path
import matplotlib.pyplot as plt

# This function takes the files from MovieLens and transforms the data to a 2D array (row = user, column = movie, value = rating).
# This stores the new array in a '.csv' file
def rearrange_data(filename):
    data = pd.read_csv("data/ratings.csv")

    # Create new array where userId is the row labels and movieId is the columns
    df = pd.DataFrame(index=data.userId.unique(), columns=data.movieId.unique())
    df = df.fillna(0)

    # Iterates old array and fills the new array.
    for index, row in data.iterrows():
        df.set_value(row['userId'], row['movieId'], row['rating'])

    # Creates new csv file for the new array so we don't have to generate constantly.
    df.to_csv(filename)

# Calculates the distance between two vectors using cosine similarity.
def cos_dist(data1, data2):
    numer = data1.values[0, 1:].dot(data2.values[0, 1:].transpose())
    denom = math.sqrt(np.square(data1.values[0, 1:]).sum() * np.square(data2.values[0, 1:]).sum())
    return float(numer / denom)

# Calculates the distance between two vectors using Euclidean distance.
def euc_dist(data1, data2):
    return math.sqrt(np.square(data1.values[0, 1:] - data2.values[0, 1:]).sum())

# Reduce dimensionality of data through k-means clustering.
def reduce_cluster(data, k):

    # Select the top k users who have the most ratings.
    top_k = (data != 0).sum(axis=1).sort_values(ascending=False)[:k].index.values.tolist()
    
    # Creating clusters.
    clusters = []
    for i in range(0, k):
        clusters.append([])

    # Groups into k different clusters.
    for row in data.index.values.tolist():
        temp = []
        for top in top_k:
            temp.append(cos_dist(data.loc[[row]], data.loc[[top]]))
        clusters[temp.index(min(temp))].append(int(row))

    # Averages the cluster points and creates a new dataframe from averages.
    df = pd.DataFrame()
    for c in clusters:
        df = df.append(data.loc[c, :].iloc[:, 1:].mean().to_frame().transpose(), ignore_index=True)

    return df



# --------------------------------------------------------------------
