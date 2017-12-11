import pandas as pd
import numpy as np
import math
import datetime
import preprocess as pp

# Decomposes given matrix into three other matrices.
def svd_alg(A):
    # Calculating eigenvalues and vectors from data matrix A.
    eig_temp = np.linalg.eig(np.dot(A.T, A))
    ###eig = -np.sort(-eig_temp[0])
    eig = eig_temp[0]

    # Produce S from the sorted eigenvalues along the diagonal.
    S = np.sqrt(np.eye(len(eig)) * eig.T)
    S_inv = np.linalg.inv(S)

    # Produce V from sorting eigenvectors to align with sorted eigenvalues.
    argS = np.argsort(-eig_temp[0])
    ###V = ((eig_temp[1].T)[argS]).T
    V = eig_temp[1]

    # Produce U from the other three matrices.
    U = np.dot(np.dot(A, V), S_inv)

    return U, S, V

# Given a new 'user' and the data matrix gives a new recommendation by SVD.
# k is how many features are kept when doing dimension reduction.
def svd_recommend(d, user, k=0):
    # Checks arguments.
    if k > d.shape[1]:
        # Auto shifts the argument.
        k = d.shape[1] - 1

    # Adds the new user to given data.
    data = d.append(user, ignore_index=True) * 2
    data = data.as_matrix()

    # Gets the matrix decomposition
    data_mean = np.mean(data, axis=1)
    data = data - data_mean.reshape(-1, 1)
    U, S, V = svd_alg(data)
    '''
    # Produce the reduced matrices.
    Uk = U[:, :k]
    Vk = V[:k, :]
    Sk = S[:k, :k]
    '''
    # Creating the Prediction Matrix
    pred_mat = np.absolute(np.rint((np.dot(np.dot(U, S), V) + data_mean.reshape(-1, 1))) / 2)
    
    return pred_mat[-1:][0]

# Given a new 'user' and the data matrix gives a new recommendation by k-neighbors.
# k is how many neighbors to consider.
def collab_recommend(d, user, k=1):
    # Checks arguments.
    if k > d.shape[0]:
        # Auto shifts the argument.
        k = d.shape[0] - 1

    # Finds neighbors array.
    temp = []
    for row in d.index.values.tolist():
        temp.append(pp.cos_dist(d.loc[[row]], user))
       
    neighbors = np.array(temp).argsort()[-k:][::-1]
    nb = d.iloc[neighbors, :]

    return np.array(nb.mean()).T

# Returns n recommendations (by index)
def get_recommend(user, pred, n):
    cust = user.as_matrix()[0]
    z = np.where(cust == 0)[0]

    pred_rate = []
    for i in z:
        pred_rate.append(pred[i])

    return z[np.array(pred_rate).argsort()[-n:][::-1]]

# The central recommender that calls on the recommendation algorithms and provides actual recommendations.
# d: given data matrix, user: the ratings of requested user recommendation, n: number of recommendations
# alg: string for algorithm name for recommendation, k: argument for recommend algorithm
def recommender(d, user, n, alg, k):
    data = d
    if alg == "svd":
        pred = svd_recommend(data, user, k)
    elif alg == "collab":
        pred = collab_recommend(data, user, k)
    else:
    # Collaborative Filtering By Default
        pred = collab_recommend(data, user, k)

    # Gets the actual user recommendations.
    recc = get_recommend(user, pred, n)

    return d.columns.values[recc]

# --------------------------------------------------------------------

