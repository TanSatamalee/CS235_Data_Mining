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

# Given a new 'user' and the matrix decompositions (U, S, V) gives a new recommendation.
# k is how many features are kept when doing dimension reduction.
def recommend(d, user, k=0):
	# Checks arguments.
	if k > d.shape[1]:
		print("Error in k argument of recommend.")
		return None

	# Adds the new user to given data.
	data = d.append(user.transpose(), ignore_index=True) * 2
	data = data.as_matrix()

	# Gets the matrix decomposition
	data_mean = np.mean(data, axis=1)
	data = data - data_mean.reshape(-1, 1)
	U, S, V = svd_alg(data)

	# Produce the reduced matrices.
	Uk = U[:, :k]
	Vk = V[:k, :]
	Sk = S[:k, :k]
	
	# Creating the Prediction Matrix
	pred_mat = np.absolute(np.rint((np.dot(np.dot(Uk, Sk), Vk) + data_mean.reshape(-1, 1))) / 2)
	print(pred_mat)


# --------------------------------------------------------------------

# For testing purposes.
filename = "data.csv"
df = pd.read_csv(filename)
# data = pp.preprocess(10)
recommend(pd.DataFrame([[5, 2, 0, 4, 1], [5, 4, 0, 5, 1], [5, 2, 0, 5, 1]]), pd.DataFrame([0, 2, 0, 5, 0]), 2)
