import pandas as pd
import numpy as np
import math
import preprocess as pp
import recommender as rc

# Splits a dataframe into two dataframe in ratio of p and (1-p) where p <= 1.
def split_data(d, p):
	if p < 0 or p > 1:
		print("Error with p during split.")
		return None

	# Determine the amount for first split.
	rows = d.shape[0]
	data = d.as_matrix()
	first_sp = int(round(rows * p))

	# Shuffle the data
	np.random.seed(0)
	select = np.arange(rows)
	np.random.shuffle(select)
	train = select[:first_sp]
	test = select[first_sp:]
	
	return d.iloc[train], d.iloc[test]

# Given a user, deletes the top n entries the user has rated and returns new user and index & rating of what was deleted.
def corruptor(users, n):
	deleted_col = []
	deleted_rate = []
	for row in users.index.values.tolist():
		temp = users.loc[row, :].as_matrix()
		to_delete = temp.argsort()[-n:][::-1]
		dc = users.columns.values[to_delete]
		deleted_col.append(dc.tolist())
		deleted_rate.append(users.loc[row][dc])
		
		users.loc[row][dc] = 0

	return users, deleted_col, deleted_rate

# Prints and returns a percentage of correct answers for svd and collab given a set of train, test and answers of data.
def evaluator(train, test, ans_col, ans_rate, alg, k=10):
	n = 0
	err = 0
	if alg == "svd":
		fxn = rc.svd_recommend
	else:
		fxn = rc.collab_recommend

	for row in test.index.values.tolist():
		user = pd.DataFrame(test.loc[row, :]).transpose()
		pred = fxn(train, user, k)
		err += (ans_rate[n][ans_col[n][0]] - pred[ans_col[n][0]]) ** 2
		n += 1
	
	err = float(math.sqrt(err) / n)
	print("The error for " + alg + " is: " + str(err))
	return err

# --------------------------------------------------------------------
