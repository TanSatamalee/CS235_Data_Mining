import pandas as pd
import numpy as np
import math
import datetime
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

# Given a user, deletes the top n entries the user has rated and returns new user and index of what was deleted.
def corruptor(users, n):
	deleted_col = []
	for row in users.index.values.tolist():
		temp = users.loc[row, :].as_matrix()
		to_delete = temp.argsort()[-n:][::-1]
		dc = users.columns.values[to_delete]
		deleted_col.append(dc.tolist())
		
		users.loc[row][dc] = 0

	return users, deleted_col

# Prints and returns a percentage of correct answers for svd and collab given a set of train, test and answers of data.
def evaluator(train, test, answers):
	n = 0
	col_corr = 0
	svd_corr = 0
	for row in test.index.values.tolist():
		user = pd.DataFrame(test.loc[row, :]).transpose()
		col_res = rc.recommender(train, user, 1, "collab", 10)
		if col_res[0] == answers[n][0]:
			col_corr += 1
		svd_res = rc.recommender(train, user, 1, "svd", 10)
		if svd_res[0] == answers[n][0]:
			svd_corr += 1
		n += 1
	print([float(col_corr / n), float(svd_corr / n)])

# --------------------------------------------------------------------
