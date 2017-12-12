from pathlib import Path
import pandas as pd
import numpy as np
import math
import time
import preprocess as pp
import recommender as rc
import evaluator as ev

def run_code(usr_grp, mov_grp, alg, k):
	t0 = time.time()
	# Checks to see if the data file that algorithms will use exists.
	filename = "data.csv"
	data_file = Path(filename)
	if not data_file.is_file():
	    rearrange_data(filename)
	t1 = time.time()
	print("Processing raw data took: " + str(t1 - t0))

	# Reads the needed data array from file.
	df = pd.read_csv(filename)
	df_red = pp.reduce_cluster(df, usr_grp)
	df_red = df_red.iloc[:, 1:]
	df_red.columns = df_red.columns.astype(int)
	df_red = pp.reduce_cluster(df_red.transpose(), mov_grp)
	df_red = df_red.transpose()
	t2 = time.time()
	print("Loading and preprocess data took: " + str(t2 - t1))

	# Splitting data to training and test sets.
	train, test = ev.split_data(df_red, 0.9)
	t3 = time.time()
	print("Splitting data took: " + str(t3 - t2))

	# Removing some values from test set.
	newtest, ans_col, ans_rate = ev.corruptor(test, 1)
	t4 = time.time()
	print("Processing test set took: " + str(t4 - t3))

	# Evaluate performance of both svd and collaborative filtering methods.
	if alg == "both":
		ev.evaluator(train, newtest, ans_col, ans_rate, "collab", k)
		t5 = time.time()
		print("Evaluating with collab took: " + str(t5 - t4))
		ev.evaluator(train, newtest, ans_col, ans_rate, "svd", k)
		t5 = time.time()
		print("Evaluating with svd took: " + str(t5 - t4))
	else:
		ev.evaluator(train, newtest, ans_col, ans_rate, alg, k)
		t5 = time.time()
		print("Evaluating took: " + str(t5 - t4))
	print("Total Time Running: " + str(t5 - t0))

run_code(20, 20, "both", 10)
