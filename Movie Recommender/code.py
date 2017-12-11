import pandas as pd
import numpy as np
import math
import time
import preprocess as pp
import recommender as rc
import evaluator as ev

t0 = time.time()
# Checks to see if the data file that algorithms will use exists.
filename = "data.csv"
data_file = Path(filename)
if not data_file.is_file():
    rearrange_data(filename)
t1 = time.time()
print("Processing raw data took:" + str(t1 - t0))

# Reads the needed data array from file.
df = pd.read_csv(filename)
df_red = pp.reduce_cluster(df, 10)
t2 = time.time()
print("Loading data took:" + str(t2 - t1))

# Splitting data to training and test sets.
train, test = ev.split_data(df_red, 0.2)
t3 = time.time()
print("Splitting data took:" + str(t3 - t2))

# Removing some values from test set.
newtest, ans = ev.corruptor(test, 1)
t4 = time.time()
print("Processing test set took:" + str(t4 - t3))

# Evaluate performance of both svd and collaborative filtering methods.
ev.evaluator(train, newtest, ans)
t5 = time.time()
print("Evaluating took:" + str(t5 - t4))
