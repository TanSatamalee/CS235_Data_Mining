import pandas as pd
import datetime
from pathlib import Path

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

# --------------------------------------------------------------------

# Checks to see if the data file that algorithms will use exists.
filename = "data.csv"
data_file = Path(filename)
if not data_file.is_file():
	rearrange_data(filename)
