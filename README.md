# CS235_Data_Mining

## Crawler and Hadoop
This project involves finding the location of conferences in data mining, machine learning, databases, and AI. This data is then cleaned and analyzed for visualization purposes. There are two parts for this project.

### Part 1
Build a crawler to crawl WikiCFP for the conferences and location every year. From this data, we use OpenRefine to clean the data obtained. This part is detailed in `report1.pdf` with the crawler in `Scraper.java`, the resulting crawled data in the four `.txt` files indicating the conference type, and the cleaned data in `final_data.tsv`.

### Part 2
Use Hadoop to compute various statistics, and then use a visualization tool to create a heatmap of number of conferences for each city over time. This part is detailed in `report2.pdf` with the code in the `hadoop` folder (`.java` file is the hadoop file that is converted to `.jar` and the respective `_ans` file is the results of hadoop. The visualization portion uses Google Drive Fusion Table app to help visualize the heatmap after some post-processing after hadoop using python and OpenRefine.

## Movie Recommender

### Overview
Developed a movie recommender that returns a specified number of movies given the MovieLens data (10k ratings used here) and a target user's ratings of movies he has seen (not seen is specified as 0). This algorithm implements clustering, singular value decomposition, and collaborative filtering (through k-nearest neighbors).

### Preprocessing (`preprocess.py`)
Transformed data from MovieLens into a 2D array (user-by-movie) of user rows and movie columns with ratings for values (0 for unrated). Provides the clustering algorithm used in the recommendation process. Clusters the rows from a given matrix (m-by-n) based on k centers that are determined by choosing users with the most ratings (results show that this was not a good idea since users with most ratings can still be relatively close to each other distance wise therefore centers should have been chosen by fartest users from each other distance wise). After centers are chosen, uses distance formula (Euclidean and Cosine Similarity compared) to classify known data points (users) into each clusters and calculates the cluster average ratings as the new matrix (k-by-n). (Transpose matrix and set as function argument to cluster columns and transpose back.)

### Recommendation (`recommender.py`)
Given a 2D matrix of users and movies, provides a recommendation of a movie the user has not seen yet from the available sources. Provides two algorithms to generate predicted user matrix through singular value decomposition and collaborative filterting (for k-nearest neighbors). (*This has a bug where after matrix reduction the recommender will return the best rated group of movies instead of a movie*).

### Postprocessing (`evaluator.py`)
Provides functions for testing recommendator. Given a testing data set (that was separated from the known data), the corruptor genearates a testing data by removing the highest rating and returning the new test matrix and the information on the removed rating. The evaluator takes in the testing data and answers to evaluate a specified algorithm by generating an averaged squared error between the rating that was removed from the corruptor and the predicted rating generated from the recommender.

### Analysis (`analysis.py`)
Uses graphing functions in Python to generate graphs used for report.

### Example Code (`maincode.py` and `demo.py`)
Some example implementations of the code used. Maincode was used to generate the errors that were included in the report and graphs. Demo provides a short demo of how the recommender works (the numbers can be changed to change the outcome).
