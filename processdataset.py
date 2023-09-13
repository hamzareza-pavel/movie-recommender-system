#author: Hamza Reza Pavel
# This file has utility methods to preprocess the dataset. To eliminate outliers, duplicates, and convert string data to numerical data.
import pandas as pd
from scipy.sparse import csr_matrix
import sys
import gc
import sklearn
from sklearn.decomposition import TruncatedSVD


def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())


def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup


def isfloat(x):
	if x.find(".") == -1:
		return False
	else:
		try:
			a = float(x)
		except ValueError:
			return False
		else:
			return True


def isint(x):
    try:
        a = float(x)
        b = int(a)
    except ValueError:
        return False
    else:
        return a == b


def process_data_set(dataset):
	attributemap = list()
	for i in range(len(dataset[0])):
		if isfloat(dataset[0][i]):
			str_column_to_float(dataset, i)
		else:
			lookup = str_column_to_int(dataset,i)
			attributemap.append((lookup, i))
	df = pd.DataFrame(dataset)
	return attributemap, df


def readMovieData(movieFileName, ratingFileName):

	mv_rating_lim = 100 #consider the movies having at least 50 ratings
	usr_rating_lim = 100 #consider the users who gave at least 50 ratings


	dataset_movies = pd.read_csv(movieFileName, usecols=['movieId', 'title'],dtype={'movieId': 'int32', 'title': 'str'})
	dataset_ratings = pd.read_csv(ratingFileName, usecols=['userId', 'movieId', 'rating'],dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})

	moviecount = pd.DataFrame(dataset_ratings.groupby('movieId').size(), columns=['count'])
	popular_movies = list(set(moviecount.query('count >= @mv_rating_lim').index))  # noqa
	movies_filter = dataset_ratings.movieId.isin(popular_movies).values

	df_users_cnt = pd.DataFrame(dataset_ratings.groupby('userId').size(),columns=['count'])
	active_users = list(set(df_users_cnt.query('count >= @usr_rating_lim').index))  # noqa
	users_filter = dataset_ratings.userId.isin(active_users).values

	df_ratings_filtered = dataset_ratings[movies_filter & users_filter]
	movie_user_mat = df_ratings_filtered.pivot(index='movieId', columns='userId', values='rating').fillna(0)
	hashmap = { movie: i for i, movie in enumerate(list(dataset_movies.set_index('movieId').loc[movie_user_mat.index].title)) }
	movie_user_mat_sparse = csr_matrix(movie_user_mat.values)

	#matrix vectorization of data
	SVD = TruncatedSVD(n_components = 20, random_state=17)
	factorized_mat = SVD.fit_transform(movie_user_mat)
	#vectorized matrix shape
	print(f"Vectorized matrix shape {factorized_mat.shape}")

	print(f"data matrix shape is {movie_user_mat.shape}")
	print(f"sparse matrix shape is {movie_user_mat_sparse.shape}")
	del dataset_movies, moviecount, df_users_cnt
	del dataset_ratings, df_ratings_filtered, movie_user_mat
	gc.collect()

	return movie_user_mat_sparse, hashmap, factorized_mat
