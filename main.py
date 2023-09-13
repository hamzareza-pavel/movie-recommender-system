#author: Hamza Reza Pavel
from random import seed
from csv import reader
import distance_measures as dm
import processdataset as pds
import knn
from fuzzywuzzy import fuzz
import time
import numpy as np



def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

def fuzzy_matching(hashmap, fav_movie):
        match_tuple = []
        # get match
        for title, idx in hashmap.items():
            ratio = fuzz.ratio(title.lower(), fav_movie.lower())
            if ratio >= 60:
                match_tuple.append((title, idx, ratio))
        # sort
        match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
        return match_tuple

def main():
    seed(1)
    print("Reading data from movies.csv and ratings.csv")
    mat, hmap, factorized_mat = pds.readMovieData('movies.csv', 'ratings.csv')

    print("Enter a movie name.......")
    #input_moviename = input()
    input_moviename = 'toy story'
    res = fuzzy_matching(hmap, input_moviename)
    print(f"input movie name could be {res}")

    reverse_hashmap = {v: k for k, v in hmap.items()}
    dataset = mat.toarray()
    testdata = []
    testdata.append(dataset[res[0][1]])
    traindata = dataset.tolist()

    start_time = int(round(time.time() * 1000))

    movie_recs_knn = knn.k_nearest_neighbors(traindata, testdata, dm.euclidean_distance, 10)

    print(f"Recommended movies for {input_moviename} are:")
    for i in movie_recs_knn[0]:
        print(reverse_hashmap[i])

    end_time = int(round(time.time() * 1000))

    print(f"output generated in {end_time - start_time} mili sec")

    #output using vectorized mat
    corr = np.corrcoef(factorized_mat)
    print(corr.shape)
    inputmovieid = res[0][1]
    corr_inputmovieid = corr[inputmovieid]
    #print(corr_coffey_hands)

    #accuricy = knn.evaluate_algorithm(traindata, knn.k_nearest_neighbors, 10,dm.euclidean_distance, 10)
    #print(f"accuricy for knn using 10 fold validation is {accuricy}")

main()
