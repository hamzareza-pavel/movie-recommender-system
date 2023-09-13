#author: Hamza Reza Pavel
import math
from scipy.spatial import distance as dst


# Calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)):
		distance += (row1[i] - row2[i])**2
	return math.sqrt(distance)


def manhattan_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)):
		distance += abs(row1[i] - row2[i])
	return distance

def hamming_distance(row1, row2):
	trow1 = row1[:-1]
	trow2 = row2[:-2]
	return len(list(filter(lambda x : int(x[0])^int(x[1]), zip(trow1, trow2))))


def minkowski_distance(row1, row2, p = 3):
	distance = 0.0
	for i in range(len(row1)):
		distance += (abs(row1[i] - row2[i]))**p
	return distance**(1/float(p))


# disttype is the type of the distance metrics one of the following: 'euclidean', 'minkowski', 'cityblock' for manhattan, or 'hamming'
def genericVectorizedDistance(testrow, dataset, disttype, p = 3):
	distances = list()
	if disttype == 'minkowski':
		distances = dst.cdist(testrow, dataset, disttype, p)
	else:
		distances = dst.cdist(testrow, dataset, disttype)
	return distances


