from random import randrange

def dataset_minmax(dataset):
	minmax = list()
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset]
		value_min = min(col_values)
		value_max = max(col_values)
		minmax.append([value_min, value_max])
	return minmax

def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for _ in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0


def evaluate_algorithm(dataset, algorithm, n_folds, distancefunc, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, distancefunc, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores

def get_neighbors(train, test_row, distancefunc, num_neighbors):
	distances = list()
	for i in range(len(train)):
		dist = distancefunc(test_row, train[i])
		distances.append(( train[i], dist, i)) # i is the id of the movie hence needs to be included
		print(f"Compared test data with {i+1} row in training set")
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i])
	return neighbors


def predict_classification(train, test_row, distancefunc, num_neighbors):
	neighbors = get_neighbors(train, test_row, distancefunc, num_neighbors)
	output_values = [row[-1] for row in neighbors]
	#prediction = max(set(output_values), key=output_values.count)
	#print(len(prediction))
	return output_values


def k_nearest_neighbors(train, test, distancefunc, num_neighbors):
	predictions = list()
	print(f"starting prediction using KNN for training set size of {len(train)}")
	for row in test:
		output = predict_classification(train, row, distancefunc, num_neighbors)
		predictions.append(output)
	return(predictions)


