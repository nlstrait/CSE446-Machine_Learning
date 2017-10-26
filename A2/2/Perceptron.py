import numpy as np
import csv


def get_data(filename):
	"""
	Fetches tsv data from specified file and returns numpy matrix
	:param filename: name of file to extract from
	:return: numpy array of tsv data
	"""
	data = np.loadtxt(filename, delimiter="\t")
	return data


def epoch(data, w=None):
	"""
	Perform one run over the data and tune our weight vector
	:param data: matrix of data, each row being an entry with y_i in data[i][0] and x_i in data[i][1:]
	:param w: weight vector; if None, is initialized as a zero vector
	:return: number of mistakes made during this epoch
	"""
	if w is None:
		w = np.zeros((data.shape[1] - 1))

	mistakes = 0

	for i in range(0, data.shape[0]):
		x_i = data[i][1:]
		y_i = data[i][0]
		y_hat = calc_y_hat(x_i, w)

		if y_hat != y_i or y_hat == 0:
			# entry is incorrectly classified, so let's adjust our weight vector
			w += y_i * x_i
			mistakes += 1

	return mistakes


def calc_y_hat(x, w):
	"""
	Calculates the value of y_hat using feature and weight vectors
	:param x: feature vector
	:param w: weight vector
	:return: y_hat; -1 if x * w < 0, +1 if x * w = 0, -1 otherwise
	"""
	dp = w.dot(x)
	if dp < 0:
		return -1
	elif dp == 0:
		return 0
	else:
		return 1


def calc_error_rate(w, data, num_decimals=3):
	"""
	Calculates the error rate of a weight vector on a set of feature vectors
	:param w: weight vectors
	:param data: feature vector matrix
	:param num_decimals: number of decimals used to represent the returned error rate
	:return: error rate
	"""
	mistakes = 0
	for row in data:
		x_i = row[1:]
		y_i = row[0]
		y_hat = calc_y_hat(x_i, w)

		if y_hat != y_i or y_hat == 0:
			mistakes += 1
	return np.round(float(mistakes) / data.shape[0], num_decimals)


def hmm():
	for i in range(2, 10):
		fn_suffix = "A2." + str(i) + "."

		train_fn = fn_suffix + "train.tsv"
		test_fn = fn_suffix + "test.tsv"
		out_fn = fn_suffix + "results.csv"

		print("---" + train_fn + "---")

		f = open(out_fn, 'w')
		out = csv.writer(f)

		test_data = get_data(train_fn)
		train_data = get_data(test_fn)

		w = np.zeros(test_data.shape[1] - 1)

		max_num_epochs = 4000
		epoch_str_len = len("epoch_" + str(max_num_epochs))

		# perform epochs on train data and note train and test error rate
		for epoch_i in range(0, max_num_epochs):
			np.random.shuffle(test_data)
			mistakes = epoch(test_data, w)
			this_epoch_str = "epoch_" + str(epoch_i)
			num_spaces = epoch_str_len - len(this_epoch_str)
			print(this_epoch_str + (" " * num_spaces) + ": " + str(mistakes))

			train_err = 1 - calc_error_rate(w, test_data)
			test_err = 1 - calc_error_rate(w, train_data)
			out.writerow([epoch_i, train_err, test_err])

			# perfect fit on training, no need to keep going
			if mistakes == 0:
				break

		# report resulting weight vector
		print("w :", w, sep=" ")

		f.close()


def do_the_shits():
	w = np.zeros(2)
	data = np.array([1, 2, 3, 4, 5, 6]).reshape((2, 3))
	print(data, w)
	epoch(data, w)
	print(data, w)

hmm()
