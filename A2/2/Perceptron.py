import numpy as np
import csv
import threading


def threaded(fn):
    """
    Calls provided function in thread
    :param fn: function to thread
    :return: returns a wrapper for this threaded function call
    """
    def wrapper(*args, **kwargs):
        threading.Thread(target=fn, args=args, kwargs=kwargs).start()
    return wrapper


def get_data(filename):
    """
    Fetches tsv data from specified file and returns numpy matrix
    :param filename: name of file to extract from
    :return: numpy array of tsv data
    """
    data = np.loadtxt(filename, delimiter="\t")
    return data


def epoch(data, w=None, b=0):
    """
    Perform one run over the data and tune our weight vector and bias scalar
    :param data: matrix of data, each row being an entry with y_i in data[i][0] and x_i in data[i][1:]
    :param w: weight vector; if None, is initialized as a zero vector
    :param b: bias scalar
    :return: updated b and number of mistakes
    """
    if w is None:
        w = np.zeros((data.shape[1] - 1))

    mistakes = 0

    for i in range(0, data.shape[0]):
        x_i = data[i][1:]
        y_i = data[i][0]
        y_hat = calc_y_hat(x_i, w, b)

        if y_hat != y_i:
            # entry is incorrectly classified, so let's adjust our weight vector and bias
            w += y_i * x_i
            b += y_i
            mistakes += 1

    return b, mistakes


def calc_y_hat(x, w, b):
    """
    Calculates the value of y_hat using feature and weight vectors and a bias
    :param x: feature vector
    :param w: weight vector
    :param b: bias scalar
    :return: y_hat; -1 if x * w < 0, +1 if x * w = 0, -1 otherwise
    """
    val = w.dot(x) + b
    if val < 0:
        return -1
    elif val == 0:
        return 0
    else:
        return 1


def calc_error_rate(w, b, data, num_decimals=3):
    """
    Calculates the error rate of a weight and bias vector on a set of feature vectors
    :param w: weight vector
    :param b: bias 
    :param data: feature vector matrix
    :param num_decimals: number of decimals used to represent the returned error rate
    :return: error rate
    """
    mistakes = 0
    for row in data:
        x_i = row[1:]
        y_i = row[0]
        y_hat = calc_y_hat(x_i, w, b)

        if y_hat != y_i:
            mistakes += 1
    return np.round(float(mistakes) / data.shape[0], num_decimals)


def find_min_margin(w, b, data, num_decimals=3):
    """
    Finds the minimum margin between all points in data and it's linear separator
    :param w: weight vector
    :param b: bias
    :param data: feature vector matrix
    :param num_decimals: number of decimals used to represent the returned error rate
    :return: minimum margin
    """
    min_margin = data[0][0] * (w.dot(data[0][1:]) + b)
    for i in range(1, data.shape[0]):
        this_margin = data[i][0] * (w.dot(data[i][1:]) + b)
        if this_margin < min_margin:
            min_margin = this_margin
    return np.round(min_margin, num_decimals)


def tune(w, b, data, num_epochs=5000, dev_ratio=0.2):
    """
    Tune hyperparameter E (number of epochs)
    :param w: weight vector
    :param b: bias
    :param data: feature vector matrix
    :param dev_ratio: percent of test data to be used for development
    :return: E
    """
    np.random.shuffle(data)
    split_idx = np.round(dev_ratio * data.shape[0])
    dev_data, train_data = data[:split_idx], data[split_idx:]
    for epoch_i in range(0, num_epochs):
        np.random.shuffle(test_data)
        b, mistakes = epoch(test_data, w, b)

        if uber_verbose:
            this_epoch_str = "epoch_" + str(epoch_i)
            num_spaces = epoch_str_len - len(this_epoch_str)
            print(this_epoch_str + (" " * num_spaces) + ": " + str(mistakes))

        train_err = calc_error_rate(w, b, test_data)
        test_err = calc_error_rate(w, b, train_data)
        out.writerow([epoch_i, train_err, test_err])

        if train_err == prev_train_err:
            num_epochs_without_change += 1
            # if our training error has remained constant for a while, let's stop
            if num_epochs_without_change == max_num_epochs_without_change:
                break
        else:
            num_epochs_without_change = 0
            prev_train_err = train_err


@threaded
def run_on_set(fn_suffix, num_epochs, uber_verbose=False):
    """
    Performs multiple epochs
    :param fn_suffix: filename suffix for data set
    :param num_epochs: number of epochs to perform
    :param uber_verbose: flag for outputting progression to stdout
    """
    train_fn = fn_suffix + "train.tsv"
    test_fn = fn_suffix + "test.tsv"
    out_fn = fn_suffix + "results.csv"

    if uber_verbose:
        print("---" + train_fn + "---")

    f = open(out_fn, 'w')
    out = csv.writer(f)
    out.writerow(["Epoch", "Train Error", "Test Error"])

    test_data = get_data(train_fn)
    train_data = get_data(test_fn)

    w = np.zeros(test_data.shape[1] - 1)
    b = 0

    epoch_str_len = len("epoch_" + str(num_epochs))

    num_epochs_without_change = 0
    max_num_epochs_without_change = 20

    prev_train_err = 1.0

    # perform epochs on train data and note train and test error rate
    for epoch_i in range(0, num_epochs):
        np.random.shuffle(test_data)
        b, mistakes = epoch(test_data, w, b)

        if uber_verbose:
            this_epoch_str = "epoch_" + str(epoch_i)
            num_spaces = epoch_str_len - len(this_epoch_str)
            print(this_epoch_str + (" " * num_spaces) + ": " + str(mistakes))

        train_err = calc_error_rate(w, b, test_data)
        test_err = calc_error_rate(w, b, train_data)
        out.writerow([epoch_i, train_err, test_err])

        if train_err == prev_train_err:
            num_epochs_without_change += 1
            # if our training error has remained constant for a while, let's stop
            if num_epochs_without_change == max_num_epochs_without_change:
                break
        else:
            num_epochs_without_change = 0
            prev_train_err = train_err

    min_margin = "-infinity"
    if train_err == 0:
        min_margin = find_min_margin(w, b, test_data)

    weight_ratios = np.round(w / sum(w), 3)

    print(fn_suffix[:-1] + " complete")
    print("  min margin : " + str(min_margin))
    print("  w : ", weight_ratios, sep=" ")
    print(" ")

    f.close()


def run_on_all_sets():
    num_epochs = 50
    for i in range(2, 10):
        fn_suffix = "A2." + str(i) + "."
        run_on_set(fn_suffix, num_epochs)


run_on_all_sets()
