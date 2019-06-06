"""

mnist_loader

Reads and loads the MNIST image data bank into a format easily readable by a neural network.
To retrieve formatted data call  ``load_formatted`` and to retrieve a more raw version,
call ``load_raw_data``
"""

import pickle
import gzip

import numpy as mmath

IMAGE_VECTOR_SIZE = 784

def _load_raw(mnist_path):
    """
    Reads in the images and results from mnist data set.
    tr_data is a 2-tuple containing the lists x, y in (x,y).
        x is a 50,000 item list containing a 28x28 image represented by a 784 length list
        y is a 50,000 item list containing the digit values corresponding to the images in x

    test_data and val_data are functionally identical to tr_data except they are 10,000 long

    :param mnist_path: relative path of mnist data set
    :return: tr_data, test_data, val_data
    """
    file = gzip.open(mnist_path, 'rb')
    tr_data, val_data, test_data = pickle.load(file, encoding="latin1")
    file.close()
    return tr_data, test_data, val_data

def load_formatted(mnist_path):
    """
    Formats the data retrieved from mnist data file into a workable format.
    training_load is 50,000 item list of 2-tuples (x,y)
        x is a is a 784-dimensional vector representing the input image (28x28=784)
        y is a 10-dimensional unit vector representing the correct digit
        Let x = [a_1 ... a_mn]^T where a_i is greyscale value of pixel a_ij in an (m x n) image
        Let y = [a_1 ... a_m]^T where {a_i in v | a_i = 1 if i == digit of input image (y), 0 otherwise}

    test_load and validation_load are 10,000 item lists of 2-tuples (x,y')
        x is defined above
        y' is the digit value represented by x

    :param mnist_path: relative path of mnist data set
    :return: training_load, test_load, validation_load
    """
    tr_data, test_data, val_data = _load_raw(mnist_path)

    # create the training load, a list of 50,000 2-tuples (x,y)
    # for each (x,y) in training data, x is 784-vector and y is a 10-vector
    training_load = []
    # TODO inquire about efficiency of zip() in loop, may be a slow approach
    for x_raw, y_raw in zip(tr_data[0], tr_data[1]):
        # format the image matrix 781-dimensional vector x represented in doc string
        x = mmath.reshape(x_raw, (IMAGE_VECTOR_SIZE, 1))

        # vectorize the result of each image into a vector y represented in doc string
        y = mmath.zeros((10, 1))
        y[y_raw] = 1.0

        training_load.append((x, y))

    # create validation and testing loads, each is a list of 10,000 2-tuples (x, y')
    # image matrices must be reshaped into a vector
    validation_x = [mmath.reshape(x, (IMAGE_VECTOR_SIZE, 1)) for x in val_data[0]]
    test_x = [mmath.reshape(x, (IMAGE_VECTOR_SIZE, 1)) for x in test_data[0]]

    # combine each list of x and y into a single list (x,y)
    validation_load = list(zip(validation_x, val_data[1]))
    test_load = list(zip(test_x, test_data[1]))
    print(len(training_load))
    return training_load, test_load, validation_load
