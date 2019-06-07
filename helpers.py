
import numpy as mmath
import json


def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + mmath.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


def quadratic_cost(output_activations, y):
    """
    Quadratic variation returning the vector of partial derivatives for the output
    activations.
    :param output_activations: output value of neurons
    :param y: the real result to compare against ``output_activations``
    :return: Vector of paritial derivatives C_x
    """
    return output_activations - y


def simulate_network(activation, biases, weights):
    """
    "Simulates" a network by taking the already trained biases and weights to calcualte output
    by feeding it forward with sig(W * A + B)
    :param activation: input for the network
    :param biases: biases for a network
    :param weights: weights for a network
    :return: output from network
    """
    for b, w in zip(biases, weights):
        activation = sigmoid(mmath.dot(w, activation) + b)
    return activation

