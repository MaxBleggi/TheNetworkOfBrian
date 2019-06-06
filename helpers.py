
import numpy as mmath


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
