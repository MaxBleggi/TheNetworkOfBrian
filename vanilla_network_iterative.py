"""
vanilla_network.py

Identical to vanilla network in all ways except this uses an iterative implementation of backpropagation.
This method is slower than the fully matrix-based approach, but it is interesting to performances.
"""

import time as stopwatch
import random as rand
import numpy as mmath
from vanilla_network import VanillaNetwork
from helpers import sigmoid, sigmoid_prime, quadratic_cost


class VanillaNetworkIterative(VanillaNetwork):
    def __init__(self, layers):
        VanillaNetwork.__init__(self, layers)

    def stoch_gradient_descent(self, training_data, epochs, batch_size, eta, test_data=None):
        """
        Trains the neural network via batch stochastic gradient descent.
        :param training_data: list of 2-tuples (x,y) where
                    x is a is a 784-dimensional vector representing the input image (28x28=784)
                    y is a 10-dimensional unit vector representing the correct digit
        :param epochs: the number of training iterations the network uses to learn
        :param batch_size: size of the subset of training_data
        :param eta: the learning rate hyperparameter of the neural network
        :param test_data: list of 2-tuples (x,y') where
                     x is defined above
                     y' is the digit value of x
        :return analytic_data: dictionary of the useful data if test data is inputed
        """
        # use data set for progress tracking
        training_data_size = len(training_data)
        test_data_size = len(test_data)
        analytic_data = {
            "epoch_delta_times": [],
            "back_prop_delta_times": [],
            "feed_forward_delta_times": [],
            "test_results": [],
            "test_sizes": test_data_size,
            "biases": [],
            "weights": []
        }

        epoch_start_time = stopwatch.time()
        for epoch in range(0, epochs):
            # randomly shuffle data to vary each batch sampling
            rand.shuffle(training_data)

            batches = []
            # create a list of subsets of training data for each batch
            for j in range(0, training_data_size, batch_size):
                batches.append(training_data[j: j + batch_size])

            for batch in batches:
                backprop_time_start = stopwatch.time()
                self.update_batch(batch, eta)
                analytic_data["back_prop_delta_times"].append(stopwatch.time() - backprop_time_start)

            if test_data:
                # surround in condition because eval is costly, so only need it during testing
                feedf_time_delta = stopwatch.time()
                result = self.evaluate_accuracy(test_data)

                analytic_data["test_results"].append(result)
                analytic_data["feed_forward_delta_times"].append(stopwatch.time() - feedf_time_delta)

                analytic_data["epoch_delta_times"].append(stopwatch.time() - epoch_start_time)
                print("epoch %i: %s / %s" % (epoch, result, test_data_size))
            else:
                print("epoch %i finished" % epoch)

        # get the final weights & biases if collecting data
        if test_data:
            # 'listify' the vectors so they can be serialized to json
            analytic_data["weights"] = [v.tolist() for v in self.weights]
            analytic_data["biases"] = [v.tolist() for v in self.biases]

        # only return results if there were any testing done
        # return empty when no test results are desired
        return analytic_data if test_data else {}

    def update_batch(self, batch, eta):
        """
        Update's the weight and bias matrices using backpropagation to each batch
        :param batch: list of 2-tuples (x,y)
        :param eta: the learning rate hyperparameter
        """
        batch_size = len(batch)

        nabla_w = [mmath.zeros(w.shape) for w in self.weights]
        nabla_b = [mmath.zeros(b.shape) for b in self.biases]

        for x, y in batch:
            delta_nabla_b, delta_nabla_w = self.back_propagation(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        assert self.weights_size == len(nabla_w), "ERROR: |weights| != |nabla weights|"

        for i in range(0, self.weights_size):
            self.weights[i] = self.weights[i] - (eta / batch_size) * nabla_w[i]
        for i in range(0, self.biases_size):
            self.biases[i] = self.biases[i] - (eta / batch_size) * nabla_b[i]

    def back_propagation(self, x, y):
        """
        Calculates the gradient of the cost function Cx for some batch of sample inputs
        :param x: a is a 784-dimensional vector representing the input image (28x28=784)
        :param y: a 10-dimensional unit vector representing the correct digit
        :return nabla_b, nabla_w: the weights and biases representing the gradient nabla_Cx
        """
        nabla_w = [0 for i in self.weights]
        nabla_b = [0 for i in self.biases]

        # calculate z values and activations for the backwards pass
        activation = x
        activations = [x]
        z_vector = []
        for b, w in zip(self.biases, self.weights):
            z = mmath.dot(w, activation) + b
            z_vector.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backwards pass through the network
        delta = quadratic_cost(activations[-1], y) * sigmoid_prime(z_vector[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = mmath.dot(delta, activations[-2].transpose())

        # don't include input layer
        for layer in range(2, self.layer_count):
            z = z_vector[-layer]
            sp = sigmoid_prime(z)
            delta = mmath.dot(self.weights[-layer + 1].transpose(), delta) * sp
            nabla_b[-layer] = delta
            nabla_w[-layer] = mmath.dot(delta, activations[-layer - 1].transpose())

        return nabla_b, nabla_w

    def feed_forward(self, act):
        """
        "Feeds" the input forward given an input, act.
        :param act: input for the network
        :return z_vector, activations: the result output, list of activation values
        """
        for b, w in zip(self.biases, self.weights):
            act = sigmoid(mmath.dot(w, act) + b)
        return act

    def evaluate_accuracy(self, test_data):
        """
        Determines the network's guess by the highest valued neuron in the output layer.
        If the guess was correct, it's summed into the total correct
        :param test_data: a list of 2-tuples (x,y)
        :return total_correct: total number of correct outputs
        """
        total_correct = 0
        for x, y in test_data:
            # the highest output value is the networks guess
            output = mmath.argmax(self.feed_forward_evaluation(x))
            if output == y: total_correct += 1

        return total_correct
