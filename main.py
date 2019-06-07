from mnist_loader import load_formatted
from vanilla_network import VanillaNetwork
from vanilla_network_iterative import VanillaNetworkIterative
import json

MNIST_PATH = 'data/mnist.pkl.gz'


def main():
    training_load, validation_load, test_load = load_formatted(MNIST_PATH)

    # [ |input_layer| , ... |hidden layer| ..., |output_layer| ]
    nnet_input = [784, 30, 10]

    # hyper-parameters
    epochs = 30
    batch_size = 10
    learning_rate = 3.0

    # note: ``stoch_gradient_descent()`` will return {} if not test_data is provided

    vanilla_net = VanillaNetwork(nnet_input)
    data_set_vn = vanilla_net.stoch_gradient_descent(
        training_load,
        epochs,
        batch_size,
        learning_rate,
        test_data=test_load
    )

    vanilla_net_iterative = VanillaNetworkIterative(nnet_input)
    data_set_vni = vanilla_net_iterative.stoch_gradient_descent(
        training_load,
        epochs,
        batch_size,
        learning_rate,
        test_data=test_load
    )

    with open('docs/nnet_output', 'w') as file:
        json.dump(data_set_vn, file)


if __name__ == '__main__':
    main()