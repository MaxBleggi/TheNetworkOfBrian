from mnist_loader import load_formatted
from vanilla_network import VanillaNetwork

MNIST_PATH = 'data/mnist.pkl.gz'

def main():
    training_load, validation_load, test_load = load_formatted(MNIST_PATH)

    net = VanillaNetwork([784, 30, 10])
    net.stoch_gradient_descent(training_load, 30, 10, 3.0, test_data=test_load)



if __name__ == '__main__':
    main()