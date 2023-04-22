import pickle
from matplotlib import pyplot as plt
import numpy as np
from data_parser import *


def activation_function(input):
    return 1 / (1 + np.exp(-input))


def activation_derivative(input):
    s = activation_function(input)
    return s * (1 - s)


def cost_function(expected, actual):
    return np.sum(np.power(actual - expected, 2)) / len(actual)


def cost_derivative(expected, actual):
    return actual - expected


class Network:
    def __init__(self, layer_counts=None, file=None):
        if file is not None:
            self.load(file)
        else:
            layers = range(1, len(layer_counts))
            self.weights = [np.random.rand(layer_counts[i], layer_counts[i - 1]) - 0.5 for i in layers]
            self.biases = [np.random.rand(layer_counts[i]) - 0.5 for i in layers]        

    def save(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self.weights, f)
            pickle.dump(self.biases, f)

    def load(self, file):
        with open(file, 'rb') as f:
            self.weights = pickle.load(f)
            self.biases = pickle.load(f)

    def eval(self, input):
        value = input
        for w, b in zip(self.weights, self.biases):
            value = activation_function(w @ value + b)
        return value
    
    def accuracy(self, batch):
        positive = 0
        for x, y in batch:
            a = self.eval(x.flatten() / 255)
            if np.argmax(a) == y:
                positive += 1
        return positive / len(batch)

    # Detailed description can be found http://neuralnetworksanddeeplearning.com/chap2.html
    # Last access: 19.04.2023
    def backpropagation(self, input, expected):
        weight_gradient = [np.zeros(w.shape) for w in self.weights]
        bias_gradient = [np.zeros(b.shape) for b in self.biases]

        # Feed forward
        activations = [input]
        zs = []
        for weight, bias in zip(self.weights, self.biases):
            z = weight @ activations[-1] + bias
            zs.append(z)
            activations.append(activation_function(z))

        # backward pass
        delta = cost_derivative(activations[-1], expected) * activation_derivative(zs[-1])
        bias_gradient[-1] = delta
        weight_gradient[-1] = np.multiply.outer(delta, activations[-2])
        
        for i in range(2, len(self.biases) + 1):
            delta = (self.weights[-i + 1].T.dot(delta)) * activation_derivative(zs[-i]) 
            bias_gradient[-i] = delta
            weight_gradient[-i] = np.multiply.outer(delta, activations[-i - 1])

        return (bias_gradient, weight_gradient)

    def train(self, batch, learning_rate):
        # Sum up gradients for batch
        bias_gradient = [np.zeros(b.shape) for b in self.biases]
        weight_gradient = [np.zeros(w.shape) for w in self.weights]
        for input, expected in batch:
            # Prepare data
            input = input.flatten() / 255
            expected_vec = np.zeros(10)
            expected_vec[expected] = 1

            d_bias_gradient, d_weight_gradient = self.backpropagation(input, expected_vec)
            for i in range(len(d_bias_gradient)):
                bias_gradient[i] += d_bias_gradient[i]
                weight_gradient[i] += d_weight_gradient[i]

        # Apply gradients to weights and biases
        fac = learning_rate / len(batch)
        for i in range(len(weight_gradient)):
            self.weights[i] += fac * weight_gradient[i]
            self.biases[i] += fac * bias_gradient[i]


def train_digit_identifier(layers, learning_rate, batch_size, training_size):
    network = Network(layer_counts=layers)
    num_batches = training_size // batch_size + 1

    training_data = parse_training_data()
    test_data = parse_test_data()[:100]
    np.random.shuffle(training_data)
    
    for i in range(num_batches):
        # Pull a batch of training data
        batch = [x for _, x in zip(range(batch_size), training_data_generator(training_data))]
        # Train network
        network.train(batch, learning_rate)
        accuracy = network.accuracy(test_data)
        print(f'\rTraining network: {(i + 1) / num_batches * 100:.3f}%, Estimated accuracy: {accuracy * 100:.3f}%', end='')
    print()

    return network


def parameter_study():
    # Parameters
    test_data = parse_test_data()
    training_size = 16
    batch_sizes = [8, 16, 32]
    learning_rates = [6.0, 6.3, 6.6, 6.9, 7.2, 7.5, 7.8, 8.1]
    samples = 2
    architecture = [784, 16, 16, 10]

    # Build networks and evaluate accuracy
    accuracies = []
    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            accuracy = 0
            for i in range(samples):
                network = train_digit_identifier(architecture, learning_rate, batch_size, training_size)
                # Calculate final accuracy
                accuracy += network.accuracy(test_data)
            accuracies.append(accuracy / samples)

    # Plot and print out results
    for bs in range(len(batch_sizes)):
        for lr in range(len(learning_rates)):
            accuracy = accuracies[bs * len(learning_rates) + lr]
            print(f'Batch size: {batch_sizes[bs]}; Learning rate: {learning_rates[lr]}; Accuracy: {accuracy * 100:.3f}%')
    X, Y = np.meshgrid(learning_rates, batch_sizes)
    Z = np.array(accuracies).reshape(len(batch_sizes), len(learning_rates))
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    plt.show()
