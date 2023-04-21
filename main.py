import pickle
import numpy as np
import data_parser
import matplotlib.pyplot as plt


def activation_function(input):
    return 1 / (1 + np.exp(-input))


def activation_derivative(input):
    s = activation_function(input)
    return s * (1 - s)


def cost_function(expected, actual):
    return np.sum(np.power(actual - expected, 2)) / (2 * len(actual))


def cost_derivative(expected, actual):
    return actual - expected


class Network:
    def __init__(self, layer_counts):
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

    # Detailed description can be found http://neuralnetworksanddeeplearning.com/chap2.html
    # Last access (19.04.2023)
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
        for input, expected in zip(batch[0], batch[1]):
            # Prepare data
            input = input.flatten() / 255
            expected_vec = np.zeros(10)
            expected_vec[expected] = 1

            d_bias_gradient, d_weight_gradient = self.backpropagation(input, expected_vec)
            for i in range(len(d_bias_gradient)):
                bias_gradient[i] += d_bias_gradient[i]
                weight_gradient[i] += d_weight_gradient[i]

        # Apply gradients to weights and biases
        fac = learning_rate / len(batch[0])
        for i in range(len(weight_gradient)):
            self.weights[i] += fac * weight_gradient[i]
            self.biases[i] += fac * bias_gradient[i]

# Network settings
LEARNING_RATE = 9.0
LEARNING_BATCH_SIZE = 100
network = Network([784, 16, 16, 10])
        
# Parse training and test data
train_images, train_labels = data_parser.parse_training_data()
test_images, test_labels = data_parser.parse_training_data()
image_batches = [train_images[x:x + LEARNING_BATCH_SIZE] for x in range(0, len(train_images), LEARNING_BATCH_SIZE)]
label_batches = [train_labels[x:x + LEARNING_BATCH_SIZE] for x in range(0, len(train_labels), LEARNING_BATCH_SIZE)]

# Train network
i = 1
for batch in zip(image_batches, label_batches):
    network.train(batch, LEARNING_RATE)
    print(f'\rTraining network: {i / len(image_batches) * 100:.3f}%', end='')
    i += 1
print()
network.save('network-sigmoid-784-32-32-32-10.pkl')

# Calculate accuracy
s = 0
for image, label in zip(test_images, test_labels):
    v = np.zeros(10)
    v[label] = 1
    a = network.eval(image.flatten() / 255)
    if np.argmax(a) == label:
        s += 1
s /= len(test_images)
print(f'Accuracy: {s * 100:.3f}%')

# Show results
for image, label in zip(test_images, test_labels): 
    result = np.argmax(network.eval(image.flatten() / 255))
    plt.imshow(image, cmap='gray')
    plt.title(f'Expected: {label}, Network output: {result}')
    plt.show()