import numpy as np
import pickle as pk
from activation import *
from mnist import *


class Layer:
    def __init__(self, numInputNodes: int, numOutputNodes: int, activation: Activation):
        self.weights = np.random.randn(numOutputNodes, numInputNodes) / np.sqrt(numInputNodes)
        self.bias = np.zeros(numOutputNodes)
        self.activation = activation
        self.biasGradient = np.zeros(shape=(numOutputNodes))
        self.weightGradient = np.zeros(shape=(numOutputNodes, numInputNodes))
        self.prevBiasGradient = np.zeros(shape=(numOutputNodes))
        self.prevWeightGradient = np.zeros(shape=(numOutputNodes, numInputNodes))

    def setFrom(self, weights: np.ndarray, bias: np.ndarray, activation: Activation):
        self.weights = weights
        self.bias = bias
        self.activation = activation
        self.biasGradient = np.zeros(shape=bias.shape)
        self.weightGradient = np.zeros(shape=weights.shape)
        self.prevBiasGradient = np.zeros(shape=bias.shape)
        self.prevWeightGradient = np.zeros(shape=weights.shape)

    def forward(self, input: np.ndarray):
        self.nodeValues = self.weights @ input + self.bias
        return self.activation.function(self.nodeValues)

    def calculateGradientOutputLayer(self, loss, prevLayerActivation):
        assert loss.shape == self.nodeValues.shape 
        delta = loss * self.activation.derivative(self.nodeValues)
        self.biasGradient += delta
        self.weightGradient += np.outer(delta, prevLayerActivation)
        return delta

    def calculateGradientHiddenLayer(self, nextLayerWeights, nextLayerDelta, prevLayerActivation):
        delta = nextLayerWeights.T @ nextLayerDelta
        delta *= self.activation.derivative(self.nodeValues)
        self.biasGradient += delta
        self.weightGradient += np.outer(delta, prevLayerActivation)
        return delta

    def applyGradient(self, learningRate, batchSize, momentum):
        self.biasGradient /= batchSize
        self.weightGradient /= batchSize
        self.bias += (1 - momentum) * learningRate * self.biasGradient + momentum * self.prevBiasGradient
        self.weights += (1 - momentum) * learningRate * self.weightGradient + momentum * self.prevWeightGradient
        self.prevBiasGradient = np.copy(self.biasGradient) 
        self.prevWeightGradient = np.copy(self.weightGradient) 
        self.biasGradient.fill(0)
        self.weightGradient.fill(0)


def saveNetwork(filePath: str, layers: list[Layer]):
    weights = []
    biases = []
    activations = []
    for layer in layers:
        weights += [layer.weights]
        biases += [layer.bias]
        activations += [layer.activation.getName()]
    with open(filePath, 'wb') as file:
        pk.dump(weights, file)
        pk.dump(biases, file)
        pk.dump(activations, file)

    
def fromFile(filePath: str) -> list[Layer]:
    with open(filePath, 'rb') as file:
        weights = pk.load(file)
        biases = pk.load(file)
        activations = pk.load(file)
    
    layers = [Layer(1, 1, SigmoidActivation) for _ in range(len(weights))]
    for l, w, b, a in zip(layers, weights, biases, activations):
        l.setFrom(w, b, fromName(a))
    return layers

