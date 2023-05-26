import numpy as np

from activation import *
from mnist import *


class Layer:
    def __init__(self, numInputNodes: int, numOutputNodes: int, activation: Activation):
        self.weights = np.random.rand(numOutputNodes, numInputNodes) - 0.5
        self.bias = np.random.rand(numOutputNodes) - 0.5
        self.activation = activation
        self.biasGradient = np.zeros(shape=(numOutputNodes))
        self.weightGradient = np.zeros(shape=(numOutputNodes, numInputNodes))

    def setWeightsAndBiase(self, weights: np.ndarray, bias: np.ndarray):
        assert self.weights.shape == weights.shape
        assert self.bias.shape == bias.shape
        self.weights = weights
        self.bias = bias

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

    def applyGradient(self, learningRate, batchSize):
        learningFactor = learningRate / batchSize
        self.bias += learningFactor * self.biasGradient
        self.weights += learningFactor * self.weightGradient
        self.biasGradient.fill(0)
        self.weightGradient.fill(0)
