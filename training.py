
import numpy as np
from activation import *
from cost import *
from mnist import *

from layer import Layer


def backpropagation(layers: list[Layer], trainingData, testData, learningRate, batchSize, epochs, costFunction):
    numTrainingData = len(trainingData)
    numBatches = numTrainingData // batchSize
    batches = [trainingData[x*batchSize:x*batchSize+batchSize] for x in range(numBatches)]

    for epoch in range(1, epochs + 1):
        counter = 1
        for batch in batches:
            averageBatchError = 0
            # Perform backpropagation
            for sample in batch:
                input = sample.imageVec
                expected = sample.labelVec

                activations = [input]
                for layer in layers:
                    activations += [layer.forward(activations[-1])]

                loss = costFunction.derivative(expected, activations[-1])
                delta = layers[-1].calculateGradientOutputLayer(loss, activations[-2])

                idx = len(layers) - 2
                while idx >= 0:
                    nextLayer = layers[idx + 1]
                    prevLayerActivation = activations[idx]
                    delta = layers[idx].calculateGradientHiddenLayer(nextLayer.weights, delta, prevLayerActivation)
                    idx -= 1

                averageBatchError += costFunction.function(expected, activations[-1])
            
            # Apply gradient and print statistics
            for layer in layers:
                layer.applyGradient(learningRate, batchSize)
            averageBatchError /= len(batch)
            print(f'\rEpoch: {epoch}: {counter / numBatches * 100:.3f}%, average cost: {averageBatchError:.6f}', end='')
            counter += 1

    # Evaluate network accuracy
    correct = 0
    for sample in testData:
        x = sample.imageVec
        for layer in layers:
            x = layer.forward(x)
        if np.argmax(x) == sample.label:
            correct += 1
    print(f'\nNetwork accuracy: {correct / len(testData) * 100:.3f}%')
