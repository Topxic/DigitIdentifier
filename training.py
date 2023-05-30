
import numpy as np
from cost import Cost
from mnist import MNISTSample

from layer import Layer


def backpropagation(layers: list[Layer], trainingDataGenerator, testData: list[MNISTSample], 
                    learningRate: float, batchSize: int, epochs: int, costFunction: Cost, momentum: float):

    for epoch in range(1, epochs + 1):
        counter = 0
        for batch, progress in trainingDataGenerator(batchSize):
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
            
            # Apply gradient
            for layer in layers:
                layer.applyGradient(learningRate, batchSize, momentum)
            averageBatchError /= len(batch)
            counter += len(batch)
            print(f'\rEpoch: {epoch} - {progress * 100:.3f}% average loss: {averageBatchError:.6f}', end='')

    # Evaluate network accuracy
    correct = 0
    for sample in testData:
        x = sample.imageVec
        for layer in layers:
            x = layer.forward(x)
        if np.argmax(x) == sample.label:
            correct += 1
    print(f'\nNetwork accuracy: {correct / len(testData) * 100:.3f}%')

