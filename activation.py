import numpy as np

class Activation: 
    def function(input: np.ndarray) -> np.ndarray:
        pass

    def derivative(input: np.ndarray) -> np.ndarray:
        pass


class SigmoidActivation(Activation):
    def function(input: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-input))
        
    def derivative(input: np.ndarray) -> np.ndarray:
        s = SigmoidActivation.function(input)
        return s * (1 - s)


class SoftmaxActivation(Activation):
    def function(input: np.ndarray) -> np.ndarray:
        return np.exp(input) / np.sum(np.exp(input))
        
    def derivative(input: np.ndarray) -> np.ndarray:
        expSum = np.sum(np.exp(input))
        exp = np.exp(input)
        return (exp * expSum - exp ** 2) / (expSum ** 2)
    

class ReLUActivation(Activation):
    def function(input: np.ndarray) -> np.ndarray:
        return np.maximum(0, input)
        
    def derivative(input: np.ndarray) -> np.ndarray:
        result = np.zeros(shape=len(input))
        for i in range(len(input)):
            if input[i] > 0:
                result[i] = 1
        return result
    