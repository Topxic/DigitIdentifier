import numpy as np


class Activation: 
    def function(signal: np.ndarray) -> np.ndarray:
        pass

    def derivative(signal: np.ndarray) -> np.ndarray:
        pass

    def getName() -> str:
        pass


class SigmoidActivation(Activation):
    def function(signal: np.ndarray) -> np.ndarray:
        signal = np.clip(signal, -500, 500)
        return 1 / (1 + np.exp(-signal))
        
    def derivative(signal: np.ndarray) -> np.ndarray:
        signal = np.clip(signal, -500, 500)
        s = SigmoidActivation.function(signal)
        return s * (1 - s)
    
    def getName() -> str:
        return 'sigmoid'


class SoftmaxActivation(Activation):
    def function(signal: np.ndarray) -> np.ndarray:
        signal = np.clip(signal, -500, 500)
        exp = np.exp(signal)
        return exp / np.sum(exp)
        
    def derivative(signal: np.ndarray) -> np.ndarray:
        signal = np.clip(signal, -500, 500)
        exp = np.exp(signal)
        expSum = np.sum(exp)
        denom = expSum ** 2
        if np.abs(denom) < 1.e-16:
            denom = np.sign(denom) * 1.e-16
        return (exp * expSum - exp ** 2) / denom
    
    def getName() -> str:
        return 'softmax'
    

class ReLUActivation(Activation):
    def function(signal: np.ndarray) -> np.ndarray:
        signal = np.clip(signal, -500, 500)
        return np.maximum(0, signal)
        
    def derivative(signal: np.ndarray) -> np.ndarray:
        signal = np.clip(signal, -500, 500)
        result = np.zeros(shape=len(signal))
        for i in range(len(signal)):
            if signal[i] > 0:
                result[i] = 1
        return result
    
    def getName() -> str:
        return 'relu'
    

def fromName(name: str) -> Activation:
    if name == 'sigmoid':
        return SigmoidActivation
    if name == 'softmax':
        return SoftmaxActivation
    if name == 'relu':
        return ReLUActivation
    raise ValueError("Activation function {name} doesn't exist")
