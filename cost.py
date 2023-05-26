import numpy as np

class Cost: 
    def function(expected: np.ndarray, actual: np.ndarray) -> float:
        pass

    def derivative(expected: np.ndarray, actual: np.ndarray) -> float:
        pass


class MeanSquareCost(Cost):
    def function(expected: np.ndarray, actual: np.ndarray) -> float:
        assert len(actual) == len(expected)
        return np.sum(np.power(actual - expected, 2)) / len(actual)
        

    def derivative(expected: np.ndarray, actual: np.ndarray) -> float:
        assert len(actual) == len(expected)
        return expected - actual
