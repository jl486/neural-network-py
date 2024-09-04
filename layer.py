import numpy as np
from neuron import Neuron


class Layer:
    def __init__(self, num_neurons: np.int64, bias: np.float64) -> None:
        self.bias = bias if bias else np.random.rand()
        self.neurons = np.empty(num_neurons)
