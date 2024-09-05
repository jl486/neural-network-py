import numpy as np
from neuron import Neuron


class Layer:
    def __init__(self, num_neurons: np.int64, bias: np.float64) -> None:
        self.bias: np.float64 = bias if bias else np.float64(np.random.rand())
        self.neurons: np.ndarray = np.empty(num_neurons)
        self.neurons.fill(Neuron(self.bias))

    def __str__(self) -> str:
        out: str = "".join(str(weight) for neuron in self.neurons for weight in neuron.weights)
        out += " ".join(str(neuron.bias) for neuron in self.neurons)
        return out

    def calculate_outputs(self, inputs: np.ndarray) -> np.ndarray:
        return np.array([neuron.calculate_output(inputs) for neuron in self.neurons])

    def get_outputs(self) -> np.ndarray:
        return np.array([neuron.output for neuron in self.neurons])
