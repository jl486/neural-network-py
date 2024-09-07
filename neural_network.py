import numpy as np
from layer import Layer


class NeuralNetwork:
    def __init__(self, layers: list[Layer]) -> None:
        self.layers: list[Layer] = layers

    # Forward propagate through the network to calculate the output
    def forward_propagate(self, input: np.ndarray) -> np.ndarray:
        output: np.ndarray = input
        for layer in self.layers:
            output = layer.calculate_output(output)
        return output

    def train(self):
        pass
