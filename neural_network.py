import numpy as np
from layer import Layer


class NeuralNetwork:
    def __init__(self, layers: list[Layer], learning_rate: float) -> None:
        self.layers: list[Layer] = layers
        self.learning_rate = learning_rate

    # Forward propagate through the network to calculate the output
    def forward_propagate(self, input: np.ndarray) -> np.ndarray:
        output: np.ndarray = input
        for layer in self.layers:
            output = layer.calculate_output(output)
        return output

    # Train the neural network with backpropagation and updating weights
    def train(self, x_train: np.ndarray, y_train: np.ndarray, cost_function, d_cost_function, epochs: int, print_cost: bool = True) -> None:
        for e in range(epochs):
            cost: float = 0
            for x, y in zip(x_train, y_train):
                output: np.ndarray = self.forward_propagate(x)
                cost += cost_function(y, output)

                gradient = d_cost_function(y, output)
                for layer in reversed(self.layers):
                    gradient = layer.back_propagate(gradient, self.learning_rate)

            cost /= len(x_train)

            if print_cost:
                print(f"{e+1} of {epochs}, cost={cost}")
