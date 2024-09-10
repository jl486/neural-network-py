import numpy as np

from layer import Layer


class Dense(Layer):
    def __init__(self, input_size: int, output_size: int) -> None:
        self.weights: np.ndarray = np.random.randn(output_size, input_size)
        self.bias: np.ndarray = np.random.randn(output_size, 1)

    # The output is calculated by Z = W * X + B
    def calculate_output(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        output: np.ndarray = np.dot(self.weights, self.input) + self.bias
        output = np.array([self.activation(value) for value in output])
        return output

    # Backpropagation uses the cost gradient wrt the output to update the
    # weights and biases to learn
    def back_propagate(self, cost_gradient_wrt_output: np.ndarray, learning_rate: float) -> np.ndarray:
        cost_gradient_wrt_weight: np.ndarray = np.dot(cost_gradient_wrt_output, np.array(self.input).T)
        self.weights -= learning_rate * cost_gradient_wrt_weight
        self.bias -= learning_rate * cost_gradient_wrt_output

        output = np.dot(self.weights.T, cost_gradient_wrt_output)
        output = np.array([self.activation(value) for value in output])

        return output

    # TODO: Allow choices for different activation functions
    def activation(self, x: float) -> float:
        return 1 / (1 + np.exp(-x))

    def d_activation(self, x: float) -> float:
        return self.activation(x) * (1 - self.activation(x))
