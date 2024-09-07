import numpy as np

from layer import Layer


class Dense(Layer):
    def __init__(self, input_size: int, output_size: int) -> None:
        self.weights: np.ndarray = np.random.rand(output_size, input_size)
        self.bias: np.ndarray = np.random.rand(output_size, 1)

    # The output is calculated by Z = W * X + B
    def calculate_output(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    # Backpropagation uses the cost gradient wrt the output to update the
    # weights and biases to learn
    def back_propagate(self, cost_gradient_wrt_output: np.ndarray, learning_rate: float) -> np.ndarray:
        cost_gradient_wrt_weight: np.ndarray = np.dot(cost_gradient_wrt_output, np.array(self.input).T)
        self.weights -= learning_rate * cost_gradient_wrt_weight 
        self.bias -= learning_rate * cost_gradient_wrt_weight

        # Return the the partial derivative of the cost wrt the input
        return np.dot(self.weights.T, cost_gradient_wrt_output)

