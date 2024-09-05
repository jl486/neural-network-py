import numpy as np


class Neuron:
    def __init__(self, bias: np.float64) -> None:
        self.bias: np.float64 = bias
        self.weights: np.ndarray = np.array([])

    def calculate_output(self, inputs: np.ndarray) -> np.float64:
        self.inputs: np.ndarray = inputs
        self.output: np.float64 = self.activation(self.calculate_net_input())
        return self.output

    def calculate_net_input(self) -> np.float64:
        net_input_arr: np.ndarray = self.inputs * self.weights
        net_input: np.float64 = np.sum(net_input_arr)
        return net_input + self.bias

    def pd_net_input_wrt_weight(self, index: int) -> np.float64:
        return self.inputs[index]

    def activation(self, net_input: np.float64) -> np.float64:
        return 1 / (1 + np.exp(-net_input))

    # derivative net input wrt input
    def d_activation(self) -> np.float64:
        return self.output * (1 - self.output)

    # Mean squared error
    def loss(self, target_output: np.float64) -> np.float64:
        return 0.5 * (target_output - self.output) ** 2

    # derivative of cost wrt output
    def d_loss(self, target_output) -> np.float64:
        return self.output - target_output
