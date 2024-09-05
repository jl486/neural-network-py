import numpy as np
from layer import Layer


class NeuralNetwork:
    def __init__(
        self,
        num_inputs: int,
        num_hidden_layers: int,
        num_neurons_per_hidden: np.int64,
        hidden_layer_biases: np.ndarray,
        num_output_neurons: np.int64,
        output_layer_bias: np.float64,
    ) -> None:
        self.num_inputs = num_inputs
        self.num_layers = num_hidden_layers + 2

        self.hidden_layer_biases: np.ndarray = hidden_layer_biases
        self.hidden_layers: np.ndarray = np.array(
            [Layer(num_neurons_per_hidden, bias) for bias in hidden_layer_biases]
        )
        self.output_layer = Layer(num_output_neurons, output_layer_bias)

    def feed_forward(self, inputs: np.ndarray) -> np.ndarray:
        next_inputs: np.ndarray = inputs
        for i in range(len(self.hidden_layers)):
            next_inputs = self.hidden_layers[i].calculate_outputs(next_inputs)

        # Return output layer after calculating outputs for hidden layers
        return self.hidden_layers[self.num_layers - 1].calculate_outputs(next_inputs)
