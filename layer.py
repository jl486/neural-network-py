import numpy as np
from abc import abstractmethod


class Layer:
    @abstractmethod
    def __init__(self, input_size: int, output_size: int) -> None:
        self.input: np.ndarray | None = None
        self.output: np.ndarray | None = None

    @abstractmethod
    def calculate_output(self, input: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def back_propagate(self, cost_gradient_wrt_output: np.ndarray, learning_rate: float) -> np.ndarray:
        pass
