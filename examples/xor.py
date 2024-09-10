import numpy as np

from neural_network import NeuralNetwork
from dense import Dense

x_train = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
y_train = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

x_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_test = np.array([[0], [1], [1], [0]])

def cost(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def d_cost(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)

network = NeuralNetwork([
    Dense(2, 3),
    Dense(3, 1)
], learning_rate=0.01)

network.train(x_train, y_train, cost, d_cost, epochs=1000)
