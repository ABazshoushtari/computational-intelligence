import numpy as np
from rsdl.optim import Optimizer


# TODO: implement RMSprop optimizer like SGD
class RMSprop(Optimizer):
    def __init__(self, layers, learning_rate=0.001, beta=0.9, epsilon=1e-8):
        super().__init__(layers)
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon

    def step(self):
        for layer in self.layers:
            self._update_weights(layer)
            if layer.need_bias:
                self._update_biases(layer)

    def _update_weights(self, layer):
        parameters = layer.parameters()

        if not hasattr(layer, 'squared_grad_avg'):
            layer.squared_grad_avg = 0.0

        layer.squared_grad_avg = self.beta * layer.squared_grad_avg + (1 - self.beta) * (parameters[0].grad ** 2)

        layer.weight = layer.weight - self.learning_rate * parameters[0].grad.data / (
                np.sqrt(layer.squared_grad_avg.data) + self.epsilon)

    def _update_biases(self, layer):
        parameters = layer.parameters()

        if not hasattr(layer, 'squared_bias_grad_avg'):
            layer.squared_bias_grad_avg = 0.0

        layer.squared_bias_grad_avg = self.beta * layer.squared_bias_grad_avg + (1 - self.beta) * (
                parameters[1].grad ** 2)

        layer.bias = layer.bias - self.learning_rate * parameters[1].grad.data / (
                np.sqrt(layer.squared_bias_grad_avg.data) + self.epsilon)
