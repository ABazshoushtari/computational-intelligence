import numpy as np

from rsdl.optim import Optimizer

# TODO: implement Momentum optimizer like SGD
class Momentum(Optimizer):
    def __init__(self, layers, learning_rate=0.1, momentum=0.9):
        super().__init__(layers)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = []

        for layer in self.layers:
            weights, bias = layer.parameters()
            weight_velocity = np.zeros_like(weights)
            bias_velocity = np.zeros_like(bias)
            self.velocities.append((weight_velocity, bias_velocity))

    def step(self):
        for i, (layer, (weight_velocity, bias_velocity)) in enumerate(zip(self.layers, self.velocities)):
            weights, bias = layer.parameters()

            weight_velocity = self.momentum * weight_velocity - self.learning_rate * weights.grad
            layer.weight = layer.weight + weight_velocity

            if layer.need_bias:
                bias_velocity = self.momentum * bias_velocity - self.learning_rate * bias.grad
                layer.bias = layer.bias + bias_velocity

            self.velocities[i] = (weight_velocity, bias_velocity)

