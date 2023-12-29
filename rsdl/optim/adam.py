import numpy as np

from rsdl import Tensor
from rsdl.optim import Optimizer

# TODO: implement Adam optimizer like SGD
class Adam(Optimizer):
    def __init__(self, layers, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(layers)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Initialize first and second moment estimates for each parameter
        for layer in self.layers:
            self._initialize_moments(layer)

        self.iteration = 0  # Counter for tracking the number of iterations

    def step(self):
        self.iteration += 1

        # Update weight and biases with Adam
        for layer in self.layers:
            self._update_moments(layer)
            self._update_parameters(layer)

    def _initialize_moments(self, layer):
        layer.moment1 = Tensor(np.zeros_like(layer.weight.data), requires_grad=False)
        layer.moment2 = Tensor(np.zeros_like(layer.weight.data), requires_grad=False)
        if layer.need_bias:
            layer.bias_moment1 = Tensor(np.zeros_like(layer.bias.data), requires_grad=False)
            layer.bias_moment2 = Tensor(np.zeros_like(layer.bias.data), requires_grad=False)

    def _update_moments(self, layer):
        layer.moment1.data = self.beta1 * layer.moment1.data + (1 - self.beta1) * layer.weight.grad.data
        layer.moment2.data = self.beta2 * layer.moment2.data + (1 - self.beta2) * layer.weight.grad.data ** 2

        if layer.need_bias:
            layer.bias_moment1.data = self.beta1 * layer.bias_moment1.data + (1 - self.beta1) * layer.bias.grad.data
            layer.bias_moment2.data = self.beta2 * layer.bias_moment2.data + (1 - self.beta2) * layer.bias.grad.data ** 2

    def _update_parameters(self, layer):
        moment1_hat = layer.moment1.data / (1 - self.beta1 ** self.iteration)
        moment2_hat = layer.moment2.data / (1 - self.beta2 ** self.iteration)
        layer.weight.data = layer.weight.data - self.learning_rate * moment1_hat / (np.sqrt(moment2_hat) + self.epsilon)

        if layer.need_bias:
            bias_moment1_hat = layer.bias_moment1.data / (1 - self.beta1 ** self.iteration)
            bias_moment2_hat = layer.bias_moment2.data / (1 - self.beta2 ** self.iteration)
            layer.bias.data = layer.bias.data - self.learning_rate * bias_moment1_hat / (
                    np.sqrt(bias_moment2_hat) + self.epsilon)
