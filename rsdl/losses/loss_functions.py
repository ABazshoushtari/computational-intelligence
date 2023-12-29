import numpy as np

from rsdl import Tensor
from rsdl.activations import Softmax
from rsdl.tensors import _tensor_neg


def MeanSquaredError(preds: Tensor, actual: Tensor):
    # TODO : implement mean squared error
    num_samples = preds.shape[0]
    squared_error = Tensor.__pow__(actual.__sub__(preds), 2.0)
    sum_squared_error = squared_error.sum()
    return sum_squared_error * Tensor(1 / num_samples, requires_grad=actual.requires_grad, depends_on=actual.depends_on)

def CategoricalCrossEntropy(preds: Tensor, actual: Tensor):
    # TODO : implement categorical cross entropy
    softmax = Softmax(preds)
    log_softmax = softmax.log()
    mul = actual * log_softmax
    ce = _tensor_neg(mul.sum())
    return ce
