from rsdl import Tensor, Dependency
import numpy as np

from rsdl.tensors import _tensor_neg


def Sigmoid(t: Tensor) -> Tensor:
    # TODO: implement sigmoid function
    # hint: you can do it using function you've implemented (not directly define grad func)
    neg_tensor = _tensor_neg(t)
    exp_tensor = neg_tensor.exp()
    return (1 + exp_tensor).__pow__(-1.0)
    # return exp_tensor.__add__(other=Tensor(np.ones_like(exp_tensor.data), requires_grad=True))


def Tanh(t: Tensor) -> Tensor:
    # TODO: implement tanh function
    # hint: you can do it using function you've implemented (not directly define grad func)
    numerator = t.exp() - _tensor_neg(t).exp()
    denominator = t.exp() + _tensor_neg(t).exp()
    return numerator * denominator.__pow__(-1.0)


def Softmax(t: Tensor) -> Tensor:
    # TODO: implement softmax function
    # hint: you can do it using function you've implemented (not directly define grad func)
    # hint: you can't use sum because it has not axis argument so there are 2 ways:
    #        1. implement sum by axis
    #        2. using matrix mul to do it :) (recommended)
    # hint: a/b = a*(b^-1)
    exp_tensor = t.exp()
    # sum_exp_tensor = Tensor(np.ones_like(exp_tensor)) @ exp_tensor
    sum_exp_tensor = exp_tensor.sum()
    return exp_tensor * sum_exp_tensor.__pow__(-1.0)


def Relu(t: Tensor) -> Tensor:
    # TODO: implement relu function

    # use np.maximum
    data = np.maximum(0, t.data)

    req_grad = t.requires_grad
    if req_grad:
        def grad_fn(grad: np.ndarray):
            # use np.where
            return np.where(data < 0, 0, grad)

        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []
    return Tensor(data=data, requires_grad=req_grad, depends_on=depends_on)


def LeakyRelu(t: Tensor, leak=0.05) -> Tensor:
    """
    fill 'data' and 'req_grad' and implement LeakyRelu grad_fn 
    hint: use np.where like Relu method but for LeakyRelu
    """
    # TODO: implement leaky_relu function

    # data = np.maximum(leak * t.data, t.data)
    data = np.where(t.data < 0, leak * t.data, t.data)

    req_grad = t.requires_grad
    if req_grad:
        def grad_fn(grad: np.ndarray):
            return np.where(data < 0, leak * grad, grad)

        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []

    return Tensor(data=data, requires_grad=req_grad, depends_on=depends_on)


print(Tanh(Tensor(np.array([[1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9]]),
                  requires_grad=True)))

print(Softmax(Tensor(np.array([[3.2],
                               [1.3],
                               [0.2],
                               [0.8]]),
                     requires_grad=True)))

print(Relu(Tensor(np.array([[-1, -2, 3],
                            [4, 5, -6],
                            [7, 8, 9]]),
                  requires_grad=True)))

print(LeakyRelu(Tensor(np.array([[-1, -2, 3],
                            [4, 5, -6],
                            [7, 8, 9]]),
                  requires_grad=True)))
