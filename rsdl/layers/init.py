import numpy as np
from rsdl import Tensor


# TODO: implement xavier_initializer, zero_initializer
def _calculate_fan_in_and_fan_out(tensor: Tensor):
    fan_in = np.size(tensor.data, axis=1)
    fan_out = np.size(tensor.data, axis=0)
    return fan_in, fan_out


def _calculate_correct_fan(tensor: Tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError(f"Mode {mode} not supported, please use one of {valid_modes}")

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out


# xavier uniform
def xavier_initializer(shape, gain: float = 1.0):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(Tensor(np.ones(shape)))
    std = gain * np.sqrt(2.0 / float(fan_in + fan_out))
    a = np.sqrt(3) * std
    # return np.random.randn(*shape) * np.sqrt(1/shape[0], dtype=np.float64)
    return np.random.uniform(-a, a, size=shape)


# kaiming uniform (he initialization uniform)
def he_initializer(shape, mode: str = "fan_in"):
    fan = _calculate_correct_fan(Tensor(np.ones(shape)), mode)
    gain = np.sqrt(2.0)  # non-linearity: relu
    std = gain / np.sqrt(fan)
    bound = np.sqrt(3.0) * std
    # return np.random.randn(*shape) * np.sqrt(2 / shape[0], dtype=np.float64)
    return np.random.uniform(-bound, bound, size=shape)

def zero_initializer(shape):
    return np.zeros(shape, dtype=np.float64)


def one_initializer(shape):
    return np.ones(shape, dtype=np.float64)


def initializer(shape, mode="xavier"):
    if mode == "xavier":
        return xavier_initializer(shape)
    elif mode == "he":
        return he_initializer(shape)
    elif mode == "zero":
        return zero_initializer(shape)
    elif mode == "one":
        return one_initializer(shape)
    else:
        raise NotImplementedError("Not implemented initializer method")
