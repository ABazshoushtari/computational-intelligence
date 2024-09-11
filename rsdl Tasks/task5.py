# Task 5
import numpy as np
# from keras.optimizers import RMSprop

from rsdl import Tensor
from rsdl.layers import Linear
from rsdl.optim import SGD, Adam, Momentum, RMSprop
from rsdl.losses import loss_functions

X = Tensor(np.random.randn(100, 3))
coef = Tensor(np.array([-7, +3, -9]))
y = X @ coef + 5

# TODO: define a linear layer using Linear() class  
l = Linear(in_channels=3, out_channels=1, need_bias=True)

# TODO: define an optimizer using SGD() class 
optimizer = SGD([l], learning_rate=0.01)
# optimizer = Adam([l], learning_rate=0.1)
# optimizer = Momentum([l], learning_rate=0.01)
# optimizer = RMSprop([l], learning_rate=0.1)

# TODO: print weight and bias of linear layer
print(l.weight)
print(l.bias)

# learning_rate = 0.01
batch_size = 10

for epoch in range(90):

    epoch_loss = 0.0

    for start in range(0, 100, batch_size):
        end = start + batch_size

        print(epoch, start, end)

        inputs = X[start:end]

        l.zero_grad()
        # optimizer.zero_grad()
        # inputs.zero_grad()

        # TODO: predicted
        predicted = l.forward(inputs)

        actual = y[start:end]
        actual.data = actual.data.reshape(batch_size, 1)

        # TODO: calcualte MSE loss
        loss = loss_functions.MeanSquaredError(predicted, actual)

        # TODO: backward
        # hint you need to just do loss.backward()
        loss.backward()

        # TODO: add loss to epoch_loss
        epoch_loss += loss

        # TODO: update w and b using optimizer.step()
        optimizer.step()

# TODO: print weight and bias of linear layer
print(f"weight of linear layer after training: {l.weight}")
print(f"bias of linear layer after training: {l.bias}")
