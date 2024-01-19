from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        activation_f = nn.ReLU
        conv_layers = 1
        dense_layers = 3
        conv_sizes = [114]
        in_size = input_shape[0]
        kernel = 3
        stack = [
            nn.Conv2d(input_shape[0], conv_sizes[0], kernel_size=kernel),
            activation_f(),
            nn.Flatten(),
        ]
        in_size = conv_sizes[-1] * (8 - conv_layers * 2) ** 2
        dense_sizes = [57, 110, 62]
        for i in range(dense_layers):
            out_size = dense_sizes[i]
            l = nn.Linear(in_size, out_size)
            in_size = out_size
            stack.extend([l, activation_f()])
        stack.append(nn.Linear(in_size, 1))
        self.stack = nn.Sequential(*stack)

    def forward(self, x):
        logits = self.stack(x)
        return logits
