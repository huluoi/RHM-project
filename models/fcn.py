import torch
from torch import nn

class Linear1d(nn.Module):
    def __init__(
        self, input_channels, out_channels, bias=False
    ):
        super(Linear1d, self).__init__()
        self.weight = nn.Parameter( # input [bs, cin], weight [cout, cin]
            torch.randn(
                out_channels,
                input_channels,
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(1, out_channels))
        else:
            self.register_parameter("bias", None)

        self.input_channels = input_channels

    def forward(self, x):
        x = x[:, None] * self.weight # [bs, cout, cin]
        x = x.sum(dim=-1) # [bs, cout]
        x /= self.input_channels ** .5
        if self.bias is not None:
            x += self.bias * 0.1
        return x


class FCN(nn.Module):
    def __init__(self, input_channels, h, out_dim, num_layers, bias=False):
        super(FCN, self).__init__()

        self.feas = []  # to store features
        self.feas1_probe = []
        self.feas2_probe = []
        self.feas3_probe = []
        self.feas4_probe = []
        self.input = []
        self.input_probe = []
        self.label_probe = []
        self.path_shallow_probe = []
        self.path_deep_probe = []
        self.mse_test1 = []
        self.mse_test2 = []
        self.mse_test3 = []
        self.mse_test4 = []
        self.mse_test5 = []
        self.mse_test6 = []
        self.mse_test7 = []
        self.input_dimension = 0

        self.hier = nn.Sequential(
            *[nn.Sequential(
                    Linear1d(
                        h, h, bias
                    ),
                    nn.ReLU(),
                )
                for _ in range(0, num_layers)
            ],
        )
        self.beta = nn.Parameter(torch.randn(h, out_dim))

    def forward(self, x):
        self.input.append(x)
        #self.input_dimension = self.input_dimension + x.size(0)
        y = x.flatten(1)  # [bs, cin, space] -> [bs, cin * space]
        
        # Traverse through the sequential layers manually to save intermediate features
        for layer in self.hier:
            y = layer(y)
            self.feas.append(y)  # append intermediate feature to self.feas
        
        y = y @ self.beta / self.beta.size(0)
        return y
