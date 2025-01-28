import torch
from torch import nn

class NonOverlappingConv1d(nn.Module):
    def __init__(
        self, input_channels, out_channels, out_dim, patch_size, bias=False
    ):
        super(NonOverlappingConv1d, self).__init__()
        self.weight = nn.Parameter( # input [bs, cin, space / 2, 2], weight [cout, cin, 1, 2]
            torch.randn(
                out_channels,
                input_channels,
                1,
                patch_size,
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(1, out_channels, 1))
        else:
            self.register_parameter("bias", None)

        self.input_channels = input_channels
        self.patch_size = patch_size

    def forward(self, x):
        bs, cin, d = x.shape
        if (d >= self.patch_size):
            x = x.view(bs, 1, cin, d // self.patch_size, self.patch_size) # [bs, 1, cin, space // patch_size, patch_size]
        else:
            x = x.view(bs, 1, -1, 1, 1)
        x = x * self.weight # [bs, cout, cin, space // patch_size, patch_size]
        x = x.sum(dim=[-1, -3]) # [bs, cout, space // patch_size]
        x /= self.input_channels ** .5
        if self.bias is not None:
            x += self.bias.expand_as(x) * 0.1
        return x


class CNN(nn.Module):
    def __init__(self, input_channels, h, out_dim, num_layers, patch_size=2, bias=False):
        super(CNN, self).__init__()

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

        d = patch_size ** num_layers
        self.d = d

        self.hier = nn.Sequential(
            *[nn.Sequential(
                NonOverlappingConv1d(
                input_channels, h, d // patch_size, patch_size, bias
            ),
            nn.ReLU(),
            )],
            *[nn.Sequential(
                    NonOverlappingConv1d(
                        h, h, d // patch_size ** (l + 1), patch_size, bias
                    ),
                    nn.ReLU(),
                )
                for l in range(1, num_layers)
            ],
        )
        self.beta = nn.Parameter(torch.randn(h, out_dim))

    def forward(self, x):
        # Traverse through the sequential layers manually to save intermediate features
        y = x
        self.input.append(x)
        for layer in self.hier:
            y = layer(y)
            self.feas.append(y)  # append intermediate feature to self.feas
        y = y.mean(dim=[-1])
        y = y @ self.beta / self.beta.size(0)
        return y


class CNNLayerWise(nn.Module):    # only for patch_size = 2!!
    """
        CNN crafted to have an effective size equal to the corresponding HLCN.
        Trainable layerwise.
    """

    def __init__(self, input_channels, h, out_dim, num_layers, bias=False):
        super(CNNLayerWise, self).__init__()

        d = 2 ** num_layers
        self.d = d
        self.h = h
        self.out_dim = out_dim

        layers = []
        for l in range(num_layers):
            layers.append(NonOverlappingConv1d(h if l else input_channels, h, d // 2 ** (l + 1), bias))
            layers.append(nn.ReLU())
        self.layers = layers

        self.hier = nn.Sequential(*layers)
        self.beta = nn.Parameter(torch.randn(h, out_dim))

    def init_layerwise_(self, l):        # this allows to keep the parameters but change the layer which is trained
        device = self.beta.device
        self.hier = nn.Sequential(*self.layers[:2 * (l + 1)])
        self.beta = nn.Parameter(torch.randn(self.h, self.out_dim, device=device))

    def forward(self, x):
        if len(self.hier) == len(self.layers):
            y = self.hier(x)
        else:
            with torch.no_grad():        # this trick allows to not train everything going under no_grad()
                y = self.hier[:-2](x)
            y = self.hier[-2:](y)
        y = y.mean(dim=[-1])
        y = y @ self.beta / self.beta.size(0)
        return y