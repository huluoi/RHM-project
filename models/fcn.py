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
        self.feas5_probe = []
        self.feas6_probe = []
        self.feas7_probe = []
        self.feas8_probe = []
        self.feas9_probe = []
        self.feas10_probe = []
        self.feas11_probe = []
        self.feas12_probe = []
        self.input = []
        self.input_probe = []
        self.label_probe = []
        self.path_shallow_probe = []
        self.path_mid_probe = []
        self.path_deep_probe = []
        self.mse_test1 = []
        self.mse_test2 = []
        self.mse_test3 = []
        self.mse_test4 = []
        self.mse_test5 = []
        self.mse_test6 = []
        self.mse_test7 = []
        self.mse_test8 = []
        self.mse_test9 = []
        self.mse_test10 = []
        self.mse_test11 = []
        self.mse_test12 = []
        self.mse_test13 = []
        self.mse_test14 = []
        self.mse_test15 = []
        self.mse_test16 = []
        self.mse_test16 = []
        self.mse_test17 = []
        self.mse_test18 = []
        self.mse_test19 = []
        self.mse_test20 = []
        self.mse_test21 = []
        self.mse_test22 = []
        self.mse_test23 = []
        self.mse_test24 = []
        self.mse_test25 = []
        self.mse_test26 = []
        self.mse_test27 = []
        self.mse_test28 = []
        self.mse_test29 = []
        self.mse_test30 = []
        self.ETF_metric_1 = []
        self.ETF_metric_2 = []
        self.ETF_metric_3 = []
        self.ETF_metric_4 = []
        self.ETF_metric_5 = []
        self.ETF_metric_6 = []
        self.nc1_list_1 = []
        self.nc1_list_2 = []
        self.nc1_list_3 = []
        self.nc1_list_4 = []
        self.nc1_list_5 = []
        self.nc1_list_6 = []
        self.WH_relation_metric1 = []
        self.WH_relation_metric2 = []
        self.WH_relation_metric3 = []
        self.WH_relation_metric4 = []
        self.WH_relation_metric5 = []
        self.WH_relation_metric6 = []


        self.input_dimension = 0

        self.hier = nn.Sequential(
            *[nn.Sequential(
                    Linear1d(
                        input_channels, h, bias
                    ),
                    nn.ReLU(),
                )],
            *[nn.Sequential(
                    Linear1d(
                        h, h, bias
                    ),
                    nn.ReLU(),
                )
                for _ in range(1, num_layers)
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
