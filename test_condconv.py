import torch
from torch import nn
from condconv.condconv import CondConv2D


batch_size = 1 # You need update param a sample.


class Model(nn.Module):
    def __init__(self, num_experts):
        super(Model, self).__init__()
        self.condconv2d = CondConv2D(10, 128, kernel_size=1, num_experts=num_experts, dropout_rate=dropout_rate)
        
    def forward(self, x):
        x = self.condconv2d(x)


input_tensor = torch.rand(1, 3, 224, 224)
conv_layer = CondConv2D(num_experts=1, in_channels=3, out_channels=6, kernel_size=3, 
                stride=1, padding=0, dilation=1, groups=1, bias=False,
                padding_mode='zeros')
output = conv_layer(input_tensor)
print(output.shape)
print("end")