import functools

import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter


class _routing(nn.Module):

    def __init__(self, in_channels, num_experts, dropout_rate):
        super(_routing, self).__init__()
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(in_channels, num_experts)

    def forward(self, x):
        x = torch.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return F.sigmoid(x)
    

class CondRotateConv2D(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', num_experts=3, dropout_rate=0.2):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(CondRotateConv2D, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

        self._avg_pooling = functools.partial(F.adaptive_avg_pool2d, output_size=(1, 1))
        self._routing_fn = _routing(in_channels, num_experts, dropout_rate)
        
        self.weight = Parameter(torch.Tensor(
            num_experts, out_channels, in_channels // groups, *kernel_size))

        self.weight_rotated = torch.Tensor(num_experts, out_channels, in_channels // groups, *kernel_size)
        
        self.kernel_theta = Parameter(torch.Tensor(
            num_experts))

        self.reset_parameters()

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        pooled_inputs = self._avg_pooling(input)  # routing function step 1 (Page 3 button)
        # pooled_inputs.shape = [1, c, h, w]
        routing_weights = self._routing_fn(pooled_inputs) # routing function step 2 & 3 (Page 3 button)
        # routing_weights.shape = [num_experts]

        for idx, theta in enumerate(self.kernel_theta):
            self.weight_rotated[idx] = rotate_kernel_3x3(self.weight[idx], kernel_theta=theta, rotate_mode='bilinear')

        kernels = torch.sum(routing_weights[:,None, None, None, None] * self.weight_rotated, 0)  # (a1W1 + ... + anWn)
        # [num_experts, 1, 1, 1, 1] x [num_experts, out_channels, in_channels // groups, kernel_h, kernel_w]
        # kernels.shape =  [out_channels, in_channels // groups, kernel_h, kernel_w]
        return self._conv_forward(input, kernels)


def rotate_kernel_3x3(weight, kernel_theta=0., rotate_mode='bilinear', sigma=0.1):
    if kernel_theta == 0.:
        return weight

    assert (weight.shape[2] == 3)
    assert (weight.shape[3] == 3)
    assert (-45. <= kernel_theta <= 45.)

    is_clockwise = kernel_theta > 0
    kernel_theta = kernel_theta if is_clockwise else -kernel_theta

    x = math.cos(kernel_theta / 180. * math.pi)
    y = math.sin(kernel_theta / 180. * math.pi)

    if rotate_mode == 'gaussian':
        if is_clockwise:
            T = torch.tensor([[x, y],
                              [-y, x]])
        else:
            T = torch.tensor([[x, -y],
                              [y, x]])
        A = torch.tensor([[-1., -1.], [-1., 0.], [-1., 1.],
                          [0, -1], [0, 0], [0, 1],
                          [1, -1], [1, 0], [1, 1]])
        A = A.transpose(0, 1)  # A: 2x9
        B = torch.mm(T, A)  # B: 2x9
        # print(A)
        # print(B)
        C = torch.mm(A.transpose(0, 1), B)  # C: 9x9
        A_sq = (torch.sum(A ** 2, 0).view(9, 1)).repeat(1, 9)  # 9x1 -> 9x9
        B_sq = (torch.sum(B ** 2, 0).view(1, 9)).repeat(9, 1)  # 1x9 -> 9x9
        Alpha = A_sq + B_sq - 2 * C

        # Beta = torch.zeros(9,9)
        # for i in range(9):
        #     for j in range(9):
        #         Beta[i,j] = torch.sum((A[:,i]-B[:,j])**2)
        # print(Alpha - Beta)
        Alpha = torch.exp(-Alpha / sigma)  # alpha: 9x9
        Alpha = Alpha / torch.sum(Alpha, 1).view(9, 1)
        Alpha[4, :] = torch.tensor([0., 0, 0, 0, 1, 0, 0, 0, 0])
        inp_c, out_c, _, _ = weight.shape
        weight = weight.view(inp_c * out_c, 9)
        weight = weight.transpose(0, 1)
        weight = (torch.mm(Alpha.cuda(), weight).transpose(0, 1)).view(inp_c, out_c, 3, 3)
    elif rotate_mode == 'bilinear':
        a = x - y
        b = x * y
        c = x + y
        Alpha = torch.tensor([[a, 1 - a, 0., 0., 0., 0., 0., 0., 0.],
                              [0., x - b, b, 0., 1 - c + b, y - b, 0., 0., 0.],
                              [0., 0.0, a, 0., 0.0, 1 - a, 0., 0., 0.],
                              [b, y - b, 0., x - b, 1 - c + b, 0., 0., 0., 0.],
                              [0., 0., 0., 0., 1., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 1 - c + b, x - b, 0., y - b, b],
                              [0., 0., 0., 1 - a, 0., 0., a, 0., 0.],
                              [0., 0., 0., y - b, 1 - c + b, 0., b, x - b, 0.],
                              [0., 0., 0., 0., 0., 0., 0., 1 - a, a]]).cuda()
        inp_c, out_c, _, _ = weight.shape
        inp_c, out_c, _, _ = weight.shape
        if is_clockwise:
            weight = weight.transpose(2, 3).contiguous().view(inp_c * out_c, 9)
            weight = weight.transpose(0, 1)
            weight = (torch.mm(Alpha, weight).transpose(0, 1)).view(inp_c, out_c, 3, 3).transpose(2, 3)
        else:
            weight = weight.view(inp_c * out_c, 9)
            weight = weight.transpose(0, 1)
            weight = (torch.mm(Alpha, weight).transpose(0, 1)).view(inp_c, out_c, 3, 3)
    else:
        raise ValueError(f'These is no {rotate_mode} mode.')
    
    return weight