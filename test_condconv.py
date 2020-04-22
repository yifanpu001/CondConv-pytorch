from condconv.condconv import CondConv2D


input_tensor = torch.rand(2, 4, 224, 224)
conv_layer = CondConv2d(num_experts=6, in_channels=4, out_channels=8, kernel_size=3, 
                stride=1, padding=0, dilation=1, groups=1, bias=False,
                padding_mode='zeros')
output = conv_layer(input_tensor)
print("end")