import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from e2cnn import gspaces, nn as e2nn  # Importing the equivariant CNN library


class MaskedConv2d(nn.Conv2d):
    """ Standard Masked Convolution for PixelCNN """

    def __init__(self, mask_type, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=True):
        super(MaskedConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        assert mask_type in ['A', 'B'], "Mask type must be 'A' or 'B'"

        self.register_buffer('mask', torch.ones_like(self.weight))
        _, _, h, w = self.weight.shape
        self.mask[:, :, h // 2, w // 2 + (mask_type == 'B'):] = 0  # Right half of center pixel
        self.mask[:, :, h // 2 + 1:, :] = 0  # Rows below the center

    def forward(self, x):
        return F.conv2d(x, self.weight * self.mask, self.bias, self.stride, self.padding)


class EquivariantMaskedConv2d(nn.Module):
    def __init__(self, mask_type, in_type, out_type, kernel_size, stride=1, padding=1, bias=False):
        super().__init__()
        assert mask_type in ['A', 'B'], "Mask type must be 'A' or 'B'"

        self.conv = e2nn.R2Conv(in_type, out_type, kernel_size, stride=stride, padding=padding, bias=bias)

        # Register mask as a buffer
        mask = torch.ones(1, 1, kernel_size, kernel_size)
        _, _, h, w = mask.shape
        mask[:, :, h // 2, w // 2 + (mask_type == 'B'):] = 0  # Block the center & future pixels
        mask[:, :, h // 2 + 1:, :] = 0  # Block lower pixels

        self.register_buffer('mask', mask)

    def forward(self, input):
        # Expand the weights to get the convolution kernels
        expanded_weights, expanded_bias = self.conv.expand_parameters()
        # Apply the mask to the expanded weights
        masked_weights = expanded_weights * self.mask 
        # Perform the convolution using the masked weights
        output = nn.functional.conv2d(
            input,
            masked_weights,
            bias=expanded_bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups
        )
        # Wrap the output in a GeometricTensor
        return e2nn.GeometricTensor(output, self.conv.out_type)

    def export(self):
        # Export the internal R2Conv layer
        return self.conv.export()



class Activation(nn.Module):
    def __init__(self, activation_func):
        super().__init__()
        if activation_func == 'gated':
            self.activation = lambda x: x * torch.sigmoid(x)
        elif activation_func == 'relu':
            self.activation = F.relu

    def forward(self, input):
        return self.activation(input)

class EquivariantPixelCNN(nn.Module):
    def __init__(self, configs, dataDims):
        super().__init__()
        self.act_func = configs.activation_function
        self.filters = configs.conv_filters
        self.layers = configs.conv_layers
        self.activation = Activation(self.act_func)
        kernel_size = configs.conv_size
        padding = kernel_size // 2
        channels = dataDims['channels']
        outmaps = dataDims['classes'] + 1
        initial_filters=4

        self.fc_depth=configs.fc_depth
        f_in = (np.ones(self.layers+ 1) * self.filters).astype(int)
        f_in[0        ] = self.filters*4
        f_out = (np.ones(self.layers + 1) * self.filters).astype(int)

        # Define the rotation-equivariant group: C4 (4-fold rotational symmetry)
        self.r2_act = gspaces.Rot2dOnR2(N=2)  # Rotational equivariance with 4 angles

        # Input and output representation spaces
        self.input_type = e2nn.FieldType(self.r2_act, channels * [self.r2_act.trivial_repr])
        self.hidden_type = e2nn.FieldType(self.r2_act, self.filters * [self.r2_act.regular_repr])
        self.output_type = e2nn.FieldType(self.r2_act, outmaps * [self.r2_act.trivial_repr])

        # Initial masked convolution (Type 'A')
        self.initial_conv = EquivariantMaskedConv2d('A', self.input_type, self.hidden_type, kernel_size, padding=padding,bias=False)
        self.conv_layers = nn.ModuleList([
            EquivariantMaskedConv2d('B', self.hidden_type, self.hidden_type, kernel_size, padding=padding,bias=True)
            for _ in range(self.layers)
        ])
#  og1workedelongated      self.conv_layers = nn.ModuleList(
#            [MaskedConv2d('B', f_in[i], f_out[i] , kernel_size, padding) for i in range(self.layers)]
#        )
        if configs.fc_norm is None:
            self.fc_norm = nn.Identity()
        elif configs.fc_norm == 'batch':
            self.fc_norm =nn.BatchNorm2d(self.fc_depth)
        else:
            print(configs.fc_norm + ' is not an implemented norm')
            sys.exit()
        # Hidden masked convolutions (Type 'B')
        # self.conv_layers = nn.ModuleList([
        #     EquivariantMaskedConv2d('B', self.hidden_type, self.hidden_type, kernel_size, padding=padding)
        #     for _ in range(self.layers)
        # ])
        self.fc_dropout = nn.Dropout(configs.fc_dropout_probability)
        # Fully connected layers
#og1        self.fc1 = nn.Conv2d(f_out[-1], self.fc_depth, kernel_size=(1,1), bias=True)  # add skip connections
        self.fc1 = nn.Conv2d(self.filters*2, self.fc_depth, kernel_size=(1,1), bias=True)
        self.fc2 = nn.Conv2d(self.fc_depth, outmaps * channels, kernel_size=(1,1), bias=True) # gated activation cuts filters by 2


    def forward(self, x):
       # x = e2nn.GeometricTensor(x, self.input_type)  # Convert to an equivariant tensor
       # print( self.initial_conv(x))
        x = self.initial_conv(x)  # Initial masked convolution

        if isinstance(x, e2nn.GeometricTensor):
           x = x.tensor  # Extract tensor if it's a GeometricTensor
     #   print([x.shape,'1'])
        x = self.activation(x)

        for layer in self.conv_layers:
            x = layer(x)
            x = x.tensor
            x = self.activation(x)

       # x = x.tensor
       #  print(self.filters * [self.r2_act.regular_repr])
       # 
        x = self.fc1(x)
        x = self.fc_norm(x)
      #  x = x.tensor
        x = self.activation(x)
        x = self.fc_dropout(x)
        x = self.fc2(x)
        # print(x.shape)
        return x  # Convert back to a standard tensor
