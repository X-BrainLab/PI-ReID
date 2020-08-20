import torch
from torch import nn
from torch.nn import functional as F
from time import time


class _Query_Guided_Attention(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_Query_Guided_Attention, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            self.max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            self.max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            self.max_pool_layer1 = nn.MaxPool2d(kernel_size=(4, 4))
            self.max_pool_layer2 = nn.MaxPool2d(kernel_size=(8, 8))
            self.gmp = nn.AdaptiveMaxPool2d(1)

            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            self.max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        # self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
        #                  kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        # self.compress_attention = conv_nd(in_channels=3, out_channels=1,
        #                    kernel_size=1, stride=1, padding=0)
        # if sub_sample:
        #     # self.g = nn.Sequential(self.g, max_pool_layer)
        #     self.phi = nn.Sequential(self.phi, max_pool_layer)

        self.relu = nn.ReLU()

        # self.gmp = nn.AdaptiveMaxPool1d(1, return_indices=True)

    def forward(self, x, x_g, attention="x", pyramid="no"):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)
        theta_x = self.theta(x)
        phi_x = self.phi(x_g)

        if attention == "x":
            theta_x = theta_x.view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)

            if pyramid == "yes":
                phi_x1 = self.max_pool_layer(phi_x).view(batch_size, self.inter_channels, -1)
                f = torch.matmul(theta_x, phi_x1)
                N = f.size(-1)
                f_div_C1 = f / N

                phi_x2 = phi_x.view(batch_size, self.inter_channels, -1)
                f = torch.matmul(theta_x, phi_x2)
                f_div_C2 = f / N

                phi_x3 = self.max_pool_layer1(phi_x).view(batch_size, self.inter_channels, -1)
                f = torch.matmul(theta_x, phi_x3)
                f_div_C3 = f / N

                phi_x4 = self.max_pool_layer1(phi_x).view(batch_size, self.inter_channels, -1)
                f = torch.matmul(theta_x, phi_x4)
                f_div_C4 = f / N

                phi_x5 = self.gmp(phi_x).view(batch_size, self.inter_channels, -1)
                f = torch.matmul(theta_x, phi_x5)
                f_div_C5 = f / N

                f_div_C = torch.cat((f_div_C1, f_div_C2, f_div_C3, f_div_C4, f_div_C5), 2)
            elif pyramid == "no":
                phi_x1 = phi_x.view(batch_size, self.inter_channels, -1)
                f = torch.matmul(theta_x, phi_x1)
                N = f.size(-1)
                f_div_C = f / N
            elif pyramid == "s2":
                phi_x1 = self.max_pool_layer(phi_x).view(batch_size, self.inter_channels, -1)
                f = torch.matmul(theta_x, phi_x1)
                N = f.size(-1)
                f_div_C = f / N

            f, max_index = torch.max(f_div_C, 2)
            f = f.view(batch_size, *x.size()[2:]).unsqueeze(1)

            W_y = x * f
            z = W_y + x

            return z, f.squeeze()

        elif attention == "x_g":
            phi_x = phi_x.view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
            theta_x = theta_x.view(batch_size, self.inter_channels, -1)
            f = torch.matmul(phi_x, theta_x)
            N = f.size(-1)
            f_div_C = f / N
            f, max_index = torch.max(f_div_C, 2)
            f = f.view(batch_size, *x_g.size()[2:]).unsqueeze(1)

            W_y = x_g * f
            z = W_y + x_g

            return z, f

class Query_Guided_Attention(_Query_Guided_Attention):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(Query_Guided_Attention, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)