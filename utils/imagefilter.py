'''
Code modified from https://github.com/cheng-01037/Causality-Medical-Image-Domain-Generalization/blob/main/models/imagefilter.py
'''
# GIN

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class GradlessGCReplayNonlinBlock_3D(nn.Module):
    def __init__(self, out_channel=32, in_channel=3, scale_pool=[1, 3], layer_id=0, use_act=True, requires_grad=False,device = 'cpu', **kwargs):
        """
        Conv-leaky relu layer. Efficient implementation by using group convolutions
        """
        super(GradlessGCReplayNonlinBlock_3D, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.scale_pool = scale_pool
        self.layer_id = layer_id
        self.use_act = use_act
        self.device = device
        self.requires_grad = requires_grad
        assert requires_grad == False
        
        # random size of kernel
        idx_k = torch.randint(high=len(self.scale_pool), size=(1,))
        self.k = self.scale_pool[idx_k[0]]
        
        self.ker = torch.randn([self.out_channel, self.in_channel, self.k, self.k, self.k], requires_grad=self.requires_grad).to(device=self.device)
        self.shift = torch.randn([self.out_channel, 1, 1, 1], requires_grad=self.requires_grad).to(device=self.device) * 1.0

    def forward(self, x_in, requires_grad=False):
        """
        Args:
            x_in: [ nb (original), nc (original), nx, ny, nz ]
        """
        
        nb, nc, nx, ny, nz = x_in.shape
        
        k = self.k
        ker = self.ker
        shift = self.shift

        x_conv = F.conv3d(x_in, ker, stride=1, padding=k // 2, dilation=1)
        x_conv = x_conv + shift
        if self.use_act:
            x_conv = F.leaky_relu(x_conv)


        return x_conv


class GINGroupConv_3D(nn.Module):
    def __init__(self, out_channel=1, in_channel=1, interm_channel=2, scale_pool=[1], n_layer=4, out_norm='frob', device = 'cpu', **kwargs):
        '''
        GIN
        '''
        super(GINGroupConv_3D, self).__init__()
        self.scale_pool = scale_pool  # don't make it too large as we have multiple layers
        self.n_layer = n_layer
        self.layers = []
        self.out_norm = out_norm
        self.out_channel = out_channel
        self.device = device

        self.layers.append(
            GradlessGCReplayNonlinBlock_3D(out_channel=interm_channel, in_channel=in_channel, scale_pool=[1], layer_id=0, device = device).to(device=self.device)
        )
        for ii in range(n_layer - 2):
            self.layers.append(
                GradlessGCReplayNonlinBlock_3D(out_channel=interm_channel, in_channel=interm_channel, scale_pool=scale_pool, layer_id=ii + 1, device = device).to(device=self.device)
            )
        self.layers.append(
            GradlessGCReplayNonlinBlock_3D(out_channel=out_channel, in_channel=interm_channel, scale_pool=[1], layer_id=n_layer - 1, use_act=False, device = device).to(device=self.device)
        )

        self.layers = nn.ModuleList(self.layers)
        
        self.alphas = 0.7+0.3*torch.rand(1).to(device = self.device)

    def forward(self, x_in):
        if isinstance(x_in, list):
            x_in = torch.cat(x_in, dim=0)

        nb, nc, nx, ny, nz = x_in.shape

        alphas = self.alphas
        
        x = self.layers[0](x_in)
        for blk in self.layers[1:]:
            x = blk(x)
        mixed = alphas * x + (1.0 - alphas) * x_in

        if self.out_norm == 'frob':
            _in_frob = torch.norm(x_in.view(nb, nc, -1), dim=(-1, -2), p='fro', keepdim=False)
            _in_frob = _in_frob[:, None, None, None, None].repeat(1, nc, 1, 1, 1)
            _self_frob = torch.norm(mixed.view(nb, self.out_channel, -1), dim=(-1, -2), p='fro', keepdim=False)
            _self_frob = _self_frob[:, None, None, None, None].repeat(1, self.out_channel, 1, 1, 1)
            mixed = mixed * (1.0 / (_self_frob + 1e-5)) * _in_frob

        return mixed
