import torch
from torch.nn import functional as F
import numpy as np

import pytorch_lightning as pl
from .models.neuroverse3D import Neuroverse3D

class LightningModel(pl.LightningModule):
    """
    We use pytorch lightning to organize our model code
    """

    def __init__(self, hparams):
        super().__init__()
        # save hparams / load hparams
        self.save_hyperparameters(hparams)
        
        # build model
        self.net = Neuroverse3D(in_channels = 1, 
                    out_channels = 1, 
                    stages = len(self.hparams.nb_inner_channels), 
                    dim = 2 if self.hparams.data_slice_only else 3,
                    inner_channels = self.hparams.nb_inner_channels,
                    conv_layers_per_stage = self.hparams.nb_conv_layers_per_stage)
    
        
        
    def forward(self, target_in, context_in, context_out, gs= 3):
        '''
        target_in  (torch.Tensor): Shape [bs, 1, 128, 128, 128]
        context_in  (torch.Tensor): Shape [bs, L, 1, 128, 128, 128]. L is the number of contexts.
        context_out  (torch.Tensor): Shape [bs, L, 1, 128, 128, 128]
        gs (int; \ell): The size of mini-context.
        '''
        # run network
        y_pred = self.net(target_in.to(self.device), context_in, context_out, l = gs)
        return y_pred
    import torch

    def normalize_3d_volume(self, target_in: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Normalizes a batch of 3D volumes (shape: [..., D, H, W]) to the range [0, 1] independently for each sample.

        Args:
            target_in (torch.Tensor): Input tensor of shape [..., 1, D, H, W], where N is the batch size,
                                      and (D, H, W) are the depth, height, and width of the 3D volume.
            eps (float): A small value to prevent division by zero. Default is 1e-8.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as the input, where each sample is scaled to [0, 1].
        """
        # Ensure the input tensor is of type float32 to avoid integer division issues
        if target_in.dtype != torch.float32:
            target_in = target_in.to(torch.float32)

        # Compute the minimum and maximum values for each sample independently
        # Input shape: [N, 1, D, H, W] → Output shape: [N, 1, 1, 1, 1]
        min_vals = torch.amin(target_in, dim=(-3, -2, -1), keepdim=True)
        max_vals = torch.amax(target_in, dim=(-3, -2, -1), keepdim=True)

        # Compute the dynamic range and prevent division by zero
        dynamic_range = max_vals - min_vals
        dynamic_range[dynamic_range < eps] = eps  # Replace small ranges with eps to avoid division by zero

        # Normalize the input tensor to the range [0, 1]
        normalized = (target_in - min_vals) / dynamic_range

        return normalized