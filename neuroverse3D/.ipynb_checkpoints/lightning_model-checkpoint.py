import torch
from torch.nn import functional as F
import numpy as np

import pytorch_lightning as pl
from .models.neuroverse3D import Neuroverse3D

from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.pairwise_measures import BinaryPairwiseMeasures as PM
from utils.imagefilter import GINGroupConv_3D
from utils.losses import SmoothL3_L1Loss


class LightningModel(pl.LightningModule):
    """
    We use pytorch lightning to organize our model code
    """

    def __init__(self, hparams):
        super().__init__()
        # save hparams / load hparams
        self.save_hyperparameters(hparams)
        self.automatic_optimization = False
        
        # build model
        self.net = Neuroverse3D(in_channels = 1, 
                    out_channels = 1, 
                    stages = len(self.hparams.nb_inner_channels), 
                    dim = 2 if self.hparams.data_slice_only else 3,
                    inner_channels = self.hparams.nb_inner_channels,
                    conv_layers_per_stage = self.hparams.nb_conv_layers_per_stage)
    
        # Define the metrics you want to calculate
        self.metrics = ['DSC', 'L1', 'MSE', 'PSNR', 'SSIM']
        
        # losses
        self.smoothl3l1Loss = SmoothL3_L1Loss(beta=1.0)
        
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
    
    def gin_transform(self, image):
        # image shape: [batch, channel, h, w, d]
        gin = GINGroupConv_3D(interm_channel=10, scale_pool=[1,3,1,1], n_layer=4, device = self.device).to(self.device)
        with torch.no_grad():
            image_ = gin(image)
        
        # Create an empty tensor to store the result
        normalized = torch.empty_like(image_)
        batch = image_.shape[0]

        # Compute the minimum and maximum values for each batch
        min_vals = image_.view(batch, -1).min(dim=1, keepdim=True)[0].view(batch, 1, 1, 1, 1)
        max_vals = image_.view(batch, -1).max(dim=1, keepdim=True)[0].view(batch, 1, 1, 1, 1)

        # Normalize using broadcasting
        normalized = (image_ - min_vals) / (max_vals - min_vals)
        
        return normalized
    
    def training_step(self, batch, batch_idx):
        task =  batch['task'][0]
        dataset_name = batch['dataset'][0]
        
        # load data
        self.automatic_optimization = False
        imgs = batch['image'][0, :, None, :]
        labs = batch['label'][0, :, None, :]
        
        # Do GIN augmentation
        if np.random.rand() < self.hparams.gin_prob:
            imgs = self.gin_transform(imgs.float().to(device=self.device))
            
            
        num_rolls = imgs.shape[0]
        optimizer = self.optimizers()
        losses = []
        
        for i in range(num_rolls):
            imgs = torch.roll(imgs, shifts=1, dims=0)
            labs = torch.roll(labs, shifts=1, dims=0)
            
            
            target_in = imgs[:1,:]
            context_in = imgs[None, 1:,:]

            target_out = labs[:1,:]
            context_out = labs[None, 1:,:]

            target_in,context_in = torch.tensor(target_in).float().to(device=self.device), torch.tensor(context_in).float().to(device=self.device)
            target_out,context_out = torch.tensor(target_out).float().to(device=self.device), torch.tensor(context_out).float().to(device=self.device)

            # run
            mask = self.forward(target_in, context_in, context_out)

            # loss
            loss = self.custom_loss(mask, target_out, task)
            
            losses.append(loss)
            self.manual_backward(loss)
            
            # clip gradients
            if self.hparams.gradient_clip_val is not None:
                self.clip_gradients(optimizer, gradient_clip_val=self.hparams.gradient_clip_val, gradient_clip_algorithm="norm")
            else:
                self.clip_gradients(optimizer, gradient_clip_val=2.5, gradient_clip_algorithm="norm")
            
            optimizer.step()  
            optimizer.zero_grad()
            
        loss_avg = sum(losses) / (len(losses)+1e-5)
        self.log("train_loss", loss_avg)
        return {'loss': loss_avg}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.hparams.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=self.hparams.lr_decline_patience, min_lr=self.hparams.lr*0.01, verbose=True)
        print('The learning rate is:',self.hparams.lr)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',  
                'interval': 'epoch',    
                'frequency': 1,         
                'strict': True         
            }
        }
    
    
    def validation_step(self, batch, batch_idx):
        # load data
        imgs = batch['image'][0, :, None, :]
        labs = batch['label'][0, :, None, :]
        task =  batch['task'][0]
        dataset_name = batch['dataset'][0]
        
        def process_input(input_mod):
            if isinstance(input_mod, str):
                return input_mod
            elif isinstance(input_mod, torch.Tensor):
                return int(input_mod.item())
            else:
                return input_mod
        input_mod = process_input(batch['input'][0])
        output_mod = process_input(batch['output'][0])
        
        # run the inference multiple times
        num_rolls = imgs.shape[0]
        for i in range(num_rolls):
            imgs = torch.roll(imgs, shifts=1, dims=0)
            labs = torch.roll(labs, shifts=1, dims=0)
            
            target_in = imgs[:1,:]
            context_in = imgs[None, 1:,:]

            target_out = labs[:1,:]
            context_out = labs[None, 1:,:]

            target_in,context_in = torch.tensor(target_in).float().to(device=self.device), torch.tensor(context_in).float().to(device=self.device)
            target_out,context_out = torch.tensor(target_out).float().to(device=self.device), torch.tensor(context_out).float().to(device=self.device)

            # run
            mask = self.forward(target_in, context_in, context_out, gs = 4)

            # loss
            loss = self.custom_loss(mask, target_out, task)
            self.log('val_loss', loss)

            # Calculate additional metrics
            p_pred = (mask > 0.5).cpu().numpy()
            p_ref = target_out.cpu().numpy()
            bpm = PM(p_pred, p_ref, dict_args={"nsd": 1, "hd_perc": 95})

            if 'NSD' in self.metrics:
                nsd = bpm.normalised_surface_distance()
                self.log(f'{dataset_name}_{str(input_mod)}{str(output_mod)}_{task}_NSD', float(nsd))
            if 'ASSD' in self.metrics:
                assd = bpm.measured_average_distance()
                self.log(f'{dataset_name}_{str(input_mod)}{str(output_mod)}_{task}_ASSD', float(assd))
            if 'HD95' in self.metrics:
                hd95 = bpm.measured_hausdorff_distance_perc()
                self.log(f'{dataset_name}_{str(input_mod)}{str(output_mod)}_{task}_HD95', float(hd95))
            if 'DSC' in self.metrics:
                dsc = bpm.dsc()
                self.log(f'{dataset_name}_{str(input_mod)}{str(output_mod)}_{task}_DSC', float(dsc))

    def on_validation_epoch_end(self):
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step(self.trainer.callback_metrics["val_loss"])

    def custom_loss(self, output, target, key):
        loss = 50*self.smoothl3l1Loss(output, target)
        return loss
        
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