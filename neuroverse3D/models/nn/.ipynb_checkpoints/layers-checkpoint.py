from dataclasses import dataclass
from typing import Literal
from pydantic import validate_arguments
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *
from .vmap import Vmap, vmap
#     dim: Literal[2, 3]
@validate_arguments
@dataclass(eq=False, repr=False)
class ResidualUnit(nn.Module):
    channels: int
    dim: int
    conv_layers: int

    def __post_init__(self):
        super().__init__()
        conv_fn = getattr(nn, f'Conv{self.dim}d')
        layers = []
        for i in range(1, self.conv_layers):
            layers.append(conv_fn(in_channels=self.channels,
                                  out_channels=self.channels,
                                  kernel_size=3,
                                  padding='same'))
            layers.append(nn.GELU())
        layers.append(conv_fn(in_channels=self.channels,
                              out_channels=self.channels,
                              kernel_size=3,
                              padding='same'))
        self.layers = nn.Sequential(*layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        residual = self.layers(input)
        return F.gelu(input + residual)

    


@dataclass(eq=False, repr=False)
class DefaultUnetStageBlock(nn.Module):
    channels: int
    kwargs: Optional[Dict[str, Any]]
    dim: Literal[2, 3] = 2
    context_attention: bool = False

    def __post_init__(self):
        super().__init__()

    def forward(self,
                context: torch.Tensor,
                target: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The forward pass of a unet-stage block.

        Args:
            context (torch.Tensor): the context embedding, shape BxLxCxHxWxD or BxLxCxHxW
            target (torch.Tensor): the target embedding, shape BxCxHxWxL or BxCxHxW

        Returns:
            context, target: the processed tensors, same shape as input.
        """
        return context, target
    
@dataclass(eq=False, repr=False)
class ConvBlock_target_encoder(DefaultUnetStageBlock):

    def __post_init__(self):
        super().__post_init__()
        self.target_conv = ResidualUnit(channels=self.channels,
                                        dim=self.dim,
                                        conv_layers=self.kwargs['conv_layers_per_stage'])

    def forward(self,
                target: torch.Tensor,
                verbose: bool = False
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The forward pass of a unet-stage block.

        Args:
            target (torch.Tensor): the target embedding, shape BxCxHxWxL or BxCxHxW

        Returns:
            , target: the processed tensors, same shape as input.
        """

        # do single convs on input
        target = self.target_conv(target)  # B,C,...

        return target
    

    
@dataclass(eq=False, repr=False)
class ConvBlock_context_c2t(DefaultUnetStageBlock):

    def __post_init__(self):
        
        super().__post_init__()
        self.context_conv = Vmap(ResidualUnit(channels=self.channels,
                                              dim=self.dim,
                                              conv_layers=self.kwargs['conv_layers_per_stage']))
    def forward(self,
                context: torch.Tensor,
                verbose: bool=False,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The forward pass of a unet-stage block.

        Args:
            context (torch.Tensor): the context embedding, shape BxLxCxHxWxD or BxLxCxHxW
        Returns:
            context, target: the processed tensors, same shape as input.
        """
        # do single convs on input
        context = self.context_conv(context)  # B,L,C,...     

        # return augmented inputs
        return context
    
@dataclass(eq=False, repr=False)
class ConvBlock_context_t2c(DefaultUnetStageBlock):

    def __post_init__(self):
        super().__post_init__()
        self.context_conv = Vmap(ResidualUnit(channels=self.channels,
                                              dim=self.dim,
                                              conv_layers=self.kwargs['conv_layers_per_stage'])
                                 )

        conv_fn = getattr(nn, f'Conv{self.dim}d')

        self.combine_conv_context = Vmap(conv_fn(in_channels=2*self.channels,
                                         out_channels=self.channels,
                                         kernel_size=1,
                                         padding='same')
                                         )

    def forward(self,
                context: torch.Tensor,
                target: torch.Tensor,
                verbose: bool = False
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The forward pass of a unet-stage block.

        Args:
            context (torch.Tensor): the context embedding, shape BxLxCxHxWxD or BxLxCxHxW
            target (torch.Tensor): the target embedding, shape BxCxHxWxL or BxCxHxW

        Returns:
            context, target: the processed tensors, same shape as input.
        """
        # do single convs on input
        context = self.context_conv(context)  # B,L,C,...

        # concat on channels
        context_target = torch.concat(
            [context, target.unsqueeze(1).expand_as(context)], dim=2)  # B,L,2C,...

        # conv query with support
        context_update = self.combine_conv_context(context_target)
        if verbose: print('context_update:',context_update.shape)
        if verbose: print('context_update[0,0,16,4,4,4]:',context_update[0,0,16,4,4,4])
        if verbose: print('context_update[0,-1,16,4,4,4]:',context_update[0,-1,16,4,4,4])

        # resudual and activation
        context = F.gelu(context + context_update)

        # return augmented inputs
        return context
    
@dataclass(eq=False, repr=False)
class PairwiseConvAvgModelBlock_c2t(DefaultUnetStageBlock):

    def __post_init__(self):
        
        super().__post_init__()
        self.target_conv = ResidualUnit(channels=self.channels,
                                        dim=self.dim,
                                        conv_layers=self.kwargs['conv_layers_per_stage'])
        conv_fn = getattr(nn, f'Conv{self.dim}d')
        self.combine_conv_target = Vmap(conv_fn(in_channels=2*self.channels,
                                                out_channels=self.channels,
                                                kernel_size=1,
                                                padding='same')
                                        )


    def forward(self,
                context_mean: torch.Tensor,
                target: torch.Tensor,
                verbose: bool=False,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The forward pass of a unet-stage block.

        Args:
            context (torch.Tensor): the context embedding, shape BxLxCxHxWxD or BxLxCxHxW
            target (torch.Tensor): the target embedding, shape BxCxHxWxL or BxCxHxW

        Returns:
            context, target: the processed tensors, same shape as input.
        """
        # do single convs on input
        target = self.target_conv(target)  # B,C,...
            
        if verbose: print('-'*50)
        if verbose: print('context_mean[0,0,16,4,4,4]:',context_mean[0,0,16,4,4,4])
            
        context_target = torch.concat(
            [context_mean, target.unsqueeze(1).expand_as(context_mean)], dim=2)  # B,L,2C,...
        target_update = self.combine_conv_target(context_target)
        
        if verbose: print('target_update 1:',target_update.shape)
        target_update = target_update.mean(dim=1, keepdim=False)  # B,C,...
        if verbose: print('target_update 2:',target_update.shape)
                
        # resudual and activation
        target = F.gelu(target + target_update)

        # return augmented inputs
        return target



@dataclass(eq=False, repr=False)
class DefaultUnetOutputBlock(nn.Module):
    """
    U-net output block. Reduces channels to out_channels. Can be used to apply additional smoothing.
    """

    in_channels: int
    out_channels: int
    kwargs: Optional[Dict[str, Any]]
    dim: Literal[2, 3] = 2

    def __post_init__(self):
        super().__init__()

        conv_fn = getattr(nn, f"Conv{self.dim}d")

        self.block = nn.Sequential(
            conv_fn(in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=1,
                    padding='same')
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): input, shape BxCinxHxWxL or BxCinxHxW

        Returns:
            torch.Tensor: output, shape BxCinxHxWxL or BxCinxHxW
        """
        return self.block(input)
    

@dataclass(eq=False, repr=False)
class UnetDownsampleAndCreateShortcutBlock(nn.Module):
    in_channels: int
    out_channels: int
    dim: Literal[2, 3]
    context_filled: bool = True
    target_filled: bool = True   

    def __post_init__(self):
        super().__init__()
        self.needs_channel_asjustment = self.in_channels != self.out_channels
        if self.needs_channel_asjustment:
            conv_fn = getattr(nn, f"Conv{self.dim}d")
            self.context_linear_layer = Vmap(conv_fn(in_channels=self.in_channels,
                                                     out_channels=self.out_channels,
                                                     kernel_size=4,
                                                     stride=2,
                                                     padding=1)
                                             ) if self.context_filled else None
            self.target_linear_layer = conv_fn(in_channels=self.in_channels,
                                               out_channels=self.out_channels,
                                               kernel_size=4,
                                               stride=2,
                                               padding=1) if self.target_filled else None
        else:
            pool_fn = getattr(nn, f"MaxPool{self.dim}d")
            self.context_pooling_layer = Vmap(
                pool_fn(kernel_size=2)) if self.context_filled else None
            self.target_pooling_layer = pool_fn(kernel_size=2) if self.target_filled else None

    def forward(self,
                context: torch.Tensor,
                target: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        # make shortcut
        shortcut = (context, target)
        # downsample
        if self.needs_channel_asjustment:
            context = self.context_linear_layer(context) if self.context_filled else context
            target = self.target_linear_layer(target) if self.target_filled else target
        else:
            context = self.context_pooling_layer(context) if self.context_filled else context
            target = self.target_pooling_layer(target) if self.target_filled else target
            
        return context, target, shortcut
    
    

@dataclass(eq=False, repr=False)
class UnetUpsampleAndConcatShortcutBlock(nn.Module):
    in_channels: int
    in_shortcut_channels: int
    out_channels: int
    dim: Literal[2, 3]
    context_filled: bool = True
    target_filled: bool = True   

    def __post_init__(self):
        super().__init__()
        self.upsampling_layer = nn.Upsample(
            scale_factor=2, mode='trilinear' if self.dim == 3 else 'bilinear', align_corners=False)

        conv_fn = getattr(nn, f"Conv{self.dim}d")
        self.context_conv_layer = Vmap(conv_fn(in_channels=self.in_channels + self.in_shortcut_channels,
                                               out_channels=self.out_channels,
                                               kernel_size=1,
                                               padding='same')
                                       ) if self.context_filled else None
        self.target_conv_layer = conv_fn(in_channels=self.in_channels + self.in_shortcut_channels,
                                         out_channels=self.out_channels,
                                         kernel_size=1,
                                         padding='same') if self.target_filled else None

    def forward(self,
                context: torch.Tensor,
                target: torch.Tensor,
                shortcut: Tuple[torch.Tensor, torch.Tensor]
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        # upsample
        context = vmap(self.upsampling_layer,
                       context) if self.context_filled else context
        target = self.upsampling_layer(target) if self.target_filled else target

        # concat with shortcut
        ctx_short, tgt_short = shortcut
        
        # B L C ...
        context = torch.cat([context, ctx_short],
                            dim=2) if self.context_filled else context
        target = torch.cat([target, tgt_short], dim=1) if self.target_filled else target  # B C ...

        # reduce dim
        context = self.context_conv_layer(
            context) if self.context_filled else context
        target = self.target_conv_layer(target) if self.target_filled else target

        return context, target

@dataclass(eq=False, repr=False)
class PairwiseConvAvgModelOutput(DefaultUnetOutputBlock):
    """
    U-net output block. Reduces channels to out_channels. Can be used to apply additional smoothing.
    """

    def __post_init__(self):
        super().__post_init__()
        conv_fn = getattr(nn, f"Conv{self.dim}d")

        self.block = nn.Sequential(
            ResidualUnit(channels=self.in_channels,
                         dim=self.dim,
                         conv_layers=self.kwargs['conv_layers_per_stage']),
            conv_fn(in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=1,
                    padding='same')
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): input, shape BxCinxHxWxL or BxCinxHxW

        Returns:
            torch.Tensor: output, shape BxCinxHxWxL or BxCinxHxW
        """
        return self.block(input)
    
    

