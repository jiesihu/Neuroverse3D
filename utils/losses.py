import torch
import torch.nn as nn

def sigmoid_modified(pred, scale = 8, offset = 0.5):
    return torch.sigmoid(scale*(pred-offset))

class CustomActivationFunction1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = input.clone()
        output[input < 0] = 0
        output[input > 1] = 1
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input <= 0] = 0
        grad_input[input >= 1] = 0
        return grad_input

class CustomActivation1(nn.Module):
    def forward(self, input):
        return CustomActivationFunction1.apply(input)

# 示例用法
if __name__ == "__main__":
    activation = CustomActivation()

    # 输入张量
    input_tensor = torch.tensor([-1.0, 0.5, 1.5, 0.0, 1.0, 0.7], requires_grad=True)

    # 前向传播
    output = activation(input_tensor)
    print("Output:", output)

    # 计算梯度
    output.sum().backward()
    print("Input gradients:", input_tensor.grad)

def weighted_smooth_l1_loss( predictions, targets):
    # 为什么不用customized activation function + Dice loss？
    # 因为统一L1跟容易在生成与分割之间balance。
    # predictions: (batch_size, channels, depth, height, width)
    # targets: (batch_size, depth, height, width)


    batch_size, channels, depth, height, width = predictions.size()

    # Flatten predictions and targets
    predictions = predictions.view(-1)
    targets = targets.view(-1)

    # Determine the number of unique classes in the targets
    unique_classes = torch.unique(targets)
    num_classes = unique_classes.numel()

    # Calculate the frequency of each class in the targets
    class_counts = torch.bincount(targets.int(), minlength=num_classes).float()
    total_voxels = targets.int().numel()

    # Compute class frequencies
    class_frequencies = class_counts / (total_voxels+1e-5)

    # Compute weights: inverse of frequency
    weights = 1.0 / (class_frequencies + 1e-5)  # Adding a small value to avoid division by zero

    # Normalize weights to ensure sum of weights is equal to num_classes
    weights = weights / weights.sum() * num_classes


    # Create a weight tensor for each voxel in the targets
    weight_map = weights[targets.int()]

    # Make the weight_map.sum() equal to the all one map.
    scale = torch.prod(torch.tensor(weight_map.shape))/weight_map.sum()
    weight_map *= scale

#     print('weights:',weights)
#     print('scale:',scale)
#     print('weight_map.sum:',weight_map.sum())

    # Compute the L1 smooth loss
    l1_loss = F.smooth_l1_loss(predictions, targets.float(), reduction='none')

    # Apply weights to the loss
    weighted_loss = l1_loss * weight_map.detach()


    # Compute the mean loss
    mean_loss = weighted_loss.mean()

    return mean_loss

"""
Losses
Code form BrainID
"""

import torch
import torch.nn.functional as F
from torch import nn as nn 

def l1_loss(outputs, targets, weights = 1.):
    return torch.mean(abs(outputs - targets) * weights)

def l2_loss(outputs, targets, weights = 1.):
    return torch.mean((outputs - targets)**2 * weights)

def gaussian_loss(outputs_mu, outputs_sigma, targets, weights = 1.):
    variance = torch.exp(outputs_sigma)
    minusloglhood = 0.5 * torch.log(2 * torch.pi * variance) + 0.5 * ((targets - outputs_mu) ** 2) / variance
    return torch.mean(minusloglhood * weights)

def laplace_loss(outputs_mu, outputs_sigma, targets, weights = 1.):
    b = torch.exp(outputs_sigma)
    minusloglhood = torch.log(2 * b) + torch.abs(targets - outputs_mu) / b
    return torch.mean(minusloglhood, weights)


class GradientLoss(nn.Module):
    def __init__(self, mode = 'smoothl1', mask = False):
        super(GradientLoss, self).__init__() 
        self.mask = mask
        if mode == 'l1':
            self.loss_func = l1_loss
        elif mode == 'l2':
            self.loss_func = l2_loss
        elif mode == 'smoothl1':
            self.loss_func = F.smooth_l1_loss
        else:
            raise ValueError('Not supported loss_func for GradientLoss:', mode)

    def gradient(self, x): 
        # x: (b, c, s, r, c) -->  dx, dy, dz: (b, c, s, r, c) 
        back = F.pad(x, [0, 1, 0, 0, 0, 0])[:, :, :, :, 1:] 
        right = F.pad(x, [0, 0, 0, 1, 0, 0])[:, :, :, 1:, :] 
        bottom = F.pad(x, [0, 0, 0, 0, 0, 1])[:, :, 1:, :, :]

        dx, dy, dz = back - x, right - x, bottom - x
        dx[:, :, :, :, -1] = 0
        dy[:, :, :, -1] = 0
        dz[:, :, -1] = 0
        return dx, dy, dz
    
    def forward_archive(self, input, target):
        dx_i, dy_i, dz_i = self.gradient(input)
        dx_t, dy_t, dz_t = self.gradient(target)
        if self.mask:
            dx_i[dx_t == 0.] = 0.
            dy_i[dy_t == 0.] = 0.
            dz_i[dz_t == 0.] = 0.
        return (self.loss_func(dx_i, dx_t) + self.loss_func(dy_i, dy_t) + self.loss_func(dz_i, dz_t)).mean()

    def forward(self, input, target, weights = 1.):
        dx_i, dy_i, dz_i = self.gradient(input)
        dx_t, dy_t, dz_t = self.gradient(target)
        if self.mask:
            diff_dx = abs(dx_i - dx_t)
            diff_dy = abs(dy_i - dy_t)
            diff_dz = abs(dz_i - dz_t)
            diff_dx[target == 0.] = 0.
            diff_dy[target == 0.] = 0.
            diff_dz[target == 0.] = 0.
            return (diff_dx + diff_dy + diff_dz).mean()
        return ((self.loss_func(dx_i, dx_t, weights) + self.loss_func(dy_i, dy_t, weights) + self.loss_func(dz_i, dz_t, weights))/3).mean()

import torch
import torch.nn as nn
import torch.nn.functional as F
# 自定义Smooth L3 based L1 Loss函数
class SmoothL3_L1Loss(nn.Module):
    def __init__(self, beta=1.0, reduction='mean'):
        super(SmoothL3_L1Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, input, target):
        n = torch.abs(input - target)
        cond = n < self.beta
        loss = torch.where(cond, 0.333 * n**3 / self.beta**2, n + 0.333 * self.beta**3-self.beta)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

'''
smoothl3l1Loss = SmoothL3_L1Loss(beta=1.0)
loss = smoothl3l1Loss(output, target)

'''

