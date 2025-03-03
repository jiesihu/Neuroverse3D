import nibabel as nib
import numpy as np
import os

def size_check(images):
    assert images.shape[2:] == (128, 128, 128), f"Image shape {img_data.shape[2:]} mismatch for (128, 128, 128)"

def split_target(images, index):
    """
    Splits the images array into two parts: target_in and target_out.
    
    Parameters:
        index (int): The index of the image to be used as target_in.
    """
    
    # Extract target_in from the image at the specified index
    target = images[index:index+1]  # Shape [1, 1, 128, 128, 128]
    
    # Extract target_out as all images except the one at index
    context = np.delete(images, index, axis=0)  # Shape [N-1, 1, 128, 128, 128]
    
    return target, context


def clip_intensity(image, lower_percentile=2, upper_percentile=98):
    """
    Clip the intensity of a numpy array (image) to the specified percentiles.

    Parameters:
    - image: numpy array, the input image (could be a 2D, 3D, or any n-dimensional array).
    - lower_percentile: float, the lower percentile (default is 5).
    - upper_percentile: float, the upper percentile (default is 95).

    Returns:
    - clipped_image: numpy array with intensity clipped to the percentiles.
    """
    # Calculate the lower and upper percentile values
    lower_value = np.percentile(image, lower_percentile)
    upper_value = np.percentile(image, upper_percentile)

    # Clip the image intensities
    clipped_image = np.clip(image, lower_value, upper_value)

    return clipped_image

def load_gen_data(image_dir, modality = '0000'):
    images = []
    labels = []
    files = [i for i in sorted(os.listdir(image_dir)) if modality in i]
    for i in files: 
        img_path = os.path.join(image_dir, i)
        img_nii = nib.load(img_path)
        img_data = img_nii.get_fdata()
        img_data = clip_intensity(img_data)

        images.append(img_data[np.newaxis, ...]) 

    images = np.stack(images, axis=0)
    
    return images

def load_seg_data(image_dir, label_dir):
    images = []
    labels = []
    
    for i in range(1, 4): 
        img_path = os.path.join(image_dir, f'{i:03d}_T1_0000.nii.gz')
        label_path = os.path.join(label_dir, f'{i:03d}_T1.nii.gz')

        img_nii = nib.load(img_path)
        label_nii = nib.load(label_path)

        img_data = img_nii.get_fdata()
        img_data = clip_intensity(img_data)
        label_data = label_nii.get_fdata()

        images.append(img_data[np.newaxis, ...]) 
        labels.append(label_data[np.newaxis, ...])

    images = np.stack(images, axis=0)
    labels = np.stack(labels, axis=0)
    
    return images, labels

import torch
def normalize_3d_volume(target_in: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
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
    min_vals = torch.amin(target_in, dim=(-3,-2,-1), keepdim=True)
    max_vals = torch.amax(target_in, dim=(-3,-2,-1), keepdim=True)
    
    # Compute the dynamic range and prevent division by zero
    dynamic_range = max_vals - min_vals
    dynamic_range[dynamic_range < eps] = eps  # Replace small ranges with eps to avoid division by zero
    
    # Normalize the input tensor to the range [0, 1]
    normalized = (target_in - min_vals) / dynamic_range
    
    return normalized


def structure_data(images_, labels, index = 0, verbose = False):
    '''
    index: index for selecting the target image
    '''
    target_in, context_in = split_target(images_, index)
    target_out_raw, context_out_raw = split_target(labels, index)

    # Duplicate context for each target image
    context_out_raw = np.repeat(context_out_raw[np.newaxis, ...], target_in.shape[0],axis = 0)
    context_in = np.repeat(context_in[np.newaxis, ...], target_in.shape[0],axis = 0)
    
    if verbose:
        print('Shape of target_in:',target_in.shape, '\nShape of target_out:',target_out_raw.shape)
        print('Shape of context_in:',context_in.shape, '\nShape of context_out:',context_out_raw.shape)
        print('The context size is:', context_in.shape[1])
    return target_in, context_in, target_out_raw, context_out_raw



import matplotlib.pyplot as plt
def plot_pred(target_in, target_out, context_in, context_out, mask, slice_, task = 'seg'):
    plt.figure(figsize = (25,4))

    plt.subplot(1,7,1)
    plt.imshow(target_in[0,0,:,:,slice_].T.cpu(),cmap = 'gray')
    plt.axis('off')
    plt.title('Input')

    plt.subplot(1,7,2)
    plt.imshow(target_out[0,0,:,:,slice_].T.cpu(),cmap = 'gray')
    plt.axis('off')
    plt.title('GT')

    plt.subplot(1,7,3)
    if task == 'seg':
        plt.imshow(mask[0,0,:,:,slice_].T.cpu()>0.5,cmap = 'gray')
    elif task == 'gen':
        plt.imshow(mask[0,0,:,:,slice_].T.cpu(),cmap = 'gray')
    plt.axis('off')
    plt.title('Prediction')

    plt.subplot(1,7,4)
    plt.imshow(context_in[0,0,0,:,:,slice_].T.cpu(),cmap = 'gray')
    plt.axis('off')
    plt.title('Context 1 Image')

    plt.subplot(1,7,5)
    plt.imshow(context_out[0,0,0,:,:,slice_].T.cpu(),cmap = 'gray')
    plt.axis('off')
    plt.title('Context 1 Mask')
    
    plt.subplot(1,7,6)
    plt.imshow(context_in[0,1,0,:,:,slice_].T.cpu(),cmap = 'gray')
    plt.axis('off')
    plt.title('Context 2 Image')

    plt.subplot(1,7,7)
    plt.imshow(context_out[0,1,0,:,:,slice_].T.cpu(),cmap = 'gray')
    plt.axis('off')
    plt.title('Context 2 Mask')

    plt.tight_layout()
    
def plot_pred2(target_in, target_out, context_in, context_out, mask, slice_, task = 'seg'):
    plt.figure(figsize = (25,4))

    plt.subplot(1,7,1)
    plt.imshow(torch.flip(target_in[0,0,:,slice_,:].T.cpu(), dims=[-2]), cmap = 'gray')
    plt.axis('off')
    plt.title('Input')
#     plt.colorbar()

    plt.subplot(1,7,2)
    plt.imshow(torch.flip(target_out[0,0,:,slice_,:].T.cpu(), dims=[-2]), cmap = 'gray')
    plt.axis('off')
    plt.title('GT')
#     plt.colorbar()

    plt.subplot(1,7,3)
    if task == 'seg':
        plt.imshow(torch.flip(mask[0,0,:,slice_,:].T.cpu()>0.5, dims=[-2]), cmap = 'gray')
    elif task == 'gen':
        plt.imshow(torch.flip(mask[0,0,:,slice_,:].T.cpu(), dims=[-2]),cmap = 'gray')
    plt.axis('off')
    plt.title('Prediction')
#     plt.colorbar()

    plt.subplot(1,7,4)
    plt.imshow(torch.flip(context_in[0,0,0,:,slice_,:].T.cpu(), dims=[-2]),cmap = 'gray')
    plt.axis('off')
    plt.title('Context 1 Image')
#     plt.colorbar()

    plt.subplot(1,7,5)
    plt.imshow(torch.flip(context_out[0,0,0,:,slice_,:].T.cpu(), dims=[-2]),cmap = 'gray')
    plt.axis('off')
    plt.title('Context 1 Mask')
#     plt.colorbar()
    
    plt.subplot(1,7,6)
    plt.imshow(torch.flip(context_in[0,1,0,:,slice_,:].T.cpu(), dims=[-2]),cmap = 'gray')
    plt.axis('off')
    plt.title('Context 2 Image')
#     plt.colorbar()

    plt.subplot(1,7,7)
    plt.imshow(torch.flip(context_out[0,1,0,:,slice_,:].T.cpu(), dims=[-2]),cmap = 'gray')
    plt.axis('off')
    plt.title('Context 2 Mask')
#     plt.colorbar()

    plt.tight_layout()
    
import matplotlib.pyplot as plt
def plot_pred3(target_in, target_out, context_in, context_out, mask, slice_, task = 'seg'):
    plt.figure(figsize = (25,4))

    plt.subplot(1,7,1)
    plt.imshow(target_in[0,0,slice_,:,:].cpu(),cmap = 'gray')
    plt.axis('off')
    plt.title('Input')
#     plt.colorbar()

    plt.subplot(1,7,2)
    plt.imshow(target_out[0,0,slice_,:,:].cpu(),cmap = 'gray')
    plt.axis('off')
    plt.title('GT')
#     plt.colorbar()

    plt.subplot(1,7,3)
    if task == 'seg':
        plt.imshow(mask[0,0,slice_,:,:].cpu()>0.5,cmap = 'gray')
    elif task == 'gen':
        plt.imshow(mask[0,0,slice_,:,:].cpu(),cmap = 'gray')
    plt.axis('off')
    plt.title('Prediction')
#     plt.colorbar()

    plt.subplot(1,7,4)
    plt.imshow(context_in[0,0,0,slice_,:,:].cpu(),cmap = 'gray')
    plt.axis('off')
    plt.title('Context 1 Image')
#     plt.colorbar()

    plt.subplot(1,7,5)
    plt.imshow(context_out[0,0,0,slice_,:,:].cpu(),cmap = 'gray')
    plt.axis('off')
    plt.title('Context 1 Mask')
#     plt.colorbar()
    
    plt.subplot(1,7,6)
    plt.imshow(context_in[0,1,0,slice_,:,:,].cpu(),cmap = 'gray')
    plt.axis('off')
    plt.title('Context 2 Image')
#     plt.colorbar()

    plt.subplot(1,7,7)
    plt.imshow(context_out[0,1,0,slice_,:,:].cpu(),cmap = 'gray')
    plt.axis('off')
    plt.title('Context 2 Mask')
#     plt.colorbar()

    plt.tight_layout()