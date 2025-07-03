"""
This script allows you to directly run Neuroverse3D on a given 3D medical imaging path.

It will automatically perform image resizing and normalization to make the input data compatible with Neuroverse3D.

The script will also automatically adjust the group size based on the available GPU to fully utilize computational resources.

The context limit is set to 8 by default to ensure computational efficiency. You can modify this limit in the read_context function if needed.
"""

import argparse
import torch
import numpy as np
import time
import os
import nibabel as nib
from tqdm import tqdm
import torch.nn.functional as F
import glob

from neuroverse3D.lightning_model import LightningModel
from utils.dataloading import *

class VolumeResizer:
    def __init__(self, ROI=(128, 128, 128)):
        self.ROI = ROI  # Target size: (H, W, D)
        self.original_size = None
        self.scale_factors = None

    def resize(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Resize a 3D volume to self.ROI using nearest neighbor interpolation.
        :param volume: torch.Tensor of shape (B, C, H, W, D)
        :return: Resized torch.Tensor of shape self.ROI
        """
        if volume.ndim != 5:
            raise ValueError("Input volume must be 3D (B, C, H, W, D)")
        
        self.original_size = volume.shape[2:]  # Save original size
        self.scale_factors = [roi / orig for roi, orig in zip(self.ROI, self.original_size)]
        
        # Reshape to (1,1,H,W,D) for interpolation
#         volume = volume.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        resized = F.interpolate(volume, size=self.ROI, mode='nearest')
        return resized

    def recover(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Recover a resized volume back to original size using trilinear interpolation.
        :param volume: torch.Tensor of shape self.ROI
        :return: Recovered torch.Tensor of original size
        """
        if self.original_size is None:
            raise RuntimeError("Must call resize() before recover()")

        if volume.shape[2:] != self.ROI:
            raise ValueError(f"Input volume must be of shape {self.ROI}")
        
        # Reshape to (1,1,H,W,D) for interpolation
        volume = volume
        recovered = F.interpolate(volume, size=self.original_size, mode='trilinear', align_corners=True)
        return recovered

    


def read_context(context_imgs, context_imgs_modality='0000', context_labs=None, context_labs_modality = None, context_size_upper_bound = 8):
    # 仅保留 .nii 和 .nii.gz 文件
    def get_nii_files(path):
        return sorted([
            f for f in os.listdir(path)
            if f.endswith('.nii') or f.endswith('.nii.gz')
        ])

    # 获取图像文件，筛选包含指定模态的文件
    img_files = get_nii_files(context_imgs)
    img_files = [f for f in img_files if context_imgs_modality + '.' in f]

    if context_labs is not None and context_labs_modality is None:
        lab_files = get_nii_files(context_labs)
        
        

        # 校验文件数量一致
        if len(img_files) != len(lab_files):
            raise ValueError(f"Number of images ({len(img_files)}) and labels ({len(lab_files)}) do not match")

        # 检查图像文件名经过模态名替换后是否和标签文件一致
        expected_lab_files = [f.replace('_'+context_imgs_modality + '.', '.') for f in img_files]
        
        if sorted(expected_lab_files) != sorted(lab_files):
            raise ValueError("Image and label filenames do not match after removing modality identifier.")

        # 返回排序后的完整路径
        img_paths = sorted([os.path.join(context_imgs, f) for f in img_files])
        lab_paths = sorted([os.path.join(context_labs, f) for f in expected_lab_files])
        
        if len(img_paths)>context_size_upper_bound:
            print(f"There are {len(img_paths)} files in the context path, which is larger than the upper bound of {context_size_upper_bound}, so we only keep {context_size_upper_bound} contexts.")
            img_paths = img_paths[:context_size_upper_bound]
            lab_paths = lab_paths[:context_size_upper_bound]
        return img_paths, lab_paths
    
    elif context_labs is not None and context_labs_modality is not None:
        lab_files = get_nii_files(context_labs)
        lab_files = [f for f in lab_files if context_labs_modality + '.' in f]
        
        
        # 校验文件数量一致
        if len(img_files) != len(lab_files):
            raise ValueError(f"Number of images ({len(img_files)}) and labels ({len(lab_files)}) do not match")

        # 检查图像文件名经过模态名替换后是否和标签文件一致
        expected_lab_files = [f.replace('_'+context_imgs_modality + '.', '_'+context_labs_modality+'.') for f in img_files]
        
        if sorted(expected_lab_files) != sorted(lab_files):
            raise ValueError("Image and label filenames do not match after removing modality identifier.")

        # 返回排序后的完整路径
        img_paths = sorted([os.path.join(context_imgs, f) for f in img_files])
        lab_paths = sorted([os.path.join(context_labs, f) for f in expected_lab_files])
        
        if len(img_paths)>context_size_upper_bound:
            print(f"There are {len(img_paths)} files in the context path, which is larger than the upper bound of {context_size_upper_bound}, so we only keep {context_size_upper_bound} contexts.")
            img_paths = img_paths[:context_size_upper_bound]
            lab_paths = lab_paths[:context_size_upper_bound]
        return img_paths, lab_paths
    
    else:
        # 仅返回图像路径
        img_paths = sorted([os.path.join(context_imgs, f) for f in img_files])
        return img_paths



def load_nii_list_to_tensor(file_list):
    """
    读取一个 NIfTI 文件路径列表，并将其组合为一个 (B, 1, W, H, D) 的 torch.Tensor。

    :param file_list: List[str] - 包含 .nii 或 .nii.gz 文件路径的列表
    :return: torch.Tensor -  (B, 1, W, H, D)
    """
    volumes = []
    for path in file_list:
        img = nib.load(path)
        data = img.get_fdata()  # 返回 float64 的 numpy array，shape: (W, H, D)
        
        if data.ndim != 3:
            raise ValueError(f"Volume at {path} is not 3D, got shape {data.shape}")
        
        data = np.expand_dims(data, axis=0)  # -> (1, W, H, D)
        volumes.append(data)
    
    # Stack into shape: (B, 1, W, H, D)
    volume_array = np.stack(volumes, axis=0)
    tensor = torch.from_numpy(volume_array).float()
    return tensor

def run_model_according_to_task(model, img_input, context_in, context_out, task, group_size = 2):
        if task =='Seg':
            task_note = '''
            Run the segmentation tasks. Inference for each class.
            '''            
            print(task_note)

            # Ensure all values are integers (allow float type but must have integer values)
            if not torch.all((context_out == context_out.int())):
                raise ValueError("context_out contains non-integer values.")

            # Extract class list (convert to sorted Python list)
            class_list = sorted(torch.unique(context_out).int().tolist())

            if len(class_list) > 50:
                print("Warning: class_list contains more than 50 unique classes.")

            # Initialize final mask (same shape as context_out)
            mask_class = torch.zeros_like(img_input, dtype=torch.float32)

            for cls in tqdm(class_list, desc="Processing classes"):
                # Create a binary 0/1 mask
                context_out_class = (context_out == cls).float()

                # Normalization
                img_input_norm = normalize_3d_volume(img_input)
                context_in_norm = normalize_3d_volume(context_in)
                context_out_class_norm = normalize_3d_volume(context_out_class)

                # Inference
                with torch.no_grad():
                    model.eval()
                    pred = model.forward(img_input_norm, context_in_norm, context_out_class_norm, gs=group_size)

                # Assign current class ID to regions where prediction > 0.5
                mask_class[pred > 0.5] = cls
            return mask_class
        else:
            task_note = '''
            Run the Generation tasks.
            '''
            print(task_note)
            # Normalization
            img_input = normalize_3d_volume(img_input)
            context_in = normalize_3d_volume(context_in)
            context_out = normalize_3d_volume(context_out)

            # 推理
            with torch.no_grad():
                model.eval()
                mask = model.forward(img_input, context_in, context_out, gs = group_size)
            return mask


def get_args():
    parser = argparse.ArgumentParser(description="Inference Configuration")

    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run the model on (e.g., "cuda:0", "cpu")')

    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint/neuroverse3D.ckpt',
                        help='Path to the model checkpoint')

    parser.add_argument('--context_imgs', type=str, default='Demo_data/seg/imgs',
                        help='Directory for context image files')

    parser.add_argument('--context_imgs_modality', type=str, default='0000',
                        help='Modality string for context images')

    parser.add_argument('--context_labs', type=str, default='Demo_data/seg/labs',
                        help='Directory for context label files')
    
    parser.add_argument('--context_labs_modality', type=str, default=None,
                        help='Modality string for context labels')

    parser.add_argument('--target_imgs', type=str, default=None,
                        help='Directory for target image files')

    parser.add_argument('--target_modality', type=str, default='0000',
                        help='Modality string for target images')

    parser.add_argument('--target_output_path', type=str, default=None,
                        help='Directory to save output predictions')

    parser.add_argument('--task', type=str, default='Seg', choices=['Seg', 'Gen'],
                        help='Task type: Seg for segmentation, Gen for generaztion tasks')

    return parser.parse_args()


args = get_args()

device = args.device
checkpoint_path = args.checkpoint_path
context_imgs, context_imgs_modality = args.context_imgs, args.context_imgs_modality
context_labs, context_labs_modality = args.context_labs, args.context_labs_modality
target_imgs, target_modality = args.target_imgs, args.target_modality
target_output_path = args.target_output_path
task = args.task

print("Device:", device)
print("Target Image Path:", target_imgs)


# Load the Checkpoint
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
hparams = checkpoint['hyper_parameters']

# load model
import warnings
warnings.filterwarnings('ignore')
model = LightningModel.load_from_checkpoint(checkpoint_path, map_location=torch.device(device))
print('Load checkpoint from:', checkpoint_path)


# Determine the group size
props = torch.cuda.get_device_properties(device)
total_memory_GB = props.total_memory / (1024 ** 3)  # GB
print(f"Device: {device}, Total Memory: {total_memory_GB:.2f} GB")

memory_table = torch.tensor([9,11,14,16,18,22,25,28])
group_size = (total_memory_GB>memory_table).sum().item()
group_size = 1 if group_size<1 else group_size
print('Group size of context:', group_size)


# Find the context files
context_imgs_list, context_labs_list = read_context(context_imgs, context_imgs_modality, context_labs, context_labs_modality)
# Load the context set
context_set_imgs = load_nii_list_to_tensor(context_imgs_list)
context_set_labs = load_nii_list_to_tensor(context_labs_list)

# Build the resizer
volumeresizer = VolumeResizer(ROI=(128, 128, 128))

# resize into 128*128*128
context_set_imgs = volumeresizer.resize(context_set_imgs)
context_set_labs = volumeresizer.resize(context_set_labs)

# Reshape for input
context_in = context_set_imgs[None, :]
context_out = context_set_labs[None, :]


# Find the target files
target_imgs_list = read_context(target_imgs, target_modality, None)

for target_path in target_imgs_list:
    # 读取图像
    nii = nib.load(target_path)
    data = nii.get_fdata()  # shape: (W, H, D)

    if data.ndim != 3:
        raise ValueError(f"Input NIfTI must be 3D. Got shape: {data.shape}")

    # 转换为 (1, 1, W, H, D)
    img_input = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).float()  # torch.Tensor

    # Resize
    img_input = volumeresizer.resize(img_input)
    
    mask = run_model_according_to_task(model,
                                       img_input,
                                       context_in,
                                       context_out,
                                       task = task,
                                       group_size = group_size
                                      )
    # Resize back
    img_input = volumeresizer.recover(img_input)

    # 后处理：移除 batch 和 channel，转换为 numpy
    mask_np = mask.squeeze(0).squeeze(0).cpu().numpy()

    # 保持原始 affine 和 header
    mask_nii = nib.Nifti1Image(mask_np, affine=nii.affine, header=nii.header)

    # 构建输出路径
    output_path = target_path.replace(target_imgs, target_output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 保存预测掩码
    nib.save(mask_nii, output_path)


