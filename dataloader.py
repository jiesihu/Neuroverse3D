import torch
from torch.utils.data import Dataset
import os
import nibabel as nib
import json
import numpy as np
import scipy.ndimage
import random
import time


# Data augmentation
import numpy as np
from scipy.ndimage import sobel
import numpy as np
from monai.transforms import (
    Compose,
    RandFlipd,
    RandRotate90d,
    RandSpatialCropd,
    RandShiftIntensityd,
    RandScaleIntensityd,
    RandGaussianNoised,
    EnsureTyped,
    Rand3DElasticd,
)

from scipy.ndimage import binary_dilation, binary_erosion


import warnings
warnings.simplefilter("ignore")

# dataset class
class MetaDataset_Multi(Dataset):
    def __init__(self, dataset_dir, 
                 data_loading_config=None, 
                 transform=None, 
                 resize_image=128, 
                 train_or_val = 'train', 
                 train_or_val_split_rate = 0.1,
                 data_split_seed = 0,
                ):
        """
        Load dataset of nnUNet format.
        Load multiple datasets at the same time.
        
        Args:
            image_dir (string): Path to the directory of dataset.
            data_loading_config (dict): Configuration of loading dataset including dataset ID, task, input, output, and sampling rate.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        assert data_loading_config is not None, 'Please input the data_loading_config'
        assert train_or_val=='train' or train_or_val=='val', 'Please input the correct train_or_val variable.'
        print('-'*10,f'Load the Meta dataset of {train_or_val} set.','-'*10)
        
        self.dataset_dir = dataset_dir
        self.data_loading_config = data_loading_config
        self.transform = transform
        self.resize_image = resize_image
        
        self.data_samples = []
        self.task_data_num = []
        self.task_sample_rate = []
        
        self.train_or_val = train_or_val
        self.train_or_val_split_rate = train_or_val_split_rate
        self.data_split_seed = data_split_seed
        
        for i in data_loading_config:
            data_samples_tmp = self.loaddata_from_config(dataset_dir,i)
            self.data_samples+=data_samples_tmp
            self.task_data_num.append(len(data_samples_tmp))
            self.task_sample_rate.append(i['sample rate'])
    
    def loaddata_from_config(self, dataset_dir,i):
        image_dir = os.path.join(dataset_dir,i['name'],'imagesTr')
#         samples = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f)) and '0000' in f]
        samples = [f for f in os.listdir(image_dir) if '0000' in f]
        samples = ['_'.join(f.split('_')[:-1]) for f in samples]
        samples = sorted(list(set(samples)))
        
        with open(os.path.join(dataset_dir,i['name'],'dataset.json'), 'r', encoding='utf-8') as file:
                config = json.load(file)
        data_samples_tmp = [{'sample': os.path.join(dataset_dir, i['name'], 'imagesTr', k) , 'input':i['input'], 'output':i['output'],\
          'task': i['task'],'task_config':i['task_config'],'sample rate': i['sample rate'], 'weight': i['weight'],
                             'config':config} for k in samples]
        
        # Split the train and val set by order.
        split_point = int(len(data_samples_tmp)-len(data_samples_tmp) * self.train_or_val_split_rate)
        if self.train_or_val=='train':
            data_samples_split = data_samples_tmp[:split_point]
        elif self.train_or_val=='val':
            data_samples_split = data_samples_tmp[split_point:]
            
        # keep part of the training data. (For Sensitivity analysis)
        if 'keep_rate' in i.keys() and 'keep_rate' is not None:
            keep_num = len(data_samples_tmp)*i['keep_rate']
            keep_num = round(keep_num)
            keep_num = keep_num if keep_num>=1 else 1
            data_samples_split = data_samples_split[:keep_num]
            
        print(f"Load {i['name']}\t{i['task']} sample rate: {i['sample rate']}", f"Total length is {len(samples)}\t",
             f"Length after spliting is {len(data_samples_split)}")
        
        return data_samples_split

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        sample = self.data_samples[idx]

        # Load images for the given input_
        image = self.load_data(sample,'input')

        # Load label
        label = self.load_data(sample, 'output')

        
        # crop
        image,crop_coords = self.crop_image(image)
        label,_ = self.crop_image(label,crop_coords)
        
        # padding
        image,padding_config =  self.pad_image(image, padding_config = None)
        label,_ =  self.pad_image(label, padding_config = padding_config)
        
        '''
        resize的时候第三个维度不变
        '''
        # resize
        image = self.resize_image_func(image, new_A=self.resize_image,order = 3)
        if sample['output']=='Seg':
            label = self.resize_image_func(label, new_A=self.resize_image,order = 0)
        else:
            label = self.resize_image_func(label, new_A=self.resize_image,order = 3)
        
        # adjust depth to 128*128*128
        image = self.adjust_image_depth(image)
        label = self.adjust_image_depth(label)
        
        # z-score again
        image = self.z_score_normalization(image)
        if sample['output']!='Seg':
            label = self.z_score_normalization(label)
        if sample['output']=='Seg':
            label = self.set_foreground_background(label,**sample['task_config'])
        
        # Task specific modified
        image, label = self.task_specific_transform( image, label, sample)
        
        # Apply transform (if any)
        if self.transform:
            image, label = self.transform(image, label)
            
        

        return {'image': image, 'label': label, 'name': idx}
    
    def task_specific_transform(self, image, label, sample):
        tasks = sample.get('task')
        task_configs = sample.get('task_config')

        # 如果任务不是列表，则转换为单元素列表
        if not isinstance(tasks, list):
            tasks = [tasks]
            task_configs = [task_configs]
 
        else:
            # 确保 task_configs 是列表
            if not isinstance(task_configs, list):
                raise ValueError("当 'task' 是列表时，'task_config' 也必须是列表。")

            if len(tasks) != len(task_configs):
                raise ValueError("任务列表的长度必须与任务配置列表的长度一致。")

        return image, label  # 确保返回修改后的 image 和 label
    
    
    def load_data(self,sample, key):
        # Load images for the given input_
        images = []
        label_dir = sample['sample'].replace('imagesTr','labelsTr')
        image_dir = sample['sample']
        config = sample['config']
        mod = sample[key]
        
        if mod == 'Seg':
            tmp = nib.load(os.path.join(f"{label_dir}{config['file_ending']}")).get_fdata()
        else:
            tmp = nib.load(os.path.join(f"{image_dir}_{str(mod).zfill(4)}{config['file_ending']}")).get_fdata()
            tmp = self.z_score_normalization(self.clip_percentiles(tmp))
        return tmp
    
    def resample_to_voxel_size(self, image_path, new_voxel_size=(1, 1, 1)):
        print('Warning! use resample_to_1mm1mm1mm function which is slow!!')
        # Load the image
        img = nib.load(image_path)
        img_data = img.get_fdata()
        original_affine = img.affine

        # Get the current voxel sizes from the affine matrix
        current_voxel_size = np.sqrt(np.sum(original_affine[:3, :3] ** 2, axis=0))

        # Calculate the zoom factors for each dimension
        zoom_factors = current_voxel_size / np.array(new_voxel_size)

        # Resample the image data
        resampled_data = zoom(img_data, zoom=zoom_factors, order=3)  # using cubic interpolation

        # Calculate the new affine matrix
        new_affine = original_affine.copy()
        new_affine[:3, :3] = np.diag(new_voxel_size)

        # Create a new NIfTI image
        resampled_img = nib.Nifti1Image(resampled_data, new_affine)

        return resampled_img
    
    # Data preprocessing
    def clip_percentiles(self, img):
        # 计算5%和95%的percentile
        p5 = np.percentile(img, 0.5)
        p95 = np.percentile(img, 99.5)

        # 使用clip函数将图像数值限制在这两个percentile之间
        img_clipped = np.clip(img, p5, p95)
        return img_clipped

    # 使用Z-Score归一化
    def z_score_normalization(self, img):
        mean = np.mean(img)
        std = np.std(img)

        img_normalized = (img - mean) / std
        return img_normalized
    
    def crop_image(self, image, crop_coords = None):
        """根据给定的阈值裁剪3D图像，移除周围的黑色背景区域。"""
        
        if crop_coords is None:
            threshold= np.percentile(image, 35)
            
            # 找到每个维度上非背景的索引
            coords = np.array(np.nonzero((image > threshold).astype('int')))
            x_min, x_max = coords[0].min(), coords[0].max()
            y_min, y_max = coords[1].min(), coords[1].max()
            z_min, z_max = coords[2].min(), coords[2].max()
            crop_coords = {'x_min':x_min,'x_max':x_max,'y_min':y_min,'y_max':y_max,'z_min':z_min, 'z_max':z_max}
        else: 
            x_min, x_max = crop_coords['x_min'],crop_coords['x_max']
            y_min, y_max = crop_coords['y_min'],crop_coords['y_max']
            z_min, z_max = crop_coords['z_min'],crop_coords['z_max']

        # 裁剪图像
        cropped_image = image[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
        return cropped_image, crop_coords
    
    def pad_image(self, image, pad_type='constant',padding_config = None):
        '''
        Make dim 0,1 equal
        '''
        pad_value = image.min()
        
        if padding_config is None:
            # Calculate the dimensions to pad to (make A and B dimensions equal)
            A, B, C = image.shape
            max_dim = max(A, B)

            # Calculate padding amounts
            pad_A = (max_dim - A) // 2
            pad_A_remainder = (max_dim - A) % 2
            pad_B = (max_dim - B) // 2
            pad_B_remainder = (max_dim - B) % 2

            # Create a padding configuration
            padding_config = (
                (pad_A, pad_A + pad_A_remainder),  # Padding for A dimension
                (pad_B, pad_B + pad_B_remainder),  # Padding for B dimension
                (0, 0)  # No padding for C dimension
            )
            

        # Pad the image
        padded_image = np.pad(image, padding_config, mode=pad_type, constant_values=pad_value)
        return padded_image, padding_config
    
    def resize_image_func(self, image, new_A=128,order = 3,nii_scale = [1,1,1], return_factor = False):
        '''
        等比例缩放三个维度
        '''
        # 获取原始尺寸
        A, _, C = image.shape

        # 计算缩放因子
        scale_A = new_A / A
        scale_C = scale_A  # 因为我们希望C维也按比例缩放

        # 使用scipy.ndimage.zoom来缩放图像
#         resized_image = scipy.ndimage.zoom(image, (scale_A, scale_A, scale_C), order=order)  # order=3表示三次样条插值
        resized_image = scipy.ndimage.zoom(image, (scale_A, scale_A, 1), order=order)  # order=3表示三次样条插值
        resized_image = scipy.ndimage.zoom(resized_image, (1, 1, scale_C), order=0)  # order=3表示三次样条插值

        return resized_image
    
    def expand_image_dim3(self, image, target_size=40):
        # Get the current size of the third dimension
        current_size = image.shape[2]

        # Check if the third dimension is less than the target size
        if current_size < target_size:
            # Calculate the zoom factor for the third dimension
            zoom_factor = target_size / current_size
            # Apply nearest-neighbor interpolation to expand the third dimension
            expanded_image = zoom(image, (1, 1, zoom_factor), order=0)
        else:
            expanded_image = image
        return expanded_image
    
    def adjust_image_depth(self, image):
        target_depth = self.resize_image
        current_depth = image.shape[2]
        min_value = image.min()

        if current_depth < target_depth:
            # 计算需要填充的层数
            pad_size = (target_depth - current_depth) // 2
            # 创建上下填充，如果是奇数，上方多填充一层
            pad_before = pad_size + (target_depth - current_depth) % 2
            pad_after = pad_size

            # 使用np.pad进行填充
            padding = ((0, 0), (0, 0), (pad_before, pad_after))
            padded_image = np.pad(image, padding, mode='constant', constant_values=min_value)
            return padded_image
        elif current_depth > target_depth:
            # 计算需要裁剪的起始和结束索引
            start = (current_depth - target_depth) // 2
            end = start + target_depth
            # 裁剪影像
            cropped_image = image[:, :, start:end]
            return cropped_image
        else:
            # 如果已经是128，则直接返回原图
            return image
        
    def set_foreground_background(self, mask, foreground_classes=None):
        """
        Set the specified classes as foreground (1) and others as background (0).

        Parameters:
        - mask: numpy array of shape (128, 128, 128) with integer values representing different classes
        - foreground_classes: list or set of integers representing the classes to set as foreground.
                              If None, randomly select one or more classes as foreground.

        Returns:
        - modified_mask: numpy array with the specified classes set as foreground (1) and others as background (0)
        """
        unique_classes = np.unique(mask)

        if foreground_classes is None:
            raise ValueError("You must set the foreground class!!!")
            # Randomly select one or more classes as foreground
            # num_classes_to_select = np.random.randint(1, len(unique_classes) + 1)
            # foreground_classes = set(np.random.choice(unique_classes, num_classes_to_select, replace=False))
        elif foreground_classes=='random':
            return mask
        else:
            foreground_classes = set(foreground_classes)

        # Create a new mask with the same shape
        modified_mask = np.zeros_like(mask, dtype=float)
        
        # Set foreground classes to 1
        mask = np.round(mask)
        for class_value in foreground_classes:
            modified_mask[mask == class_value] = 1.

        return modified_mask

class MetaDataset_Multi_Extended(MetaDataset_Multi):
    def __init__(self, dataset_dir, 
                 data_loading_config=None, 
                 transform=None, 
                 resize_image=128, 
                 train_or_val='train', 
                 train_or_val_split_rate=0.1,
                 data_split_seed=0,
                 group_size=3,
                 skip_resize = False,
                 cut_length = False,
                 random_context_size = False,
                 in_order = False,
                ):
        """
        Extend the MetaDataset_Multi class to include a group size variable.
        
        Args:
            dataset_dir (string): Path to the directory of dataset.
            data_loading_config (dict): Configuration of loading dataset including dataset ID, task, input, output, and sampling rate.
            transform (callable, optional): Optional transform to be applied on a sample.
            resize_image (int): Size to resize the images to.
            train_or_val (string): Specify whether the dataset is for training or validation.
            train_or_val_split_rate (float): Split rate for training and validation datasets.
            data_split_seed (int): Seed for random data splitting.
            group_size (int): Size of the support set size+1.
        """
        super().__init__(dataset_dir, 
                         data_loading_config, 
                         transform, 
                         resize_image, 
                         train_or_val, 
                         train_or_val_split_rate, 
                         data_split_seed,)
        
        self.group_size = group_size
        
        self.skip_resize = skip_resize
        if self.skip_resize:
            print('---Do skip_resize!---'*3)
            
        self.cut_length = cut_length # cut the validation length
        
        self.random_context_size = random_context_size
        if self.random_context_size:
            print('---Do random_context_size!---'*3)
        self.in_order = in_order
        
    def get_single_item(self, idx):
        sample = self.data_samples[idx]
        
        # Load images for the given input_
        image = self.load_data(sample,'input')

        # Load label
        label = self.load_data(sample, 'output')
        
        if self.skip_resize:
            pass
        else:
            # crop
            image,crop_coords = self.crop_image(image)
            label,_ = self.crop_image(label,crop_coords)

            # padding
            image,padding_config =  self.pad_image(image, padding_config = None)
            label,_ =  self.pad_image(label, padding_config = padding_config)

            # resize
            image = self.resize_image_func(image, new_A=self.resize_image,order = 3)
            if sample['output']=='Seg':
                label = self.resize_image_func(label, new_A=self.resize_image,order = 0)
            else:
                label = self.resize_image_func(label, new_A=self.resize_image,order = 3)

            # resize the third dimension if it is smaller than 40
            image = self.expand_image_dim3(image, target_size=40)
            label = self.expand_image_dim3(label, target_size=40)

            # adjust depth to 128*128*128
            image = self.adjust_image_depth(image)
            label = self.adjust_image_depth(label)
        
        # normalize again
        image = self.min_max_normalize_with_clipping(image, clip = False)
        if sample['output']!='Seg':
            label = self.min_max_normalize_with_clipping(label, clip = False)
        if sample['output']=='Seg':
            label = np.round(label)
        
        # Task specific modified
        image, label = self.task_specific_transform( image, label, sample)
        
        # rotate if it is from synthetic data
        
            
        return {'image': image, 'label': label, 'idx': idx, 'task': sample['task'],
                'dataset': sample['sample'].split('/')[-3], 'weight': sample['weight'],
               'input':sample['input'],'output':sample['output'],'sample':sample}
    
    def __len__(self):
        if self.train_or_val == 'val':
            len_ = 50
        elif self.train_or_val == 'train': 
            len_ = 1000
        return len_
    
    def min_max_normalize_with_clipping(self, image, min_percentile=0.1, max_percentile=99.9, clip = True):
        """
        Perform Min-Max normalization on an input image array with clipping to handle outliers.

        Parameters:
        image (numpy.ndarray): Input image array of shape (H, W) or (C, H, W).
        min_percentile (float): The lower percentile to clip the pixel values.
        max_percentile (float): The upper percentile to clip the pixel values.

        Returns:
        numpy.ndarray: Normalized image array with pixel values in the range [0, 1].
        """
        if clip:
            # Calculate the min and max values based on the given percentiles
            min_val = np.percentile(image, min_percentile)
            max_val = np.percentile(image, max_percentile)

            # Clip the image values to the calculated min and max
            clipped_image = np.clip(image, min_val, max_val)

            # Normalize the clipped image to [0, 1]
            normalized_image = (clipped_image - min_val) / (max_val - min_val + 1e-5)  # Add a small epsilon to avoid division by zero
        else:
            min_ = image.min()
            max_ = image.max()
            normalized_image = (image - min_) / (max_ - min_ + 1e-5)  # Add a small epsilon to avoid division by zero

        return normalized_image
    
    def __getitem__(self, idx):
        if len(self.task_data_num) != len(self.task_sample_rate):
            raise ValueError("List self.task_data_num and self.task_sample_rate must have the same length")
        
        self.task_idx = idx%len(self.task_data_num)

        # Return index of current task
        start_idx = sum(self.task_data_num[:self.task_idx])
        end_idx = sum(self.task_data_num[:self.task_idx+1])
        
        # add context_size randomness
        if self.random_context_size and random.random() < 0.5:
            group_size = random.choice([2, 3, 4, 5, 6, 7, 8, 9])
        else:
            group_size =  self.group_size
        

        self.all_idx = [random.randint(start_idx, end_idx-1) for _ in range(group_size)]
        
        # load the image one by one
        item = {'image': [], 'label': [], 'idx':[],'task': None,'samples':[]}
        for i in self.all_idx:
            tmp = self.get_single_item(i)
            item['image'].append(tmp['image'])
            item['label'].append(tmp['label'])
            item['idx'].append(tmp['idx'])
            item['task']=tmp['task']
            item['dataset']=tmp['dataset']
            item['weight']=tmp['weight']
            item['input']=tmp['input']
            item['output']=tmp['output']
            item['samples'].append(tmp['sample'])
            
        item['image'] = np.stack(item['image'], axis=0)
        item['label'] = np.stack(item['label'], axis=0)
        
        # load label setting of segmentation 
        if item['task']=='Seg':
            task_config = item['samples'][0]['task_config']
        elif 'Seg' in item['task']:
            task_config = item['samples'][0]['task_config'][item['task'].index('Seg')]
            
        if 'Seg' in item['task']:
            item['label'] = self.set_foreground_background(item['label'],
                                                   task_config,
                                                   config = item['samples'][0]['config'],)
            
        # Apply transform (if any)
        if self.transform:
            item = self.transform(item,transform_individual_rotate = '_syn_type' in item['dataset'])
        
        if not isinstance(item['label'], torch.Tensor):
            item['label'] = torch.tensor(item['label'])
        if not isinstance(item['image'], torch.Tensor):
            item['image'] = torch.tensor(item['image'])
        
        # Normalize the data
        item['label'] = self.normalize_3d_volume(item['label'])
        item['image'] = self.normalize_3d_volume(item['image'])
            
        return item
    
    def set_foreground_background(self, mask, task_config, config = None):
        """
        Set the specified classes as foreground (1) and others as background (0).

        Parameters:
        - mask: numpy array of shape (128, 128, 128) with integer values representing different classes
        - foreground_classes: list or set of integers representing the classes to set as foreground.
                              If None, randomly select one or more classes as foreground.

        Returns:
        - modified_mask: numpy array with the specified classes set as foreground (1) and others as background (0)
        """
        foreground_classes = task_config.get('foreground_classes', None)
        remove = task_config.get('remove', None)
        

        if foreground_classes is None:
            raise ValueError("You must set the foreground class!!!")
        elif foreground_classes=='random':
            class_all = [int(i) for i in list(config['labels'].keys())]
            # remove specific class
            if remove is not None:
                remove = [int(x) for x in remove]
                class_all = [x for x in class_all if x not in remove]
                
            max_length = len(class_all)-1 if len(class_all)-1 < 5 else 5
            length = random.randint(1, max_length)
            foreground_classes = random.sample(class_all, length)
            foreground_classes = set(foreground_classes)
        else:
            foreground_classes = set(foreground_classes)

        # Create a new mask with the same shape
        modified_mask = np.zeros_like(mask, dtype=float)
        
        # Set foreground classes to 1
        mask = np.round(mask)
        for class_value in foreground_classes:
            modified_mask[mask == class_value] = 1.
        return modified_mask
    
    def get_task_config(self, task_aug):
        sample_tmp = {'task':[],'task_config':[]}
        # shuffled the order of the task aug
        shuffled_task_aug = dict(random.sample(task_aug.items(), len(task_aug)))
        for task, prob in shuffled_task_aug.items():
            if np.random.rand() < prob:
                sample_tmp['task'].append(task)
                sample_tmp['task_config'].append({})
        return sample_tmp
    
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



class MetaDatasetf_transform_1channel(object):
    def __init__(self, 
                 flip_prob=0.05,
                 sobel_prob = 0.05,
                 dilation_errosion_prob = 0.05,
                 individual_rotate_prob = 0.25,
                 augmentation_seg = True,
                 augmentation_gen = True,
                 ):
        """
        Parameters:
        flip_prob (float): The probability of flipping the mask values.
        sobel_prob (float): The probability of applying sobel edge detection.
        augmentation_seg (bool): Whether apply augmentation for segmentation.
        augmentation_gen (bool): Whether apply augmentation for generation task.
        """
        
        self.flip_prob = flip_prob
        self.sobel_prob = sobel_prob
        self.dilation_errosion_prob = dilation_errosion_prob
        self.individual_rotate_prob = individual_rotate_prob
        
        
        # Build augmentation object
        self.transform_rotate = Compose([
                RandRotate90d(keys=['image', 'label'], prob=0.1, max_k=3, spatial_axes=(0,1)),
                RandRotate90d(keys=['image', 'label'], prob=0.1, max_k=3, spatial_axes=(1,2)),
                RandRotate90d(keys=['image', 'label'], prob=0.1, max_k=3, spatial_axes=(0,2)),
        ])
        
        self.transform_seg = Compose([
                RandFlipd(keys=['image', 'label'], prob=0.1, spatial_axis=0),
                RandFlipd(keys=['image', 'label'], prob=0.1, spatial_axis=1),
                RandFlipd(keys=['image', 'label'], prob=0.1, spatial_axis=2),
                RandRotate90d(keys=['image', 'label'], prob=0.1, max_k=3, spatial_axes=(0,1)),
                RandRotate90d(keys=['image', 'label'], prob=0.1, max_k=3, spatial_axes=(1,2)),
                RandRotate90d(keys=['image', 'label'], prob=0.1, max_k=3, spatial_axes=(0,2)),
                RandomResample(keys=["image", "label"], prob=0.15),
                RandShiftIntensityd(keys='image', prob=0.2, offsets=0.2),
                RandScaleIntensityd(keys='image', prob=0.2, factors=0.1),
                RandGaussianNoised(keys='image', prob=0.1, mean=0.0, std=0.02),
                Rand3DElasticd(
                        keys=['image', 'label'],
                        prob=0.05,
                        sigma_range=(7, 8),
                        magnitude_range=(100, 200),
                        translate_range=(0, 0, 0),
                        rotate_range=(0.3, 0.3,0.3),
                        scale_range=(0.2, 0.2, 0.2),
                        mode=('bilinear', 'nearest')  # Specify different modes for image and label
                    ),
                EnsureTyped(keys=['image', 'label'])
            ])
        self.augmentation_seg = augmentation_seg
        
        self.transform_gen = Compose([
                RandFlipd(keys=['image', 'label'], prob=0.1, spatial_axis=0),
                RandFlipd(keys=['image', 'label'], prob=0.1, spatial_axis=1),
                RandFlipd(keys=['image', 'label'], prob=0.1, spatial_axis=2),
                RandRotate90d(keys=['image', 'label'], prob=0.1, max_k=3, spatial_axes=(0,1)),
                RandRotate90d(keys=['image', 'label'], prob=0.1, max_k=3, spatial_axes=(1,2)),
                RandRotate90d(keys=['image', 'label'], prob=0.1, max_k=3, spatial_axes=(0,2)),
                RandomResample(keys=["image", "label"], prob=0.15),
                RandShiftIntensityd(keys=['image', 'label'], prob=0.2, offsets=0.2),
                RandScaleIntensityd(keys=['image', 'label'], prob=0.2, factors=0.1),
                RandGaussianNoised(keys='image', prob=0.1, mean=0.0, std=0.025),
                RandGaussianNoised(keys='image', prob=0.05, mean=0.0, std=0.05),
                RandGaussianNoised(keys='image', prob=0.025, mean=0.0, std=0.075),
                Rand3DElasticd(
                        keys=['image', 'label'],
                        prob=0.05,
                        sigma_range=(7, 8),
                        magnitude_range=(100, 200),
                        translate_range=(0, 0, 0),
                        rotate_range=(0.3, 0.3,0.3),
                        scale_range=(0.1, 0.1, 0.1),
                        mode=('bilinear', 'bilinear')  # Specify different modes for image and label
                    ),
                EnsureTyped(keys=['image', 'label'])
            ])
        self.augmentation_gen = augmentation_gen

    
    def sobel_edge_detection_3d(self, image):
        # 初始化梯度数组
        sobel_edges = np.zeros_like(image)

        # 对每个方向应用Sobel滤波器
        for axis in range(3):
            sobel_edges += np.abs(sobel(image, axis=axis))/3

        return sobel_edges
    def sobel_edge_detection_batch(self, mask):
        for i in range(mask.shape[0]):
            mask[i,:] = self.sobel_edge_detection_3d(mask[i,:])
        mask = mask/mask.max()
        mask = (mask>0.3).astype(float)
        return mask
    
    def dilate_mask(self, mask, structure_size=3, iterations=1):
        """
        对单个3D二值掩码进行形态学膨胀。
        """
        structure = np.ones((structure_size, structure_size, structure_size), dtype=bool)
        dilated = binary_dilation(mask, structure=structure, iterations=iterations)
        return dilated.astype(mask.dtype)

    def dilate_batch_masks(self, batch_mask, structure_size=3, iterations=1):
        """
        对批量的3D二值掩码进行形态学膨胀。
        参数:
        - batch_mask: 形状为 (batch_size, D, H, W) 的4D NumPy数组。
        """
        dilated_masks = [self.dilate_mask(mask, structure_size, iterations) for mask in batch_mask]
        return np.stack(dilated_masks, axis=0)
    
    def shrink_mask(self, mask, structure_size=3, iterations=1):
        """
        对单个3D二值掩码进行形态学收缩。
        """
        structure = np.ones((structure_size, structure_size, structure_size), dtype=bool)
        shrunk = binary_erosion(mask, structure=structure, iterations=iterations)
        return shrunk.astype(mask.dtype)

    def shrink_batch_masks(self, batch_mask, structure_size=3, iterations=1):
        """
        对批量的3D二值掩码进行形态学收缩。
        参数:
        - batch_mask: 形状为 (batch_size, D, H, W) 的4D NumPy数组。
        """
        shrunk_masks = [self.shrink_mask(mask, structure_size, iterations) for mask in batch_mask]
        return np.stack(shrunk_masks, axis=0)
    
    def process_item_individually(self, item, transform_fn):
        """
        将item中的'label'和'image'拆分为(1, 128, 128, 128)的块，
        分别应用transform_fn处理后重新组合。

        参数：
        - item: 包含 'label' 和 'image' 的字典，每个形状为 (4, 128, 128, 128)
        - transform_fn: 用于处理每个 (1, 128, 128, 128) 子数组的函数

        返回：
        - 重新组合的item，结构与输入的item相同
        """
        # 初始化输出结构，保持与输入一致
        new_item = {'label': np.zeros_like(item['label']), 'image': np.zeros_like(item['image'])}

        # 遍历4个样本，并分别应用transform_fn处理
        for i in range(item['label'].shape[0]):
            # 取出 (1, 128, 128, 128) 的子数组，并扩展维度
            label_temp = np.expand_dims(item['label'][i], axis=0)  # (1, 128, 128, 128)
            image_temp = np.expand_dims(item['image'][i], axis=0)  # (1, 128, 128, 128)
            item_temp = {'label': label_temp, 'image': image_temp}
            # 应用transform_fn进行变换
            item_temp = transform_fn(item_temp)

            # 将处理后的结果放回新的item中
            new_item['label'][i] = np.squeeze(item_temp['label'])  # 去掉多余维度
            new_item['image'][i] = np.squeeze(item_temp['image'])  # 去掉多余维度

        return new_item
    
    def __call__(self, item, transform_individual_rotate = False):
        '''
        Item includes keys: image, label, idx, task.
        Transforms for ICL segmentation task. All transforms are on item['label'].
        Shape of item['label'] is: (B,H,W,D)

        '''
#         print("item['label']:",item['label'].shape,type(item['label']))
        if item['task'] == 'Seg' or 'Seg' in item['task']:
            # flip
            if np.random.rand() < self.flip_prob:
                item['label'] = 1-item['label']
            if np.random.rand() < self.flip_prob:
                item['image'] = 1-item['image']
                
            # edge detection
            if np.random.rand() < self.sobel_prob:
                item['label'] = self.sobel_edge_detection_batch(item['label'])
            
            if np.random.rand() < self.dilation_errosion_prob:
                # Inflation
                item['label'] = self.dilate_batch_masks(item['label'],iterations=np.random.randint(1,2))
            elif np.random.rand() < self.dilation_errosion_prob:
                # Shrinkage
                item['label'] = self.shrink_batch_masks(item['label'],iterations=1)
            
            # Augmentation
            if self.augmentation_seg:
                item = self.transform_seg(item)
                
            if transform_individual_rotate and (np.random.rand() < self.individual_rotate_prob):
                new_item = self.process_item_individually(item, self.transform_rotate)
                item['label'],item['image'] = new_item['label'],new_item['image']
                
        elif item['task'] in ['ModTran','Bias','Denoi','2D23D','Inp','SupRes'] or isinstance(item['task'], list):
            if np.random.rand() < self.flip_prob:
                item['label'] = 1-item['label']
            if np.random.rand() < self.flip_prob:
                item['image'] = 1-item['image']
                
            # Augmentation
            if self.augmentation_gen:
                item = self.transform_gen(item)
#         print("item['label']:",item['label'].shape,type(item['label']))
        return item


import numpy as np
import torch
import torch.nn.functional as F
from typing import Mapping, Hashable, Sequence, Union, Optional

from monai.transforms import MapTransform, InvertibleTransform, LazyTransform


class RandomResample(MapTransform, InvertibleTransform, LazyTransform):
    """
    随机在 3D 体数据 (B, H, W, D) 的某一个空间维上做下采样 / 还原的可逆字典式变换。
    """

    backend = ["numpy", "torch"]

    def __init__(
        self,
        keys: Sequence[str] = ("image", "label"),
        prob: float = 0.05,
        allow_missing_keys: bool = False,
        lazy: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)

        object.__setattr__(self, "_lazy_evaluation", bool(lazy))
        # ------------------------------------------------------

        self.prob = float(prob)
        self.rng = np.random.default_rng()

    # ----------------- internal helpers ----------------- #
    @staticmethod
    def _downsample(x, dim: int, factor: int):
        slicer = [slice(None)] * x.ndim
        slicer[dim] = slice(None, None, factor)
        return x[tuple(slicer)]

    @staticmethod
    def _pad_to_size(x, dim: int, target: int):
        size = x.shape[dim]
        if size >= target:
            return x
        pad_total = target - size
        pad_pre = pad_total // 2
        pad_post = pad_total - pad_pre

        if isinstance(x, torch.Tensor):
            # (D2, D1, W2, W1, H2, H1) 的顺序
            pad_cfg = [0, 0, 0, 0, 0, 0]
            if dim == 3:          # D
                pad_cfg[0:2] = [pad_post, pad_pre]
            elif dim == 2:        # W
                pad_cfg[2:4] = [pad_post, pad_pre]
            elif dim == 1:        # H
                pad_cfg[4:6] = [pad_post, pad_pre]
            return F.pad(x, pad_cfg, mode="constant", value=x.min().item())

        pad_width = [(0, 0)] * x.ndim
        pad_width[dim] = (pad_pre, pad_post)
        return np.pad(x, pad_width, mode="constant", constant_values = x.min())

    @staticmethod
    def _resize_nn(x, dim: int, target: int):
        if x.shape[dim] == target:
            return x
        rep = target // x.shape[dim] + 1
        if isinstance(x, torch.Tensor):
            x = x.repeat_interleave(rep, dim=dim)
        else:
            x = np.repeat(x, rep, axis=dim)
        slicer = [slice(None)] * x.ndim
        slicer[dim] = slice(0, target)
        return x[tuple(slicer)]

    # ------------------- forward ------------------- #
    def __call__(
        self,
        data: Mapping[Hashable, Union[torch.Tensor, np.ndarray]],
        lazy: Optional[bool] = None,
    ):
        # 1) 是否执行
        if np.random.rand() >= self.prob:
            return data
        
        d = dict(data)

        # 2) 本批次共享参数
        axis = int(self.rng.choice([1, 2, 3]))      # H / W / D
        factor = int(self.rng.choice([2, 3, 4]))    # 2, 3, 4
        use_padding = bool(self.rng.random() < 0.5) # True=pad, False=resize

        for key in self.key_iterator(d):
            x = d[key]
            if x.ndim != 4:
                raise ValueError(
                    f"{self.__class__.__name__}: 期望 4-D (B,H,W,D)，收到 {x.shape}"
                )
            orig_len = x.shape[axis]
            x_ds = self._downsample(x, axis, factor)
            x_out = (
                self._pad_to_size(x_ds, axis, orig_len)
                if use_padding
                else self._resize_nn(x_ds, axis, orig_len)
            )
            d[key] = x_out

        return d

    # ------------------- inverse ------------------- #
    def inverse(self, data):
        raise NotImplementedError(
            "如需可逆操作，请在 __call__ 缓存 axis/factor/use_padding 后在此处实现。"
        )







