import os
import multiprocessing as mp
mp.set_start_method("fork", force=True)

os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_SOCKET_IFNAME"] = "eth0"
os.environ["NCCL_LAUNCH_MODE"] = "GROUP"
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

import torch, torch.nn as nn, torch.utils.data as data, torchvision as tv, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import torch
from torch.nn import functional as F
import os
import cv2
from tqdm import tqdm
import argparse
import random
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sys
import json
import warnings
warnings.filterwarnings("ignore")

from dataloader import MetaDataset_Multi, MetaDataset_Multi_Extended, MetaDatasetf_transform_1channel
from config import get_parser







args = get_parser().parse_args()
print(args)



data_loading_config = \
[{'name':'Dataset987_syn_test', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random'},'sample rate':0.25, 'weight': 1} for i in range(5)]+\
[{'name':'Dataset988_syn_test', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random'},'sample rate':0.25, 'weight': 1} for i in range(5)]+\
[{'name':'Dataset989_syn_test', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random'},'sample rate':0.25, 'weight': 1} for i in range(5)]+\
[{'name':'Dataset990_syn_test', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random'},'sample rate':0.25, 'weight': 1} for i in range(5)]+\
[{'name':'Dataset991_syn_test', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random'},'sample rate':0.25, 'weight': 1} for i in range(5)]+\
[{'name':'Dataset992_syn_test', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random'},'sample rate':0.25, 'weight': 1} for i in range(5)]+\
[{'name':'Dataset993_syn_test', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random'},'sample rate':0.25, 'weight': 1} for i in range(5)]+\
[{'name':'Dataset994_syn_test', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random'},'sample rate':0.25, 'weight': 1} for i in range(5)]+\
[{'name':'Dataset995_syn_test', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random'},'sample rate':0.25, 'weight': 1} for i in range(5)]+\
[{'name':'Dataset996_syn_test', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random'},'sample rate':0.25, 'weight': 1} for i in range(5)]+\
[{'name':'Dataset997_syn_test', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random'},'sample rate':0.25, 'weight': 1} for i in range(5)]+\
[{'name':'Dataset998_syn_test', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random'},'sample rate':0.25, 'weight': 1} for i in range(5)]+\
[{'name':'Dataset999_syn_test', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random'},'sample rate':0.25, 'weight': 1} for i in range(5)]+\
[{'name':'Dataset1000_syn_test', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random'},'sample rate':0.25, 'weight': 1} for i in range(5)]+\
[{'name':'Dataset1001_syn_test', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random'},'sample rate':0.25, 'weight': 1} for i in range(5)]+\
[{'name':'Dataset1002_syn_test', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random'},'sample rate':0.25, 'weight': 1} for i in range(5)]+\
[{'name':'Dataset1003_syn_test', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random'},'sample rate':0.25, 'weight': 1} for i in range(5)]+\
[{'name':'Dataset1004_syn_test', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random'},'sample rate':0.25, 'weight': 1} for i in range(5)]+\
[{'name':'Dataset1005_syn_test', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random'},'sample rate':0.25, 'weight': 1} for i in range(5)]+\
[{'name':'Dataset1006_syn_test', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random'},'sample rate':0.25, 'weight': 1} for i in range(5)]+\
[{'name':'Dataset1007_syn_test', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random'},'sample rate':0.25, 'weight': 1} for i in range(5)]+\
[{'name':'Dataset1008_syn_test', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random'},'sample rate':0.25, 'weight': 1} for i in range(5)]+\
[{'name':'Dataset1009_syn_test', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random'},'sample rate':0.25, 'weight': 1} for i in range(5)]+\
[{'name':'Dataset1010_syn_test', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random'},'sample rate':0.25, 'weight': 1} for i in range(5)]+\
[{'name':'Dataset1011_syn_test', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random'},'sample rate':0.25, 'weight': 1} for i in range(5)]+\
[{'name':'Dataset1012_syn_test', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random'},'sample rate':0.25, 'weight': 1} for i in range(5)]+\
[{'name':'Dataset1013_syn_test', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random'},'sample rate':0.25, 'weight': 1} for i in range(5)]+\
[{'name':'Dataset1014_syn_test', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random'},'sample rate':0.25, 'weight': 1} for i in range(5)]+\
[{'name':'Dataset1015_syn_test', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random'},'sample rate':0.25, 'weight': 1} for i in range(5)]+\
[{'name':'Dataset1016_syn_test', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random'},'sample rate':0.25, 'weight': 1} for i in range(5)]+\
[{'name':'Dataset1016_syn_test', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random'},'sample rate':0.25, 'weight': 1} for i in range(5)]+\
[{'name':'Dataset1017_syn_test', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random'},'sample rate':0.25, 'weight': 1} for i in range(5)]+\
[{'name':'Dataset1018_syn_test', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random'},'sample rate':0.25, 'weight': 1} for i in range(5)]+\
[{'name':'Dataset1019_syn_test', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random'},'sample rate':0.25, 'weight': 1} for i in range(5)]+\
[{'name':'Dataset1020_syn_test', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random'},'sample rate':0.25, 'weight': 1} for i in range(5)]+\
[{'name':'Dataset1021_syn_test', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random'},'sample rate':0.25, 'weight': 1} for i in range(5)]+\
[{'name':'Dataset1022_syn_test', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random'},'sample rate':0.25, 'weight': 1} for i in range(5)]+\
[{'name':'Dataset1023_syn_test', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random'},'sample rate':0.25, 'weight': 1} for i in range(5)]+\
[{'name':'Dataset1024_syn_test', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random'},'sample rate':0.25, 'weight': 1} for i in range(5)]+\
[{'name':'Dataset1025_syn_test', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random'},'sample rate':0.25, 'weight': 1} for i in range(5)]+\
[{'name':'Dataset1026_syn_test', 'input':i,'output':'Seg','task':'Seg','task_config':{'foreground_classes': 'random'},'sample rate':0.25, 'weight': 1} for i in range(5)]
 


transform = MetaDatasetf_transform_1channel(flip_prob=args.flip_prob,
                 sobel_prob = args.sobel_prob,)

dataset_train = MetaDataset_Multi_Extended(
        dataset_dir = args.data_dir, 
        skip_resize = args.skip_resize,
        data_loading_config = data_loading_config,
        group_size = args.context_size+1,
        transform = transform,
        random_context_size = args.random_context_size
        )

dataset_val = MetaDataset_Multi_Extended(
        dataset_dir = args.data_dir, 
        skip_resize = args.skip_resize,
        data_loading_config = data_loading_config,
        train_or_val = 'val',
        group_size = args.context_size+1,)

batchsize = args.batch_size

# train
dataloader_train = DataLoader(dataset_train, 
                              batch_size=batchsize,
                              num_workers=args.workers, 
                              shuffle = True,
                              pin_memory=False,
                              persistent_workers=False,
                              )

# val
dataloader_val = DataLoader(dataset_val, 
                            batch_size=batchsize,
                            num_workers=args.workers,
                            pin_memory=False,
                            persistent_workers=False,
                            )

'''
Start training

'''

# load model
import warnings
from pytorch_lightning.callbacks import LearningRateMonitor,ModelCheckpoint
warnings.filterwarnings('ignore')

model_module = f'from {args.model_name}.lightning_model import LightningModel'
print(model_module)
exec(model_module)

if args.checkpoint_path is None:
    model = LightningModel(args).to('cpu')
    print("hparams['nb_inner_channels']",model.hparams['nb_inner_channels'])
else:
    checkpoint_path = args.checkpoint_path
    tmp = sorted([i for i in os.listdir(checkpoint_path) if '.pth' in i or '.ckpt' in i])[args.checkpoint_index]
    checkpoint_path = os.path.join(checkpoint_path, tmp)
    print('load check points from:', checkpoint_path)
    model = LightningModel.load_from_checkpoint(checkpoint_path, map_location='cpu',strict = False)


total_params = sum(p.numel() for p in model.parameters())
print("Total number of parameters: ", total_params)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of trainable parameters: ", trainable_params)

# PL Callback 
lr_monitor = LearningRateMonitor(logging_interval='epoch')
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',   # Monitor validation loss
    save_top_k=1,  # Save the best model
    mode='min'   # Save when the loss is minimized
)
checkpoint_callback_latest = ModelCheckpoint(
    save_top_k=2,                # Keep only the latest 2 models
    monitor='epoch',
    mode='max', 
    every_n_epochs=1,            # Save once per epoch
    save_last=False ,
)

from pytorch_lightning.callbacks import Callback
import logging
import warnings
class SuppressWarningsCallback(Callback):
    def on_fit_start(self, trainer, pl_module):
        logging.getLogger('nibabel').setLevel(logging.ERROR)
        warnings.simplefilter("ignore")
    def on_validation_epoch_start(self, trainer, pl_module):
        logging.getLogger('nibabel').setLevel(logging.ERROR)
        warnings.simplefilter("ignore")

# Trainer
from pytorch_lightning.strategies import DDPStrategy
trainer = pl.Trainer(callbacks=[checkpoint_callback,
                                    checkpoint_callback_latest,
                                    lr_monitor,
                                    SuppressWarningsCallback()],
                         accelerator="gpu", 
                         devices=args.train_gpus, 
                         precision=args.precision, 
                         strategy=DDPStrategy(find_unused_parameters=False, process_group_backend="gloo"),
                         max_epochs = args.max_epochs,
                        )

trainer.fit(model, dataloader_train, dataloader_val)

