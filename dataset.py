import sys
from pathlib import Path

import cv2
import numpy as np
# import pandas as pd
import torch
from torch.nn import L1Loss, MSELoss, HuberLoss
from torch.utils.data import ConcatDataset, RandomSampler, WeightedRandomSampler

import json
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as F
import os
# from skimage.util import random_noise
import glob
# from scipy.spatial.distance import cdist
from PIL import Image
import re


class DatasetA2D2(Dataset): #21 steering angle and segmentation
    def __init__(self, path):
       self.image_paths = path
       self.transforms = transforms.Compose([transforms.Resize((1208, 1920)),
                                             transforms.ToTensor()])
       self.transforms_seg = transforms.Compose([transforms.ToTensor()
                                            ])
       


    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path=self.image_paths[index]
        seg_path=img_path.replace('/camera/', '/multi_label/').replace('_camera_', '_label_')#.replace('.png','.npy')


        image = Image.open(img_path).convert('RGB') 

        image=self.transforms(image)
        

        
        print(img_path)
        segmentation = Image.open(seg_path)
        segmentation = self.transforms(segmentation)        

        return {'image':image, 'segmentation': segmentation}
    
