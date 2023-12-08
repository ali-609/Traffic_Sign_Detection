import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as F
import os
import glob
from PIL import Image
import re


class DatasetA2D2(Dataset): 
    def __init__(self, path):
       self.image_paths = path
       self.transforms = transforms.Compose([transforms.Resize((1208, 1920)),
                                             transforms.ToTensor()])
   
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path=self.image_paths[index]
        seg_path=img_path.replace('/camera/', '/multi_label/').replace('_camera_', '_label_')


        image = Image.open(img_path).convert('RGB') 
        image=self.transforms(image)        

        segmentation = Image.open(seg_path)
        segmentation = self.transforms(segmentation)        

        return {'image':image, 'segmentation': segmentation}
    
