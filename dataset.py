import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image



class DatasetA2D2(Dataset): 
    def __init__(self, path):
       self.image_paths = path
       self.transforms = transforms.Compose([transforms.Resize((224, 224)),
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
    
