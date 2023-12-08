import sys
from pathlib import Path
from tqdm import tqdm

import numpy as np
# import pandas as pd
import torch
from torch.nn import L1Loss, MSELoss, HuberLoss
from torch.utils.data import ConcatDataset, RandomSampler, WeightedRandomSampler

from torch.utils.data import Dataset, random_split, DataLoader,ConcatDataset
import torchvision
from torchvision import transforms
# import torchvision.transforms.functional as F
import torch.nn.functional as F
import torchvision.models as models
# from skimage.util import random_noise
import glob
from dataset import DatasetA2D2

import random
import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models import resnet34
import cv2
# from models.first_hydra import HydraNet
from UNet import UNet
# import wandb

# A2D2_path_all=A2D2_path_all=sorted(glob.glob("/gpfs/space/home/alimahar/hydra/Datasets/A2D2/semantic/camera_lidar_semantic/2018*/camera/cam_front_center/*.png"))
A2D2_path_all=sorted(glob.glob("./Dataset/camera_lidar_semantic/2018*/camera/cam_front_center/*.png"))
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2
# exit()
A2D2_path_train=A2D2_path_all[:int(len(A2D2_path_all) * TRAIN_SPLIT)]
A2D2_path_val=A2D2_path_all[-int(len(A2D2_path_all) * VAL_SPLIT):]


A2D2_dataset_train=DatasetA2D2(A2D2_path_train)
A2D2_dataset_val=DatasetA2D2(A2D2_path_val)

def create_boxes(path_to_org_image,path_to_label, output):
  im = cv2.imread(path_to_label)
  imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
  ret,thresh = cv2.threshold(imgray,127,255,0)
  contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
  tmp = np.zeros_like(im)
  boundary = cv2.drawContours(tmp, contours, -1, (40,255,0), 2)
  image2 = cv2.imread(path_to_org_image)
  mask = np.where(boundary == 0, 1, 0)
  merged_image = boundary + (mask * image2)
  cv2.imwrite(output, merged_image)



model_path = 'seg_only_overfit_lr3.pth'  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model= UNet()


model.load_state_dict(torch.load(model_path))
model.eval()
imput_sample=A2D2_dataset_train[20]

imput=imput_sample['image'].unsqueeze(0)
with torch.no_grad():
    output = model(imput)
    output = F.sigmoid(output)
    

print(np.array(output).max())

reverse_transform = transforms.Compose([
    transforms.ToPILImage()])


output = (output > 0.05).float()


output=reverse_transform(output[0])

input_im=reverse_transform(imput_sample['image'])
ground=reverse_transform(imput_sample['segmentation'])

segmentation_loss =  nn.BCELoss()


output.save('threshold_2.png')
input_im.save('input.png')
ground.save('gt.png')

create_boxes('input.png','threshold_2.png','output.png')