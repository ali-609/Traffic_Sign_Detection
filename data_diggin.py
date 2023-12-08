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
from multiprocessing import Pool
import cv2
import numpy as np
import os

A2D2_path_all_seg=sorted(glob.glob("Dataset/camera_lidar_semantic/2018*/label/cam_front_center/*.png"))



colors=np.array([
# Traffic Sign Colors
[0, 255, 255],  # Traffic sign 1
[30, 220, 220],  # Traffic sign 2   #21
[60, 157, 199],  # Traffic sign 3   #22
])



def process_image(path):
    img = cv2.imread(path)
    out_path = path.replace('/label/', '/multi_label/')  # .replace('.png', '.npy')

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = np.zeros((img.shape[0], img.shape[1],1), dtype=np.uint8)

    mask = np.all(img[:, :, None] ==[0, 255, 255], axis=-1)

    out[mask] = 255

    mask = np.all(img[:, :, None] ==[30, 220, 220], axis=-1)

    out[mask] = 255


    mask = np.all(img[:, :, None] ==[60, 157, 199], axis=-1)

    out[mask] = 255


    out_dir = os.path.dirname(out_path)
    if not os.path.exists(out_dir):
        try:
            os.makedirs(out_dir)
        except FileExistsError:
            pass

    cv2.imwrite(out_path, out)
    print(out.shape,"__",out_path, " : Success")



if __name__ == '__main__':
    with Pool(processes=48) as pool:  
        pool.map(process_image, A2D2_path_all_seg)

