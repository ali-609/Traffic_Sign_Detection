import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F
import glob
import torch
import torch.nn as nn
import cv2



from dataset import DatasetA2D2
from UNet import UNet


A2D2_path_all=sorted(glob.glob("./Dataset/camera_lidar_semantic/2018*/camera/cam_front_center/*.png"))
A2D2_dataset=DatasetA2D2(A2D2_path_all)


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

sample=A2D2_dataset[20]

imput=sample['image'].unsqueeze(0)
with torch.no_grad():
    output = model(imput)
    output = F.sigmoid(output)
    



reverse_transform = transforms.Compose([
    transforms.ToPILImage()])


output = (output > 0.05).float()


output=reverse_transform(output[0])

input_im=reverse_transform(sample['image'])
ground=reverse_transform(sample['segmentation'])


output.save('mask.png')
input_im.save('input.png')
ground.save('ground_truth.png')

create_boxes('input.png','mask.png','output.png')

