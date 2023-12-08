import sys
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import glob
import torch
import torch.nn as nn

from dataset import DatasetA2D2
from UNet import UNet
torch.manual_seed(0)

import wandb
wandb.init(project="local")


BATCH_SIZE = 3

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2

A2D2_path_all=sorted(glob.glob("./Dataset/camera_lidar_semantic/2018*/camera/cam_front_center/*.png"))

A2D2_path_train=A2D2_path_all[:int(len(A2D2_path_all) * TRAIN_SPLIT)]
A2D2_path_val=A2D2_path_all[-int(len(A2D2_path_all) * VAL_SPLIT):]


A2D2_dataset_train=DatasetA2D2(A2D2_path_train)
A2D2_dataset_val=DatasetA2D2(A2D2_path_val)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Used calculation device: ",device)


print('No of train samples', len(A2D2_dataset_train)*BATCH_SIZE )
print('No of validation Samples', len(A2D2_dataset_val)*BATCH_SIZE)


train_dataloader = DataLoader(A2D2_dataset_train, batch_size=BATCH_SIZE, shuffle=False,num_workers=6)
val_dataloader = DataLoader(A2D2_dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=6)


model = UNet()
model.to(device=device)
loss =  nn.BCEWithLogitsLoss()

lr = 1e-5
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

n_epochs=5

best_val_loss=999999

wandb.watch(model)
for epoch in range(n_epochs):
    
    model.train()

    total_training_loss = 0

    for i, data in enumerate( tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{n_epochs}')):
     
        #Loading input and labesls to device
        inputs = data["image"].to(device=device) 
        segmentation_label = data["segmentation"].to(device=device)


        #Output
        optimizer.zero_grad()
        segmentation_output = model(inputs)

        # Loss calculation
        segmentation_loss = loss(segmentation_output, segmentation_label)

        #Backward
        segmentation_loss.backward()
        optimizer.step()

        #Logging        
        total_training_loss += segmentation_loss
        wandb.log({'Train Loss': segmentation_loss})
        

    avgTrainLoss = total_training_loss / len(train_dataloader.dataset)


# #---------------------------------------------------------------------------------------------------------


    model.eval()
    total_validation_loss = 0


    with torch.no_grad():
        for i, data in enumerate(tqdm(val_dataloader, desc=f'Epoch {epoch + 1}/{n_epochs}')):
            inputs = data["image"].to(device=device) 
            segmentation_label = data["segmentation"].to(device=device)

            segmentation_output = model(inputs)
            segmentation_loss = segmentation_loss(segmentation_output, segmentation_label)


            
            total_validation_loss += segmentation_loss.item()

    avgValLoss = total_validation_loss / len(val_dataloader)


    print('Epoch [{}/{}]\n'
          'Train Loss: {:.5f} | Train Segmentation Loss: {:.5f}\n'
          'Validation Loss: {:.5f}  | Validation Segmentation Loss: {:.5f}'
          .format(epoch + 1, n_epochs, avgTrainLoss, avgValLoss))
    wandb.log({'Average Train Loss' : avgTrainLoss,
               'Average Validation Loss ': avgValLoss})
    
    if avgValLoss<best_val_loss:
        best_val=model
        best_val_loss=avgValLoss
        torch.save(best_val.state_dict(), "model_state.pth")




