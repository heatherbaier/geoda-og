from torchvision import transforms, models
# from sklearn.manifold import TSNE
from PIL import Image
from torch import nn
import pandas as pd
import numpy as np
import argparse
import random
import torch
import time
import copy
import os

from dataloader import *
from utils import *




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_name', type = str, required = True)
    parser.add_argument('--iso', type = str, required = False)
    parser.add_argument('--gpu', type = str, required = True)
    parser.add_argument('--pool', type = bool, required = False)
    args = parser.parse_args()
    
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    records_dir = args.folder_name  
    os.mkdir(records_dir)
    os.mkdir(os.path.join(records_dir, "models"))        

    if not args.pool:
        ecup = f"../../imagery/{args.iso}/"
        image_names = [ecup + _ for _ in os.listdir(ecup)]
    else:
        isos = os.listdir("../../imagery/")
        image_names = []
        for iso in isos:
            path = f"../../imagery/{iso}/"
            images = [path + _ for _ in os.listdir(path)]
            image_names += [*images]
    
    print(len(image_names))

#     daga

    data = Dataloader(image_names, "../data/clean/wealth_random_points_mean_norm.csv", records_dir, args.iso, batch_size = 32, pooling = args.pool)
    
#     print(data.load_data(1, "val"))
    
#     agdsga
    
    device = "cuda"
    model_ft = models.resnet50(pretrained = True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 1)
    model_ft = model_ft.to(device)
    criterion = nn.L1Loss()
    optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=0.0001)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)    

    train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, data, device, os.path.join(records_dir, "models"), num_epochs = 50)
