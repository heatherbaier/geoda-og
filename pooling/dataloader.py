from torchvision import transforms, models
from sklearn.manifold import TSNE
from PIL import Image
from torch import nn
import pandas as pd
import numpy as np
import random
import torch
import time
import copy
import os

from utils import *


class Dataloader():
    
    def __init__(self, image_names, dta, records_dir, batch_size = 16, split = .75):
        
        self.dta = pd.read_csv(dta)
        self.dta["id"] = self.dta["DHSID"] + "_" + self.dta["cumsum"].astype(str)
        self.records_dir = records_dir

        self.image_names = image_names
        random.shuffle(self.image_names)
        
        self.split = split
        self.batch_size = batch_size
        train_num = int(len(self.image_names) * self.split)
        train_indices = random.sample(range(len(self.image_names)), train_num)
        val_indices = list(set([i for i in range(0, len(self.image_names))]).difference(train_indices))        
        
        self.train_images = list(map(self.image_names.__getitem__, train_indices))
        self.val_images = list(map(self.image_names.__getitem__, val_indices))
        
        with open(f"{records_dir}/valimages.txt", "a") as f:
            [f.write(_ + "\n") for _ in self.val_images]
        
        self.train_batches = [self.train_images[i * self.batch_size:(i + 1) * self.batch_size] for i in range((len(self.train_images) + self.batch_size - 1) // self.batch_size )]
        self.val_batches = [self.val_images[i * self.batch_size:(i + 1) * self.batch_size] for i in range((len(self.val_images) + self.batch_size - 1) // self.batch_size )]
        
        
    def shuffle_imagery(self):
        
        random.shuffle(self.train_images)
        random.shuffle(self.val_images)

        self.train_batches = [self.train_images[i * self.batch_size:(i + 1) * self.batch_size] for i in range((len(self.train_images) + self.batch_size - 1) // self.batch_size )]
        self.val_batches = [self.val_images[i * self.batch_size:(i + 1) * self.batch_size] for i in range((len(self.val_images) + self.batch_size - 1) // self.batch_size )]
    
    def load_image(self, impath):
        
        if ".ipynb" not in impath:
            im = data_transforms(np.array(Image.open(impath).convert("RGB"))).unsqueeze(0)
            return im        
        
    def get_range(self, phase):
        
        if phase == "train":            
            return len(self.train_batches)
        else:
            return len(self.val_batches)
        
        
    def get_num(self, phase):
        if phase == "train":
            return len(self.train_images)
        else:
            return len(self.val_images)        
        
    
    def load_data(self, idx, phase):
        
        if phase == "train":
            imagery = self.train_batches[idx]
            imagery = [_ for _ in imagery if self.load_image(_) is not None]
        else:
            imagery = self.val_batches[idx]
            imagery = [_ for _ in imagery if self.load_image(_) is not None]
            
        dhsids = [str(_.split("/")[4].split(".")[0]) for _ in imagery]
        cur = self.dta[(self.dta["id"].isin(dhsids))]# & self.dta["cumsum"].isin(cumsums)]

        x = torch.cat([self.load_image(_) for _ in imagery])
        y = torch.tensor([cur[cur["id"] == _]["wealth_class"].squeeze() if _ in cur["id"].to_list() else 0 for _ in dhsids])                
        
        return (x, y)
        


class Valoader():
    
    def __init__(self, image_names, dta, records_dir):
        
        self.dta = pd.read_csv(dta)
        self.dta["id"] = self.dta["DHSID"] + "_" + self.dta["cumsum"].astype(str)
        self.records_dir = records_dir

        self.image_names = image_names
        random.shuffle(self.image_names)

               
    def load_image(self, impath):
        
        if ".ipynb" not in impath:
            im = data_transforms(np.array(Image.open(impath).convert("RGB"))).unsqueeze(0)
            return im        
        
    def get_range(self, phase):
        if phase == "train":
            return len(self.train_batches)
        else:
            return len(self.val_batches)
        
        
    def get_num(self, phase):
        if phase == "train":
            return len(self.train_images)
        else:
            return len(self.val_images)        
        
    
    def load_data(self, idx):
        
        dhsid = str(self.image_names[idx].split("/")[4].split(".")[0])
        x = torch.cat([self.load_image(self.image_names[idx])])
        y = torch.tensor([self.dta[self.dta["id"] == dhsid]["wealth_class"].squeeze()])
                            
        return (x, y)