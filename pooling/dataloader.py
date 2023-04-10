from torchvision import transforms, models
from imgaug import parameters as iap
from imgaug import augmenters as iaa
from sklearn.manifold import TSNE
from PIL import Image
from torch import nn
import pandas as pd
import imgaug as ia
import numpy as np
import random
import pickle
import torch
import json
import time
import copy
import os

from utils import *



#### TRANSFORM DATA ####
class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
#             iaa.Scale((224, 224)),
            iaa.Sometimes(0.30, iaa.GaussianBlur(sigma=(0, 3.0))),
            iaa.Sometimes(0.25, iaa.Multiply((0.5, 1.5), per_channel=0.5)),
            iaa.Sometimes(0.20, iaa.Invert(0.25, per_channel=0.5)),
            iaa.Sometimes(0.25, iaa.ReplaceElementwise(
                        iap.FromLowerResolution(iap.Binomial(0.1), size_px=8),
                        iap.Normal(128, 0.4*128),
                        per_channel=0.5)),
            iaa.Sometimes(0.30, iaa.AdditivePoissonNoise(40)),
            iaa.Sometimes(0.30, iaa.Fliplr(0.5)),
            iaa.Sometimes(0.30, iaa.Affine(rotate=(-20, 20), mode='symmetric')),
            iaa.Sometimes(0.30,
                          iaa.OneOf([iaa.Dropout(p=(0, 0.1)),
                                     iaa.CoarseDropout(0.1, size_percent=0.5)])),
            iaa.Sometimes(0.30, iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True))
        ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)






class Dataloader():
    
    def __init__(self, image_names, dta, records_dir, iso, batch_size = 16, split = .75, pooling = False):
        
        self.dta = pd.read_csv(dta)
        self.dta["id"] = self.dta["DHSID"] + "_" + self.dta["cumsum"].astype(str)
        
        if not pooling:
            self.dta = self.dta[self.dta["iso2"] == iso]
        else:
            iso = "global"
            
#         self.dta["hv271"] = self.dta["hv271"] + abs(min(self.dta["hv271"]))
#         self.dta["hv271"] = (self.dta["hv271"] / abs(max(self.dta["hv271"]))) * 100 
        self.dta["mean_norm_hv271"] = self.dta["mean_norm_hv271"] * 100

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
        
        with open("../transform_stats.json", "r") as f:
            transform_stats = json.load(f)
        
        
        self.train_transforms = transforms.Compose([
                    ImgAugTransform(),
                    transforms.ToTensor(),
                    transforms.Normalize(transform_stats[iso]["mean"], transform_stats[iso]["std"])
                  ])
        
        
        self.val_transforms = transforms.Compose([
#                     ImgAugTransform(),
                    transforms.ToTensor(),
                    transforms.Normalize(transform_stats[iso]["mean"], transform_stats[iso]["std"])
                  ])        
        
        self.transforms = {"train": self.train_transforms, "val": self.val_transforms}
        
        
    def shuffle_imagery(self):
        
        random.shuffle(self.train_images)
        random.shuffle(self.val_images)

        self.train_batches = [self.train_images[i * self.batch_size:(i + 1) * self.batch_size] for i in range((len(self.train_images) + self.batch_size - 1) // self.batch_size )]
        self.val_batches = [self.val_images[i * self.batch_size:(i + 1) * self.batch_size] for i in range((len(self.val_images) + self.batch_size - 1) // self.batch_size )]
    
    
    def load_image(self, impath, phase):
        trans = self.transforms[phase]
        if ".ipynb" not in impath:
            im = trans(np.array(Image.open(impath).convert("RGB"))).unsqueeze(0)
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
            imagery = [_ for _ in imagery if self.load_image(_, phase) is not None]
        else:
            imagery = self.val_batches[idx]
            imagery = [_ for _ in imagery if self.load_image(_, phase) is not None]
            
        dhsids = [str(_.split("/")[4].split(".")[0]) for _ in imagery]
        cur = self.dta[(self.dta["id"].isin(dhsids))]# & self.dta["cumsum"].isin(cumsums)]

        x = torch.cat([self.load_image(_, phase) for _ in imagery])
        y = torch.tensor([cur[cur["id"] == _]["mean_norm_hv271"].squeeze() if _ in cur["id"].to_list() else 0 for _ in dhsids])                
        
        return (x, y)
        


class Valoader():
    
    def __init__(self, image_names, dta, records_dir, iso, pooling = False):
        
        self.dta = pd.read_csv(dta)
        self.dta["id"] = self.dta["DHSID"] + "_" + self.dta["cumsum"].astype(str)
#         self.dta = self.dta[self.dta["iso2"] == iso]
#         self.dta["hv271"] = self.dta["hv271"] + abs(min(self.dta["hv271"]))
#         self.dta["hv271"] = (self.dta["hv271"] / abs(max(self.dta["hv271"]))) * 100        
        self.dta["mean_norm_hv271"] = self.dta["mean_norm_hv271"] * 100
        self.records_dir = records_dir

        self.image_names = image_names
        random.shuffle(self.image_names)
        
        if pooling:
            iso = "global"        
        
        with open("../transform_stats.json", "r") as f:
            transform_stats = json.load(f)        
        
        self.trans = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(transform_stats[iso]["mean"], transform_stats[iso]["std"])
                  ])        
        
    def load_image(self, impath):
        if ".ipynb" not in impath:
            im = self.trans(np.array(Image.open(impath).convert("RGB"))).unsqueeze(0)
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
        y = torch.tensor([self.dta[self.dta["id"] == dhsid]["mean_norm_hv271"].squeeze()])
                            
        return (x, y)
    