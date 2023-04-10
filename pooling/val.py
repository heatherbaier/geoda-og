from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix
from torchvision import transforms, models
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
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
    os.mkdir(f"{records_dir}/results")
    results_dir = f"{records_dir}/results"

    with open(f"./{records_dir}/valimages.txt", "r") as f:
        image_names = f.read().splitlines()
    image_names = [i for i in image_names if ".ipynb" not in i]

    data = Valoader(image_names, "../data/clean/wealth_random_points_mean_norm.csv", records_dir, args.iso, pooling = args.pool)

    
#     ahaha
    
    
    device = "cuda"
    model_ft = models.resnet50(pretrained = True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 1)
    model_ft = model_ft.to(device)
    
    
    saved_models = [i for i in os.listdir(records_dir + "/models/") if i != "records.txt"]
    saved_models = [i for i in saved_models if ".ipynb" not in i]
    saved_models = [i for i in saved_models if ".sav" not in i]
    weights = records_dir + "/models/" + saved_models[-1]
    print(weights)    

    testm = torch.load(weights)["model_state_dict"]
    model_ft.load_state_dict(testm)
    model_ft.eval();


    preds, trues, ids = [], [], []

    for i in range(len(image_names)):
#     for i in range(10):
        try:
            inp, out = data.load_data(i)
#             print(inp.shape, out)
            inp, out = inp.to(device), out.to(device)
            pred = model_ft(inp)
#             print(pred.item())
            trues.append(out.item())
            preds.append(pred.item())
            ids.append(image_names[i])
            print(i, len(image_names), end = "\r")
        except:
            pass

        
df = pd.DataFrame()
df["school_id"] = ids
df["pred"] = preds
df["true"] = trues
print(df.head())

acc = r2_score(df["true"], df["pred"])
cm = mean_absolute_error(df["true"], df["pred"])

print("R2 Score: ", acc)
print("MAE: ", cm)

df.to_csv(f"{records_dir}/results/preds.csv", index = False)

plt.scatter(df["pred"], df["true"])
plt.savefig(f"{records_dir}/results/results.png")
