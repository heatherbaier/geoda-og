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
import time
import copy
import os


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
                        per_channel=0.5)
                                             ),
                    iaa.Sometimes(0.30, iaa.AdditivePoissonNoise(40)),
            iaa.Fliplr(0.5),
            iaa.Affine(rotate=(-20, 20), mode='symmetric'),
            iaa.Sometimes(0.30,
                          iaa.OneOf([iaa.Dropout(p=(0, 0.1)),
                                     iaa.CoarseDropout(0.1, size_percent=0.5)])),
            iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
        ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)



data_transforms = transforms.Compose([
                    ImgAugTransform(),
                    transforms.ToTensor(),
#                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                  ])
	
# data_transforms = {
#     'train': transforms.Compose([
#     		ImgAugTransform(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
#     'val': transforms.Compose([
#     		ImgAugTransform(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
# }


def train_model(model, criterion, optimizer, scheduler, data, device, folder_name, num_epochs=25):

    epoch_num = 0

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000000
    best_acc = 0

    for epoch in range(num_epochs):
        
        data.shuffle_imagery()
        
        with open(f"./{folder_name}/records.txt", "a") as f:
            f.write('Epoch {}/{}\n'.format(epoch, num_epochs - 1))
            
        with open(f"./{folder_name}/records.txt", "a") as f:
            f.write('----------\n')
            
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data.
            for batch in range(data.get_range(phase)):

#             for batch in range(5):

                inputs, labels = data.load_data(batch, phase)
                    
#                 print(inputs, labels)
                
#                 print(labels)
                
                
#                 agad
                
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)    
                    
#                     print(preds, labels)
                    
#                     print(labels)
                    
#                     _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
    
                    loss = torch.nan_to_num(loss, 0)
                    
#                     print(outputs, loss)
# #                     
#                     aldgjla

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
#                     sdaga

                # statistics
                running_loss += loss.item() * inputs.size(0)
#                 print(running_loss, loss, data.get_num(phase))
                running_corrects += torch.sum(preds == labels.data)
    
#                 print(running_corrects)
                
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / data.get_num(phase)
            epoch_acc = running_corrects.double() / data.get_num(phase)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            with open(f"./{folder_name}/records.txt", "a") as f:
                f.write('{} Loss: {:.4f} Acc: {:.4f}\n'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

                # Save each epoch that achieves a higher accuracy than the current best_acc in case the model crashes mid-training
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': best_model_wts,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': criterion,
                        }, f"./{folder_name}/model_epoch{epoch}.torch")     
                
                pickle.dump(model, open(f"./{folder_name}/model_epoch{epoch}.sav", 'wb'))                
                
                
        epoch_num += 1

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

