import os
import random
import scipy.io
import numpy as np 
from PIL import Image
from skimage import io, color
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import torchvision.transforms as transforms
import torchvision.models as models
import torch_geometric.utils as pyg_ut
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
import dataloaders.utils_augmentation as utils_augmentation
from torch.autograd import Variable

from chamferdist import ChamferDistance
from dataloaders.breast_cancer_bags_loader import BreastCancerBags
from models.breast import Net

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
ds = BreastCancerBags(path='datasets\\breast_cancer_dataset\\breast_cancer', train_val_idxs=range(100), test_idxs=[], loc_info=False)


def load_train_test_val(ds):
    N = len(ds)
    train = []
    test = []
    val = []
    step = N * 2 // 20
    [train.append((ds[i][0], ds[i][1])) for i in range(0, step)]
    print(f"train loaded {len(train)} items")
    [test.append((ds[i][0], ds[i][1])) for i in range(step,  step + step // 2)]
    print(f"test loaded {len(test)} items")
    return train, test, val

model = Net().cuda()
optimizer = torch.optim.Adam(model.parameters(),lr=5e-4, betas=(0.9, 0.999), weight_decay=1e-3)
train_loader, test_loader, val_loader = load_train_test_val(ds) 

def train(train_loader):
    loss_all = 0
    ALL = 0
    batch = 1
    ERROR = 0
    
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if data.shape[0] == 1: # prevent when bag's length equal 1
            continue
        
        target = torch.tensor(target, dtype=torch.float, requires_grad=True)
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        if batch_idx % batch == 0:
            optimizer.zero_grad()

        output, l = model(data)

        netloss = (output[0][0] - target) ** 2
        # loss = model.cross_entropy_loss(output, target) + l
        loss = netloss + l
        loss_all += loss.item()
        if batch_idx % batch == 0:
            loss.backward()
            optimizer.step()

        ERROR += model.calculate_classification_error(output, target)
        ALL += 1


    return loss_all, ERROR / ALL

@torch.no_grad()
def test(loader):
    model.eval()
    ALL = 0
    ERROR = 0
    for batch_idx, (data, target) in enumerate(loader):
        if data.shape[0] == 1:
            continue
        target = torch.tensor(target, dtype=torch.float, requires_grad=True)
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        output, _ = model(data)  

        ERROR += model.calculate_classification_error(output, target)
        ALL += 1
    
    return ERROR / ALL


print("Cuda is is_available: ", torch.cuda.is_available())

for epoch in range(1, 3000):
    train_loss, train_EROR = train(train_loader)
    test_ERROR = test(test_loader)
    print('Epoch: {:03d}, Train Loss: {:.7f}, Train ERROR: {:.3f}, Test ERROR: {:.3f}'.format(epoch, train_loss, train_EROR, test_ERROR))