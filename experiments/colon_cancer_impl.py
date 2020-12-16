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
from dataloaders.colon_dataset import ColonCancerBagsCross
from models.colone_cancer import Net

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
ds = ColonCancerBagsCross(path='/home/ikostiuk/git_repos/Multiple-instance-learning-with-graph-neural-networks/datasets/ColonCancer', train_val_idxs=range(100), test_idxs=[], loc_info=False)


def load_train_test_val(ds):
    N = len(ds)
    train = []
    test = []
    val = []
    step = N * 2 // 100
    [train.append((ds[i][0], ds[i][1][0])) for i in range(0, step)]
    print(f"train loaded {len(train)} items")
    #[val.append((ds[i][0], ds[i][1][0])) for i in range(step, step + step // 4)]
    #print(f"valid loaded {len(val)} items")
    [test.append((ds[i][0], ds[i][1][0])) for i in range(step,  step + step // 2)]
    print(f"test loaded {len(test)} items")
    return train, test, val

model = Net().cuda()
optimizer = torch.optim.Adam(model.parameters(),lr=3e-4, betas=(0.9, 0.999), weight_decay=1e-3)
train_loader, test_loader, val_loader = load_train_test_val(ds) 

def train(train_loader):
    model.train()
    loss_all = 0
    ERROR_BY_ITEM = 0.
    ALL = 0
    TP = 0
    FP = 0
    FN = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        if data.shape[0] == 1: # prevent when bag's length equal 1
            continue
        target = torch.tensor(target, dtype=torch.float, requires_grad=True)
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output, l = model(data)

        loss = model.MSE(output, target) + l / 1000
        loss.backward()
        loss_all += loss.item()
        optimizer.step()

        output = torch.ge(output, 0.5)
        TP += output.eq(target.view(-1)).sum().item()
        FN += len(target) - output.eq(target.view(-1)).sum().item()
        ALL += len(target)
        ERROR_BY_ITEM += torch.all(target.eq(torch.ge(output, 0.5).squeeze()))
    
    FP = ALL - TP
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    F1 = 2 * (P * R) / (P + R)
    ERROR_BY_ITEM = ERROR_BY_ITEM / len(train_loader)

    return loss_all, P, R, F1, ERROR_BY_ITEM 

@torch.no_grad()
def test(loader):
    model.eval()
    ERROR_BY_ITEM = 0.
    ALL = 0
    TP = 0
    FP = 0
    FN = 0
    for batch_idx, (data, target) in enumerate(loader):
        if data.shape[0] == 1:
            continue
        target = torch.tensor(target, dtype=torch.float, requires_grad=True)
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        output, _ = model(data)  
        output = torch.ge(output, 0.5)
        TP += output.eq(target.view(-1)).sum().item()
        FN += len(target) - output.eq(target.view(-1)).sum().item()
        ALL += len(target)
        ERROR_BY_ITEM += torch.all(target.eq(torch.ge(output, 0.5).squeeze()))
    
    FP = ALL - TP
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    F1 = 2 * (P * R) / (P + R)
    ERROR_BY_ITEM = ERROR_BY_ITEM / len(loader)

    return P, R, F1, ERROR_BY_ITEM


print("Cuda is is_available: ", torch.cuda.is_available())

for epoch in range(1, 300):
    train_loss, train_P, train_R, train_F1, train_FT = train(train_loader)
    test_P, test_R, test_F1, test_FT = test(test_loader)
    print('Epoch: {:03d}, Train Loss: {:.7f}, Train Prec: {:.3f}, Recall: {:.3f}, F1: {:.3f}, FT: {:.3f},\
    Test Prec: {:.3f}, Recall: {:.3f}, F1: {:.3f}, FT: {:.3f},'.format(epoch, train_loss, train_P, train_R,\
                                                     train_F1, train_FT, test_P, test_R, test_F1, test_FT))