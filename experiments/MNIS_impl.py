from __future__ import print_function
import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable

from dataloaders.mnist_bags_loader import MnistBags
from models.MNIST import GraphBased28x28x1, GraphBased32x32x3
from dataloaders.colon_dataset import ColonCancerBagsCross



def load_MNIST_train_test(number_train_items=10, number_test_items=5):
    print('Loading Train and Test Sets')
    train_loader = data_utils.DataLoader(MnistBags(target_number=9,
                                                   mean_bag_length=10,
                                                   var_bag_length=2,
                                                   num_bag=number_train_items,
                                                   seed=1,
                                                   train=True),
                                         batch_size=1,
                                         num_workers=1,
                                         pin_memory=True,
                                         shuffle=True)
    
    test_loader = data_utils.DataLoader(MnistBags(target_number=9,
                                                  mean_bag_length=10,
                                                  var_bag_length=2,
                                                  num_bag=number_test_items,
                                                  seed=1,
                                                  train=False),
                                        batch_size=1,
                                        num_workers=1,
                                        pin_memory=True,
                                        shuffle=False)
    print("Loaded")
    return train_loader, test_loader
    
def load_CC_train_test(ds):
    N = len(ds)
    train = []
    test = []
    step = N * 2 // 3
    [train.append((ds[i][0], ds[i][1][0])) for i in range(0, step)]
    print(f"train loaded {len(train)} items")
    [test.append((ds[i][0], ds[i][1][0])) for i in range(step,  step + step // 2)]
    print(f"test loaded {len(test)} items")
    return train, test



def train(model, optimizer, train_loader):
    model.train()
    train_loss = 0.
    batch = 4
    ERROR = 1.
    ALL = 1.
    for batch_idx, (data, label) in enumerate(train_loader):
        if data.shape[0] == 1 and not MNIST:
            continue

        target = torch.tensor(label[0], dtype=torch.long)
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        if batch_idx % batch == 0:
            optimizer.zero_grad()

        output, l = model(data)
        loss = model.cross_entropy_loss(output, target) #+ l

        ERROR += model.calculate_classification_error(output, target)
        ALL += 1
        train_loss += loss

        if batch_idx % batch == 0:
            loss.backward()
            optimizer.step()

    train_loss /= ALL
    ERROR /= ALL
    return train_loss, ERROR
    
def test(model, test_loader):
    model.eval()

    ERROR = 1.
    ALL = 1.
    for batch_idx, (data, label) in enumerate(test_loader):
        if data.shape[0] == 1 and not MNIST:
            continue

        target = torch.tensor(label[0], dtype=torch.long)
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        
        output, _ = model(data)  
        ERROR += model.calculate_classification_error(output, target)
        ALL += 1

    ERROR /= ALL

    return  ERROR


if __name__ == "__main__":
    torch.manual_seed(1)
    MNIST = False

    if MNIST:
        train_loader, test_loader = load_MNIST_train_test(300, 100)  
        model = GraphBased28x28x1().cuda()
    else:
        ds = ColonCancerBagsCross(path='/home/ikostiuk/git_repos/Multiple-instance-learning-with-graph-neural-networks/datasets/ColonCancer', train_val_idxs=range(100), test_idxs=[], loc_info=False)
        train_loader, test_loader = load_CC_train_test(ds)
        model = GraphBased32x32x3().cuda()

    optimizer = optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999), weight_decay=1e-3)
    
    for epoch in range(0, 3000):
        train_loss, train_error = train(model, optimizer, train_loader)
        test_error = test(model, test_loader)
    
        print('Epoch: {}, Train Loss: {:.4f}, Train error: {:.4f}, Test error: {:.4f}'.format(epoch, train_loss, train_error, test_error))

