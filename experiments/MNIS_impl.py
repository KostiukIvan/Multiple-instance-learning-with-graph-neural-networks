from __future__ import print_function
import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable

from models.MNIST import GraphBased28x28x1, GraphBased27x27x3, GraphBased50x50x3

from dataloaders.mnist_bags_loader import MnistBags
from dataloaders.colon_dataset import ColonCancerBagsCross
from dataloaders.breast_cancer_bags_loader import BreastCancerBags

MNIST = False
COLON = False
BREAST = True

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
    step = N * 7 // 10
    [train.append((ds[i][0], ds[i][1][0])) for i in range(0, step)]
    print(f"train loaded {len(train)} items")
    [test.append((ds[i][0], ds[i][1][0])) for i in range(step, step + N * 3 // 10)]
    print(f"test loaded {len(test)} items")
    return train, test

def load_BREAST_train_test(ds):
    N = len(ds)
    train = []
    test = []
    step = N * 7 // 10
    [train.append((ds[i][0], ds[i][1])) for i in range(0, step)]
    print(f"train loaded {len(train)} items")
    [test.append((ds[i][0], ds[i][1])) for i in range(step, step + N * 3 // 10)]
    print(f"test loaded {len(test)} items")
    return train, test


def train(model, optimizer, train_loader):
    model.train()
    train_loss = 0.
    batch = 4
    
    TP = [0.]
    TN = [0.]
    FP = [0.]
    FN = [0.]
    ALL = 0.
    for batch_idx, (data, label) in enumerate(train_loader):
        if data.shape[0] == 1 and not MNIST:
            continue
        
        if BREAST:
            target = torch.tensor(label, dtype=torch.long)
        else:
            target = torch.tensor(label[0], dtype=torch.long)
        
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        if batch_idx % batch == 0:
            optimizer.zero_grad()

        output, l = model(data)
        loss = model.cross_entropy_loss(output, target) + l

        model.calculate_classification_error(output, target, TP, TN, FP, FN)
        ALL += 1
        train_loss += loss

        if batch_idx % batch == 0:
            loss.backward()
            optimizer.step()

    train_loss /= ALL

    Accuracy = (TP[0] + TN[0]) / ALL
    Precision = TP[0] / (TP[0] + FP[0]) if (TP[0] + FP[0]) != 0. else TP[0]
    Recall =  TP[0] / (TP[0] + FN[0]) if (TP[0] + FN[0]) != 0. else TP[0]
    F1 = 2 * (Recall * Precision) / (Recall + Precision) if (Recall + Precision) != 0 else  2 * (Recall * Precision)

    return train_loss, Accuracy, Precision, Recall, F1
    
def test(model, test_loader):
    model.eval()

    TP = [0.]
    TN = [0.]
    FP = [0.]
    FN = [0.]
    ALL = 0.
    for batch_idx, (data, label) in enumerate(test_loader):
        if data.shape[0] == 1 and not MNIST:
            continue
        
        if BREAST:
            target = torch.tensor(label, dtype=torch.long)
        else:
            target = torch.tensor(label[0], dtype=torch.long)
        
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        
        output, _ = model(data)  
        model.calculate_classification_error(output, target, TP, TN, FP, FN)
        ALL += 1

    Accuracy = (TP[0] + TN[0]) / ALL
    Precision = TP[0] / (TP[0] + FP[0]) if (TP[0] + FP[0]) != 0. else TP[0]
    Recall =  TP[0] / (TP[0] + FN[0]) if (TP[0] + FN[0]) != 0. else TP[0]
    F1 = 2 * (Recall * Precision) / (Recall + Precision) if (Recall + Precision) != 0 else  2 * (Recall * Precision)

    return  Accuracy, Precision, Recall, F1


if __name__ == "__main__":
    torch.manual_seed(1)
    PATH = '/home/ikostiuk/git_repos/Multiple-instance-learning-with-graph-neural-networks/models/saved/'


    if MNIST:
        train_loader, test_loader = load_MNIST_train_test(300, 100)  
        model = GraphBased28x28x1().cuda()
        optimizer = optim.Adam(model.parameters(), lr=3e-6, betas=(0.9, 0.999), weight_decay=1e-3)
    elif COLON:
        ds = ColonCancerBagsCross(path='/home/ikostiuk/git_repos/Multiple-instance-learning-with-graph-neural-networks/datasets/ColonCancer', train_val_idxs=range(100), test_idxs=[], loc_info=False)
        train_loader, test_loader = load_CC_train_test(ds)
        model = GraphBased27x27x3().cuda()
        optimizer = optim.Adam(model.parameters(), lr=3e-6, betas=(0.9, 0.999), weight_decay=1e-3)
    elif BREAST:
        ds = BreastCancerBags(path='/mnt/users/ikostiuk/local/MIL/breast_cancer_bags/breast_cancer', train_val_idxs=range(100), test_idxs=[], loc_info=False)
        train_loader, test_loader = load_BREAST_train_test(ds)
        model = GraphBased50x50x3().cuda()
        optimizer = optim.Adam(model.parameters(), lr=3e-6, betas=(0.9, 0.999), weight_decay=1e-3)
    else:
        print("You don't have such dataset!!!")

    
    
    for epoch in range(0, 100000):
        train_loss, tr_Accuracy, tr_Precision, tr_Recall, tr_F1 = train(model, optimizer, train_loader)
        ts_Accuracy, ts_Precision, ts_Recall, ts_F1 = test(model, test_loader)
        '''
        if epoch % 100 == 0:
            # save model
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            }, os.path.join(PATH, "MNIST_model_" + str(epoch) + ".pth"))
        '''

    
        print('Epoch: {}, Train Loss: {:.4f}, Train A: {:.4f}, P: {:.4f}, R: {:.4f}, F1: {:.4f}, Test A: {:.4f}, P: {:.4f}, R: {:.4f}, F1: {:.4f}'.format(epoch, train_loss, \
                    tr_Accuracy, tr_Precision, tr_Recall, tr_F1, ts_Accuracy, ts_Precision, ts_Recall, ts_F1))