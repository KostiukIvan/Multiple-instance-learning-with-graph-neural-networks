from __future__ import print_function
import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable

from dataloaders.mnist_bags_loader import MnistBags
from models.MNIST import GraphBased


def load_train_test(number_train_items=10, number_test_items=5):
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
    



def train(model, optimizer, train_loader):

    model.train()
    train_loss = 0.
    train_error = 0.
    for batch_idx, (data, label) in enumerate(train_loader):
        bag_label = label[0]
        if torch.cuda.is_available():
            data, bag_label = data.cuda(), bag_label.cuda()

        optimizer.zero_grad()
        loss = model.cross_entropy_loss(data, bag_label)
        train_loss += loss
        error, _ = model.calculate_classification_error(data, bag_label)
        train_error += error
        loss.backward()
        optimizer.step()

    train_loss /= len(train_loader)
    train_error /= len(train_loader)
    return train_loss, train_error
    
def test(model, test_loader):
    model.eval()
    test_loss = 0.
    test_error = 0.
    for batch_idx, (data, label) in enumerate(test_loader):
        bag_label = label[0]
        instance_labels = label[1]
        if torch.cuda.is_available():
            data, bag_label = data.cuda(), bag_label.cuda()

        error, predicted_label = model.calculate_classification_error(data, bag_label)
        test_error += error

    test_error /= len(test_loader)

    return train_loss, train_error


if __name__ == "__main__":
    torch.manual_seed(1)
    train_loader, test_loader = load_train_test(100, 50)
        
    model = GraphBased().cuda()
    optimizer = optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999), weight_decay=1e-3)
    
    for epoch in range(0, 100):
        train_loss, train_error = train(model, optimizer, train_loader)
        test_loss, test_error = test(model, test_loader)
    
        print('Epoch: {}, Train Loss: {:.4f}, Train error: {:.4f}, Test error: {:.4f}'.format(epoch, train_loss.cpu().detach().numpy(), train_error, test_error))

