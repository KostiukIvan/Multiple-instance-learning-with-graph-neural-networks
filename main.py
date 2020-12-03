from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plt

import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable

from dataloader import MnistBags
from model import GraphBased

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
#parser.add_argument('--epochs', type=int, default=20, metavar='N',
#                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                    help='weight decay')
parser.add_argument('--target_number', type=int, default=9, metavar='T',
                    help='bags have a positive labels if they contain at least one 9')
parser.add_argument('--mean_bag_length', type=int, default=10, metavar='ML',
                    help='average bag length')
parser.add_argument('--var_bag_length', type=int, default=2, metavar='VL',
                    help='variance of bag length')
parser.add_argument('--num_bags_train', type=int, default=100, metavar='NTrain',
                    help='number of bags in training set')
parser.add_argument('--num_bags_test', type=int, default=30, metavar='NTest',
                    help='number of bags in test set')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--model', type=str, default='graph_based', help='Choose b/w attention and gated_attention or graph_based')




def load_train_test(args):
    print('Load Train and Test Set')
    
    train_loader = data_utils.DataLoader(MnistBags(target_number=args.target_number,
                                                   mean_bag_length=args.mean_bag_length,
                                                   var_bag_length=args.var_bag_length,
                                                   num_bag=args.num_bags_train,
                                                   seed=args.seed,
                                                   train=True),
                                         batch_size=1,
                                         shuffle=True)
    
    test_loader = data_utils.DataLoader(MnistBags(target_number=args.target_number,
                                                  mean_bag_length=args.mean_bag_length,
                                                  var_bag_length=args.var_bag_length,
                                                  num_bag=args.num_bags_test,
                                                  seed=args.seed,
                                                  train=False),
                                        batch_size=1,
                                        shuffle=False)
    return train_loader, test_loader
    
def load_model(args):

    if args.model=='graph_based':
        model = GraphBased()
    if args.cuda:
        model.cuda()
    
    return model


def train(model, optimizer, train_loader):

    model.train()
    train_loss = 0.
    train_error = 0.
    for batch_idx, (data, label) in enumerate(train_loader):
        bag_label = label[0]
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()

        # reset gradients
        optimizer.zero_grad()
        # calculate loss and metrics
        loss = model.cross_entropy_loss(data, bag_label)
        train_loss += loss
        error, _ = model.calculate_classification_error(data, bag_label)
        train_error += error
        # backward pass
        loss.backward()
        # step
        optimizer.step()

    # calculate loss and error for epoch
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
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()

        error, predicted_label = model.calculate_classification_error(data, bag_label)
        test_error += error

    test_error /= len(test_loader)

    return train_loss, train_error


if __name__ == "__main__":
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        print('\nGPU is ON!')
        
    train_loader, test_loader = load_train_test(args)
        
    model = load_model(args)
    optimizer = optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999), weight_decay=1e-3)
    
    for epoch in range(0, 100):
        train_loss, train_error = train(model, optimizer, train_loader)
        test_loss, test_error = test(model, test_loader)
    
        print('Epoch: {}, Train Loss: {:.4f}, Train error: {:.4f}, Test error: {:.4f}'.format(epoch, train_loss.cpu().detach().numpy(), train_error, test_error))

