"""Pytorch Dataset object that loads 27x27 patches that contain single cells."""

import os
import random
import scipy.io
import numpy as np
from PIL import Image
from skimage import io, color

import torch
import torch.utils.data as data_utils
import torchvision.transforms as transforms

import dataloaders.utils_augmentation as utils_augmentation 


class ColonCancerBagsCross(data_utils.Dataset):
    def __init__(self, path, train_val_idxs=None, test_idxs=None, train=True, shuffle_bag=False, data_augmentation=False, loc_info=False):
        self.path = path
        self.train_val_idxs = train_val_idxs
        self.test_idxs = test_idxs
        self.train = train
        self.shuffle_bag = shuffle_bag
        self.data_augmentation = data_augmentation
        self.location_info = loc_info

        self.data_augmentation_img_transform = transforms.Compose([utils_augmentation.RandomHEStain(),
                                                                   utils_augmentation.HistoNormalize(),
                                                                   utils_augmentation.RandomRotate(),
                                                                   utils_augmentation.RandomVerticalFlip(),
                                                                   transforms.RandomHorizontalFlip(),
                                                                   transforms.ToTensor(),
                                                                   transforms.Normalize((0.5, 0.5, 0.5),
                                                                                        (0.5, 0.5, 0.5))])

        self.normalize_to_tensor_transform = transforms.Compose([
                                                                 utils_augmentation.HistoNormalize(),
                                                                 transforms.ToTensor(),
                                                                 transforms.Normalize((0.5, 0.5, 0.5),
                                                                                      (0.5, 0.5, 0.5))
                                                                    ])

        self.dir_list_train, self.dir_list_test = self.split_dir_list(self.path, self.train_val_idxs, self.test_idxs)
        if self.train:
            self.bag_list_train, self.labels_list_train = self.create_bags(self.dir_list_train)
        else:
            self.bag_list_test, self.labels_list_test = self.create_bags(self.dir_list_test)
            

    @staticmethod
    def split_dir_list(path, train_val_idxs, test_idxs):

        dirs = [x[0] for x in os.walk(path)]
        dirs.pop(0)
        dirs.sort()

        dir_list_train = [dirs[i] for i in train_val_idxs]
        dir_list_test = [dirs[i] for i in test_idxs]
  
        return dir_list_train, dir_list_test

    def create_bags(self, dir_list):
        bag_list = []
        labels_list = []
        for dir in dir_list:
            # Get image name
            img_name = dir.split('\\')[-1]

            # bmp to pillow
            img_dir = dir + '\\' + img_name + '.bmp'
            img = io.imread(img_dir)
            if img.shape[2] == 4:
                img = color.rgba2rgb(img)

            if self.location_info:
                xs = np.arange(0, 500)
                xs = np.asarray([xs for i in range(500)])
                ys = xs.transpose()
                img = np.dstack((img, xs, ys))
            
            cropped_cells = []
            labels = []
            # crop cells
            for label, cell_type in enumerate(['epithelial', 'fibroblast', 'inflammatory', 'others']):
                dir_cell = dir + '/' + img_name + '_' + cell_type + '.mat'
                with open(dir_cell, 'rb') as f:
                    mat_cell = scipy.io.loadmat(f)
                
                for (x,y) in mat_cell['detection']:
                    x = np.round(x)
                    y = np.round(y)

                    if self.data_augmentation:
                        x = x + np.round(np.random.normal(0, 3, 1))
                        y = y + np.round(np.random.normal(0, 3, 1))

                    if x < 13:
                        x_start = 0
                        x_end = 27
                    elif x > 500 - 14:
                        x_start = 500 - 27
                        x_end = 500
                    else:
                        x_start = x - 13
                        x_end = x + 14

                    if y < 13:
                        y_start = 0
                        y_end = 27
                    elif y > 500 - 14:
                        y_start = 500 - 27
                        y_end = 500
                    else:
                        y_start = y - 13
                        y_end = y + 14

                    cropped_cells.append(img[int(y_start):int(y_end), int(x_start):int(x_end)])
                    labels.append(label)

                # generate bag
                bag = cropped_cells

            # store single cell labels
            labels = np.array(labels)

            # shuffle
            if self.shuffle_bag:
                zip_bag_labels = list(zip(bag, labels))
                random.shuffle(zip_bag_labels)
                bag, labels = zip(*zip_bag_labels)

            # append every bag two times if training
            if self.train:
                for _ in [0,1]:
                    bag_list.append(bag)
                    labels_list.append(labels)
            else:
                bag_list.append(bag)
                labels_list.append(labels)

            # bag_list.append(bag)
            # labels_list.append(labels)

        return bag_list, labels_list

    def transform_and_data_augmentation(self, bag):
        if self.data_augmentation:
            img_transform = self.data_augmentation_img_transform
        else:
            img_transform = self.normalize_to_tensor_transform

        bag_tensors = []
        for img in bag:
            if self.location_info:
                bag_tensors.append(torch.cat(
                    (img_transform(img[:, :, :3]), 
                    torch.from_numpy(img[:, :, 3:].astype(float).transpose((2, 0, 1))).float())))
            else:
                bag_tensors.append(img_transform(img))
        
        return torch.stack(bag_tensors)

    def __len__(self):
        if self.train:
            return len(self.labels_list_train)
        else:
            return len(self.labels_list_test)

    def __getitem__(self, index):
        if self.train:
#             print(self.dir_list_train[index])
            bag = self.bag_list_train[index]
            bag_lbls = np.array([1.0 if cat in self.labels_list_train[index] else 0.0 for cat in range(4)])
            label = [bag_lbls, self.labels_list_train[index]]
#             label = [max(self.labels_list_train[index]), self.labels_list_train[index]]
        else:
            bag = self.bag_list_test[index]
            bag_lbls = np.array([1.0 if cat in self.labels_list_test[index] else 0.0 for cat in range(4)])
            label = [bag_lbls, self.labels_list_test[index]]
#             label = [max(self.labels_list_test[index]), self.labels_list_test[index]]

        return self.transform_and_data_augmentation(bag), label
