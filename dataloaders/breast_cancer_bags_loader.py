import os
import random
import scipy.io
import numpy as np
from PIL import Image
from skimage import io, color
import glob

import torch
import torch.utils.data as data_utils
import torchvision.transforms as transforms

import dataloaders.utils_augmentation as utils_augmentation 


class BreastCancerBags(data_utils.Dataset):
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
                                                                 transforms.Resize((50, 50)),
                                                                 transforms.ToTensor(),
                                                                 transforms.Normalize((0.5, 0.5, 0.5),
                                                                                      (0.5, 0.5, 0.5))
                                                                    ])

        
        self.healthy, self.cancerous = self.load_image_dirs(self.path)
        if self.train:
            self.bag_list_train, self.labels_list_train = self.create_bags(self.healthy, self.cancerous)
        else:
            self.bag_list_test, self.labels_list_test = self.create_bags(self.healthy, self.cancerous)
        
            
    def load_image_dirs(self, path):
        dirs = glob.glob(path + "/*/")
        class0, class1 = [], []
        for patient in dirs:
            cls0, cls1 = glob.glob(patient + "/*/")
            class0.append(cls0)
            class1.append(cls1)

        return class0, class1

    def create_bags(self, healthy, cancerous, cancerous_chance=0.5, cancerous_max=3, mean_bag_len=10, var_bag_len=2):
        bag_list = []
        labels_list = []

        random_numbers = np.random.randint(mean_bag_len - var_bag_len, mean_bag_len + var_bag_len + 1, len(healthy))
        cancer_chances = np.random.random(len(healthy)) >= cancerous_chance
        for i, patient in enumerate(zip(healthy, cancerous)):
            healthy_images = glob.glob(patient[0] + "/*")
            cancerous_images = glob.glob(patient[1] + "/*")

            #shuffling images for every patient separately
            data = list(zip(healthy_images, cancerous_images))
            random.shuffle(data)
            healthy_images, cancerous_images = zip(*data)
            
            bag_len = random_numbers[i]
            is_cancerous = cancer_chances[i]

            amount_cancerous = 0
            amount_healthy = bag_len

            if is_cancerous:
                amount_cancerous = np.random.randint(cancerous_max) + 1
                amount_healthy = bag_len - amount_cancerous

            amount_healthy = min(amount_healthy, len(healthy_images))
            amount_cancerous = min(amount_cancerous, len(cancerous_images))

            if amount_healthy + amount_cancerous < bag_len:
                continue

            cut_healthy_images, cut_cancerous_images = [], []
            for j in range(amount_healthy):
                cut_healthy_images.append(healthy_images[j])

            for j in range(amount_cancerous):
                cut_cancerous_images.append(cancerous_images[j])

            healthy_images, cancerous_images = cut_healthy_images, cut_cancerous_images

            bag, labels = [], []

            #loading images
            for j, img in enumerate(healthy_images):
                bag.append(io.imread(img))

            for j, img in enumerate(cancerous_images):
                bag.append(io.imread(img))

            # shuffle
            if self.shuffle_bag:
                random.shuffle(bag)

            bag_list.append(bag)
            labels_list.append(int(is_cancerous))

        return np.asarray(bag_list), np.asarray(labels_list)

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
            bag = self.bag_list_train[index]
            label = self.labels_list_train[index]
        else:
            bag = self.bag_list_test[index]
            label = self.labels_list_test[index]

            

        return self.transform_and_data_augmentation(bag), label
