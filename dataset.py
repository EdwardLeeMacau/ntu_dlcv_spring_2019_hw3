"""
  FileName     [ dataset.py ]
  PackageName  [ DLCV Spring 2019 - GAN / DANN ]
  Synopsis     [ Dataset of the HW3: CelebA, USPS, SVHN, MNISTM. ]
"""

import csv
import os
import random
import time

import numpy as np
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset

import pandas as pd
import utils


class CelebA(Dataset):
    def __init__(self, root, feature, transform=None):
        """ 
          Save the imageNames and the labelNames and read in future.
        """
        self.datas = []
        self.root  = root
        self.feature   = feature
        self.transform = transform

        image_folder = os.path.join(root, "train")
        anno_file    = os.path.join(root, "train.csv")
        dataFrame = pd.read_csv(anno_file)
        
        for _, row in dataFrame.iterrows():
            img_name, keyFeature = row["image_name"], row[feature]
            img_name = os.path.join(image_folder, img_name)
            
            self.datas.append((img_name, keyFeature))
        
        self.len = len(self.datas)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        img_name, feature = self.datas[index]
        
        img = Image.open(img_name)
        
        if feature == 0:
            feature = torch.Tensor([1, 0])
        elif feature == 1:
            feature = torch.Tensor([0, 1])
        
        if self.transform: 
            img = self.transform(img)
        
        return img, feature

class NumberClassify(Dataset):
    def __init__(self, root, feature, train, black=False, transform=None):
        """ 
        Save the imageNames and the labelNames and read in future.
        
        Parameters
        ----------
        root : str

        feature : str

        train : bool

        black : bool
            The dataset is grey

        transform : 
            (...)
        """
        self.datas     = []
        self.black     = black
        self.feature   = feature
        self.transform = transform

        if train:
            image_folder = os.path.join(root, feature, "train")
            anno_file = os.path.join(root, feature, "train.csv")
        else:
            image_folder = os.path.join(root, feature, "test")
            anno_file = os.path.join(root, feature, "test.csv")

        dataFrame = pd.read_csv(anno_file)
        
        for _, row in dataFrame.iterrows():
            img_name, label = row["image_name"], row["label"]
            img_name = os.path.join(image_folder, img_name)
            
            self.datas.append((img_name, label))
        
        self.len = len(self.datas)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        img_name, label = self.datas[index]
        
        img = Image.open(img_name)
        label = torch.Tensor([label])
        
        if self.black:
            img = img.convert("RGB")

        if self.transform: 
            img = self.transform(img)

        return img, label, img_name

# TODO: Merge to NumberClassify
class NumberPredict(Dataset):
    def __init__(self, img_folder, black=False, transform=None):
        """ Handling read imgs only """
        self.datas = []
        self.img_folder = img_folder
        self.black      = black
        self.transform  = transform
        
        self.datas = [os.path.join(img_folder, img_name) for img_name in os.listdir(img_folder)]
        self.len   = len(self.datas)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        img_name = self.datas[index]
        img      = Image.open(img_name)
        
        if self.black:
            img = img.convert("RGB")

        if self.transform: 
            img = self.transform(img)

        return img, img_name
