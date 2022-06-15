"""
  FileName     [ adda.py ]
  PackageName  [ DLCV Spring 2019 - ADDA ]
  Synopsis     [ ADDA Models ]

  Reference
  1. https://arxiv.org/abs/1702.05464
"""

import torch.nn.functional as F
from torch import nn

__all__ = ['Feature', 'Classifier', 'Discriminator']

class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()
        
        self.restored = False

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),

            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
        )

    def forward(self, img):
        feature = self.encoder(img)

        return feature

class Classifier(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims):
        super(Classifier, self).__init__()
        
        self.classify = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(hidden_dims, output_dims),
        )

    def forward(self, feature):
        x = self.classify(feature)

        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.restored = False
        self.classify = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=3, padding=1),
            # nn.Linear(input_dims, hidden_dims),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=0),
            # nn.Linear(hidden_dims, hidden_dims),
            nn.LeakyReLU(0.2, inplace=True),
            
            # nn.Linear(hidden_dims, output_dims),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
            # Using CrossEntropy to train this layer.
            # nn.LogSoftmax()
        )

    def forward(self, feature):
        out = self.classify(feature)

        return out
