"""
  FileName     [ dann.py ]
  PackageName  [ DLCV Spring 2019 - DANN ]
  Synopsis     [ DANN Models ]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Function
from torch.utils.data import DataLoader
from torchvision import transforms

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.alpha
        return grad_output, None

def grad_reverse(x, constant):
    return ReverseLayerF.apply(x, constant)

class Feature_Extractor(nn.Module):
    def __init__(self):
        super(Feature_Extractor, self).__init__()

        self.feature = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),

            nn.MaxPool2d(kernel_size=2, stride=2),

            # Layer 2
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),

            nn.MaxPool2d(kernel_size=2, stride=2),

            # Layer 3
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Dropout2d(0.5),
        )

    def forward(self, x):
        x = x.expand(-1, 3, 28, 28)
        x = self.feature(x)
        # print("Feature.shape: \t{}".format(x.shape))

        return x

class Class_Classifier(nn.Module):
    def __init__(self):
        super(Class_Classifier, self).__init__()
        
        self.class_detect = nn.Sequential(
            # Layer 1
            nn.Linear(128 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            # Layer 2
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),

            # Layer 3
            nn.Linear(2048, 10),
            # nn.LogSoftmax(dim=1),
        )

    def forward(self, feature):
        x = self.class_detect(feature)
    
        return x

class Domain_Classifier(nn.Module):
    def __init__(self):
        super(Domain_Classifier, self).__init__()

        self.domain_detect = nn.Sequential(
            # Layer 1
            nn.Linear(128 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            # Layer 2
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            # Layer 3
            nn.Linear(1024, 2),
            # nn.LogSoftmax(dim=1),
        )

    def forward(self, feature, constant):
        # feature = ReverseLayerF.grad_reverse(feature, constant)
        feature = grad_reverse(feature, constant)
        d = self.domain_detect(feature)

        return d


def dann_unittest():
    SOURCE = "usps"
    DEVICE = utils.selectDevice()
    
    feature_extractor = Feature_Extractor().to(DEVICE)
    class_classifier  = Class_Classifier().to(DEVICE)
    domain_classifier = Domain_Classifier().to(DEVICE)

    trainset = dataset.NumberClassify("./hw3_data/digits", SOURCE, train=True, black=True, transform=transforms.ToTensor())
    
    loader = DataLoader(trainset, batch_size=2, shuffle=True, num_workers=0)
    
    for _, (img, label, _) in enumerate(loader):
        img, label = img.to(DEVICE), label.to(DEVICE)

        print("Label.shape: \t{}".format(label.shape))

        f = feature_extractor(img)
        print("Feature.shape: \t{}".format(f.shape))

        f = f.view(f.shape[0], -1)
        y = class_classifier(f)
        print("Class_Label.shape: \t{}".format(y.shape))

        d = domain_classifier(f, 0.5)
        print("Domain_Label.shape: \t{}".format(d.shape))

        # Print 1 time only
        break

if __name__ == "__main__":
    dann_unittest()
    print("DANN Unittest Done!")
