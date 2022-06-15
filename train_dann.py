"""
  FileName     [ train.py ]
  PackageName  [ DLCV Spring 2019 - DANN ]
  Synopsis     [ DANN training methods ]
"""

import argparse
import datetime
import os
import random

import numpy as np
import pandas as pd
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import dataset
import predict
import utils
from TransferLearning.dann import (Class_Classifier, Domain_Classifier,
                                   Feature_Extractor, ReverseLayerF,
                                   grad_reverse)

# Set as true when the I/O shape of the model is fixed
cudnn.benchmark = True
DEVICE = utils.selectDevice()

def train(feature_extractor, class_classifier, domain_classifier, source_loader, target_loader, 
          optim, epoch, class_criterion, domain_criterion):
    """ 
    Train each epoch with DANN framework. 
    
    Parameters
    ----------
    feature_extractor : nn.Module

    class_classifier : nn.Module

    domain_classifier : nn.Module

    source_loader, target_loader : DataLoader

    optim :

    epoch : int

    class_criterion : 
        Loss Function

    domain_criterion :
        Loss Function of Domain Classifier 
    """
    loss = []
    feature_extractor.train()
    class_classifier.train()
    domain_classifier.train()

    for index, (source_batch, target_batch) in enumerate(zip(source_loader, target_loader), 1):
        # Prepare the images and labels
        source_img, source_label, _ = source_batch
        target_img, target_label, _ = target_batch
        batch_size_src, batch_size_tgt = source_img.shape[0], target_img.shape[0]

        source_img   = source_img.to(DEVICE)
        source_label = source_label.type(torch.long).view(-1).to(DEVICE)
        target_img   = target_img.to(DEVICE)
        target_label = target_label.type(torch.long).view(-1).to(DEVICE)

        source_domain_labels = torch.zeros(batch_size_src).type(torch.long).to(DEVICE)
        target_domain_labels = torch.ones(batch_size_tgt).type(torch.long).to(DEVICE)
        
        #-------------------------------------------
        # Setup optimizer
        # Prepare the learning rate and the constant
        #-------------------------------------------
        # p = float(index + (epoch - 1) * len(source_loader)) / (opt.epochs * len(source_loader))
        constant = opt.alpha * (1 + index / min(len(source_loader), len(target_loader)))
        # constant = opt.alpha

        # optim = utils.set_optimizer_lr(optim, p)
        optim.zero_grad()

        #-------------------------------
        # Get features, class pred, domain pred:
        #-------------------------------
        source_feature = feature_extractor(source_img).view(-1, 128 * 7 * 7)
        target_feature = feature_extractor(target_img).view(-1, 128 * 7 * 7)
        
        source_class_predict  = class_classifier(source_feature)
        source_domain_predict = domain_classifier(source_feature, constant)
        target_domain_predict = domain_classifier(target_feature, constant)

        #---------------------------------------
        # Compute the loss
        # For the case of unsupervised learning:
        #   When source domain img:
        #     loss = class_loss + domain_loss
        #   When target domain img:
        #     loss = domain_loss
        # 
        #   Needs to maximize the domain_loss 
        #------------------------------------
        # 1. class loss
        class_loss = class_criterion(source_class_predict, source_label)
        
        # 2. Domain loss
        # print("Source_domain_predict.shape: \t{}".format(source_domain_predict.shape))
        # print("Source_domain_labels.shape: \t{}".format(source_domain_labels.shape))
        source_domain_loss = domain_criterion(source_domain_predict, source_domain_labels)
        target_domain_loss = domain_criterion(target_domain_predict, target_domain_labels)
        domain_loss = target_domain_loss + source_domain_loss

        # Final loss
        loss = class_loss + constant * domain_loss
        loss.backward()
        optim.step()

        source_class_predict = source_class_predict.cpu().detach().numpy()
        source_label = source_label.cpu().detach().numpy()

        source_acc = np.mean(np.argmax(source_class_predict, axis=1) == source_label)

        if index % opt.log_interval == 0:
            # print(constant)
            print("[Epoch {}] [ {:4d}/{:4d} ] [src_acc: {:.2f}%] [loss_Y: {:.4f}] [loss_D: {:.4f}]".format(
                    epoch, index, min(len(target_loader), len(source_loader)), 100 * source_acc, class_loss.item(), constant * domain_loss.item()))

    return feature_extractor, class_classifier, domain_classifier

def domain_adaptation(source, target, epochs, threshold, lr, weight_decay):
    """ 
    Using DANN framework to train the network. 
    
    Parameter
    ---------
    source, target : str

    epochs : int

    threshold : float
        Maximum accuracy of pervious epochs

    lr : float
        Learning Rate 

    weight_decay : float
        Weight Regularization

    Return
    ------
    feature_extractor, class_classifier, domain_classifier : nn.Module
        (...)
    """
    feature_extractor = Feature_Extractor().to(DEVICE)
    class_classifier  = Class_Classifier().to(DEVICE)
    domain_classifier = Domain_Classifier().to(DEVICE)

    optimizer = optim.Adam([
        {'params': feature_extractor.parameters()},
        {'params': class_classifier.parameters()},
        {'params': domain_classifier.parameters()}], 
        lr=lr, betas=(opt.b1, opt.b2), weight_decay=weight_decay
    )
    
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.2)
    
    class_criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
    domain_criterion = torch.nn.CrossEntropyLoss().to(DEVICE)

    # Create Dataloader
    # TODO: Make attributes nametuple for the datasets
    source_black = True if source == 'usps' else False
    target_black = True if target == 'usps' else False
    source_train_set = dataset.NumberClassify(
        "./hw3_data/digits", source, 
        train=True, 
        black=source_black, 
        transform=transforms.ToTensor()
    )
    target_train_set = dataset.NumberClassify(
        "./hw3_data/digits", target, 
        train=True, 
        black=target_black, 
        transform=transforms.ToTensor()
    )
    source_test_set  = dataset.NumberClassify(
        "./hw3_data/digits", source, 
        train=False, 
        black=source_black, 
        transform=transforms.ToTensor()
    )
    target_test_set  = dataset.NumberClassify(
        "./hw3_data/digits", target, 
        train=False, 
        black=target_black, 
        transform=transforms.ToTensor()
    )
    print("Source_train: \t{}, {}".format(source, len(source_train_set)))
    print("Source_test: \t{}, {}".format(source, len(source_test_set)))
    print("Target_train: \t{}, {}".format(target, len(target_train_set)))
    print("Target_test: \t{}, {}".format(target, len(target_test_set)))
    
    source_train_loader = DataLoader(source_train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.threads)
    source_test_loader  = DataLoader(source_test_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.threads)
    target_train_loader = DataLoader(target_train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.threads)
    target_test_loader  = DataLoader(target_test_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.threads)
    
    source_pred_values = []
    target_pred_values = []

    for epoch in range(1, epochs + 1):
        scheduler.step()

        feature_extractor, class_classifier, domain_classifier = train(feature_extractor, class_classifier, domain_classifier, source_train_loader, target_train_loader, optimizer, epoch, class_criterion, domain_criterion)
        source_pred_value = predict.val(feature_extractor, class_classifier, domain_classifier, source_test_loader, 0, class_criterion, domain_criterion)
        target_pred_value = predict.val(feature_extractor, class_classifier, domain_classifier, target_test_loader, 1, class_criterion, domain_criterion)
        # Return: class_acc, class_loss, domain_acc, domain_loss

        print("[Epoch {}] [ src_acc: {:.2f}% ] [ tgc_acc: {:.2f}% ] [ src_loss: {:.4f} ] [ tgc_loss: {:.4f} ]".format(
               epoch, 100 * source_pred_value[0], 100 * target_pred_value[0], source_pred_value[1] + source_pred_value[3], target_pred_value[1] + target_pred_value[3]))

        # Tracing the accuracy, loss
        source_pred_values.append(source_pred_value)
        target_pred_values.append(target_pred_value)

        y_source = np.asarray(source_pred_values, dtype=float)
        y_target = np.asarray(target_pred_values, dtype=float)

        x = np.arange(start=1, stop=epoch+1)
        plt.clf()
        plt.figure(figsize=(12.8, 7.2))
        plt.plot(x, y_source[:, 0], 'r', label='Source Class Accuracy', linewidth=1)
        plt.plot(x, y_target[:, 0], 'b', label='Target Class Accuracy', linewidth=1)
        plt.plot(x, np.asarray([0.3]).repeat(len(x)), 'b-', linewidth=0.1)
        plt.plot(x, np.asarray([0.4]).repeat(len(x)), 'b-', linewidth=0.1)
        plt.plot(x, y_source[:, 2], 'r:', label='Source Domain Accuracy', linewidth=1)
        plt.plot(x, y_target[:, 2], 'b:', label='Target Domain Accuracy', linewidth=1)
        plt.legend(loc=0)
        plt.xlabel("Epochs(s)")
        plt.title("[Acc - Train {} Test {}] vs Epoch(s)".format(source, target))
        plt.savefig("DANN_{}_{}_{}-Acc.png".format(opt.alpha, source, target))
        plt.close()

        plt.clf()
        plt.figure(figsize=(12.8, 7.2))
        plt.plot(x, y_source[:, 1], 'r', label='Source Class Loss', linewidth=1)
        plt.plot(x, y_target[:, 1], 'b', label='Target Class Loss', linewidth=1)
        plt.plot(x, y_source[:, 3], 'r:', label='Source Domain Loss', linewidth=1)
        plt.plot(x, y_target[:, 3], 'b:', label='Target Domain Loss', linewidth=1)
        plt.legend(loc=0)
        plt.xlabel("Epochs(s)")
        plt.title("[Loss - Train {} Test {}] vs Epoch(s)".format(source, target))
        plt.savefig("DANN_{}_{}_{}-Loss.png".format(opt.alpha, source, target))
        plt.close()

        with open('statistics.txt', 'a') as textfile:
            textfile.write(datetime.datetime.now().strftime("%d, %b %Y %H:%M:%S"))
            textfile.write(str(source_pred_values))
            textfile.write(str(target_pred_values))

        if target_pred_value[0] > threshold:
            # Update the threshold to the maximum if the target acc was improved.
            if target_pred_value[0] > threshold:
                threshold = target_pred_value[0]
            
            utils.saveDANN("./models/dann/{}/DANN_{}_{}_{}.pth".format(opt.tag, source, target, epoch), feature_extractor, class_classifier, domain_classifier)
            
    return feature_extractor, class_classifier, domain_classifier

def dann_performance_test(source, target, epochs, threshold, lr, weight_decay):
    #-----------------------------------------------------
    # Create Model, optimizer, scheduler, and loss function
    #------------------------------------------------------
    feature_extractor = Feature_Extractor().to(DEVICE)
    class_classifier  = Class_Classifier().to(DEVICE)
    domain_classifier = Domain_Classifier().to(DEVICE)

    optimizer = optim.Adam([
        {'params': feature_extractor.parameters()},
        {'params': class_classifier.parameters()},
        {'params': domain_classifier.parameters()}
        ], lr=lr, betas=(opt.b1, opt.b2), weight_decay=weight_decay
    )
    
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=0.1)
    
    class_criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
    domain_criterion = torch.nn.CrossEntropyLoss().to(DEVICE)

    #------------------
    # Create Dataloader
    #------------------
    source_black = True if source == 'usps' else False
    target_black = True if target == 'usps' else False
    source_train_set = dataset.NumberClassify("./hw3_data/digits", source, train=True, black=source_black, transform=transforms.ToTensor())
    target_train_set = dataset.NumberClassify("./hw3_data/digits", target, train=True, black=target_black, transform=transforms.ToTensor())
    source_test_set  = dataset.NumberClassify("./hw3_data/digits", source, train=False, black=source_black, transform=transforms.ToTensor())
    target_test_set  = dataset.NumberClassify("./hw3_data/digits", target, train=False, black=target_black, transform=transforms.ToTensor())
    print("Source_train: \t{}, {}".format(source, len(source_train_set)))
    print("Source_test: \t{}, {}".format(source, len(source_test_set)))
    print("Target_train: \t{}, {}".format(target, len(target_train_set)))
    print("Target_test: \t{}, {}".format(target, len(target_test_set)))
    
    source_train_loader = DataLoader(source_train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.threads)
    source_test_loader  = DataLoader(source_test_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.threads)
    target_train_loader = DataLoader(target_train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.threads)
    target_test_loader  = DataLoader(target_test_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.threads)
    
    for epoch in range(1, epochs + 1):
        scheduler.step()

        loss = []
        feature_extractor.train()
        class_classifier.train()
        domain_classifier.train()

        for index, (source_batch, target_batch) in enumerate(zip(source_train_loader, target_train_loader), 1):
            source_img, source_label, _ = source_batch
            target_img, target_label, _ = target_batch
            batch_size_src, batch_size_tgt = source_img.shape[0], target_img.shape[0]

            source_img   = source_img.to(DEVICE)
            source_label = source_label.type(torch.long).view(-1).to(DEVICE)
            target_img   = target_img.to(DEVICE)
            target_label = target_label.type(torch.long).view(-1).to(DEVICE)

            source_domain_labels = torch.zeros(batch_size_src).type(torch.long).to(DEVICE)
            target_domain_labels = torch.ones(batch_size_tgt).type(torch.long).to(DEVICE)
            
            constant = opt.alpha

            optimizer.zero_grad()

            source_feature = feature_extractor(source_img).view(-1, 128 * 7 * 7)
            target_feature = feature_extractor(target_img).view(-1, 128 * 7 * 7)
            
            source_class_predict  = class_classifier(source_feature)
            source_domain_predict = domain_classifier(source_feature, constant)
            target_domain_predict = domain_classifier(target_feature, constant)

            #---------------------------------------
            # Compute the loss
            # For the case of unsupervised learning:
            #   When source domain img:
            #     loss = class_loss + domain_loss
            #   When target domain img:
            #     loss = domain_loss
            # 
            #   Needs to maximize the domain_loss 
            #------------------------------------
            # 1. class loss
            class_loss = class_criterion(source_class_predict, source_label)
            
            # 2. Domain loss
            # print("Source_domain_predict.shape: \t{}".format(source_domain_predict.shape))
            # print("Source_domain_labels.shape: \t{}".format(source_domain_labels.shape))
            source_domain_loss = domain_criterion(source_domain_predict, source_domain_labels)
            target_domain_loss = domain_criterion(target_domain_predict, target_domain_labels)
            domain_loss = target_domain_loss + source_domain_loss

            # Final loss
            # loss = class_loss + constant * domain_loss
            # loss = constant * domain_loss
            # loss.backward()
            if epoch > 5: domain_loss.backward()
            if epoch <=5: class_loss.backward()
            optimizer.step()

            source_class_predict = source_class_predict.cpu().detach().numpy()
            source_label = source_label.cpu().detach().numpy()

            source_acc = np.mean(np.argmax(source_class_predict, axis=1) == source_label)

            if index % opt.log_interval == 0:
                print("[Epoch %d] [ %d/%d ] [src_acc: %d%%] [loss_Y: %f] [loss_D: %f]" % (
                        epoch, index, len(source_train_loader), 100 * source_acc, class_loss.item(), constant * domain_loss.item()))

def main():
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./models/dann", exist_ok=True)
    os.makedirs("./models/dann/{}".format(opt.tag), exist_ok=True)
    
    DOMAINS = [("usps", "mnistm"), ("mnistm", "svhn"), ("svhn", 'usps')]
    
    for SOURCE, TARGET in DOMAINS:
        
        if TARGET == "mnistm":
            threshold = 0.3
        elif TARGET == "svhn":
            threshold = 0.4
        elif TARGET == "usps":
            threshold = 0.3
        else:
            raise NotImplementedError

        domain_adaptation(SOURCE, TARGET, opt.epochs, threshold, lr=opt.lr, weight_decay=opt.weight_decay)

if __name__ == "__main__":
    os.system("clear")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs of training")
    parser.add_argument("--alpha", type=float, default=4.0, help="scalar of negative backpropagation")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight regularization")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--threads", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--tag", type=str, help="name of the model")
    parser.add_argument("--save_interval", type=int, default=10, help="interval epoch between everytime saving the model.")
    parser.add_argument("--log_interval", type=int, default=10, help="interval between everytime logging the training status.")
    
    opt = parser.parse_args()
    main()
