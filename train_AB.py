"""
  FileName     [ train_AB.py ]
  PackageName  [ DLCVSpring2019 - GAN ]
  Synopsis     [ DANN training: Without DANN strategic ]

  Dataset:
    USPS: 28 * 28 * 1 -> 28 * 28 * 3
    SVHN: 28 * 28 * 3
    MNISTM: 28 * 28 * 3
"""

import argparse
import datetime
import os
import random

import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import dataset
import utils
from dann import Class_Classifier, Domain_Classifier, Feature_Extractor

# Set as true when the I/O shape of the model is fixed
cudnn.benchmark = True
DEVICE = utils.selectDevice()

def train(feature_extractor, class_classifier, source_train_loader, optim, epoch, class_criterion):
    """ Train each epoch with default framework. """
    loss = []
    feature_extractor.train()
    class_classifier.train()

    for index, source_batch in enumerate(source_train_loader, 1):
        #-------------------------------
        # Prepare the images and labels
        #-------------------------------
        source_img, source_label, _ = source_batch
        source_img, source_label = source_img.to(DEVICE), source_label.view(-1).type(torch.long).to(DEVICE)
        batch_size = len(source_label)
        # print("Label.shape: \t{}".format(source_label.shape))
        # print(source_label)

        #-------------------------------
        # Setup optimizer
        #   Dynamic adjust the learning rate with parameter p
        #-------------------------------
        # optim = utils.set_optimizer_lr(optim, p)
        optim.zero_grad()

        #-------------------------------
        # Get features, class pred, domain pred:
        #-------------------------------
        source_feature = feature_extractor(source_img).view(batch_size, -1)
        class_predict  = class_classifier(source_feature)

        #-------------------------------
        # Compute the accuracy, loss
        #-------------------------------
        # print(class_predict.type())
        # print(source_label.type())
        # print(class_predict.shape)
        # print(source_label.shape)
        loss = class_criterion(class_predict, source_label)
        loss.backward()
        optim.step()

        class_predict = class_predict.cpu().detach().numpy()
        source_label  = source_label.cpu().detach().numpy()

        source_acc = np.mean(np.argmax(class_predict, axis=1) == source_label)

        if index % opt.log_interval == 0:
            print("[Epoch %d] [ %d/%d ] [src_acc: %d%%] [loss_Y: %f] [loss_D: N/A]" % (epoch, index, len(source_train_loader), 100 * source_acc, loss.item()))
    
    return feature_extractor, class_classifier

def val(feature_extractor, class_classifier, source_loader, target_loader, class_criterion):
    domain_loss, domain_accuracy = [], []
    feature_extractor.eval()
    class_classifier.eval()

    for loader in [source_loader, target_loader]:
        batch_acc = []
        batch_los = []

        #------------------------
        # Calculate the accuracy, loss
        #------------------------
        for _, (img, label, _) in enumerate(loader, 1):
            batch_size = len(label)
            img, label = img.to(DEVICE), label.type(torch.long).view(-1).to(DEVICE)
            feature    = feature_extractor(img).view(batch_size, -1)
            class_pred = class_classifier(feature)

            # Loss
            loss = class_criterion(class_pred, label)
            batch_los.append(loss.item())

            # Accuracy
            class_pred = class_pred.cpu().detach().numpy()
            label = label.cpu().detach().numpy()
            acc = np.mean(np.argmax(class_pred, axis=1) == label)
            batch_acc.append(acc)

        acc = np.mean(batch_acc)
        domain_accuracy.append(acc)

        los = np.mean(batch_los)
        domain_loss.append(los)

    return domain_accuracy, domain_loss

def train_A_test_B(source, target, epochs, lr, weight_decay):
    """ 
      Train the model with source data. Test the model with target data. 
    """ 
    #-----------------------------------------------------
    # Create Model, optimizer, scheduler, and loss function
    #------------------------------------------------------
    feature_extractor = Feature_Extractor().to(DEVICE)
    class_classifier  = Class_Classifier().to(DEVICE)

    optimizer = optim.Adam([{'params': feature_extractor.parameters()},
                            {'params': class_classifier.parameters()}], 
                            lr=lr, betas=(opt.b1, opt.b2), weight_decay=weight_decay)
    
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=0.1)
    class_criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
    
    source_black = True if source == 'usps' else False
    target_black = True if target == 'usps' else False

    source_train_set = dataset.NumberClassify("./hw3_data/digits", source, train=True, black=source_black, transform=transforms.ToTensor())
    source_test_set = dataset.NumberClassify("./hw3_data/digits", source, train=False, black=source_black, transform=transforms.ToTensor())
    target_train_set = dataset.NumberClassify("./hw3_data/digits", target, train=True, black=target_black, transform=transforms.ToTensor())
    target_test_set = dataset.NumberClassify("./hw3_data/digits", target, train=False, black=target_black, transform=transforms.ToTensor())
    print("Source_train: \t{}, {}".format(source, len(source_train_set)))
    print("Source_test: \t{}, {}".format(source, len(source_test_set)))
    print("Target_train: \t{}, {}".format(target, len(target_train_set)))
    print("Target_test: \t{}, {}".format(target, len(target_test_set)))
    source_train_loader = DataLoader(source_train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.threads)
    source_test_loader = DataLoader(source_test_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.threads)
    target_train_loader = DataLoader(target_train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.threads)
    target_test_loader = DataLoader(target_test_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.threads)

    src_acc = []
    src_los = []
    tgt_acc = []
    tgt_los = []

    for epoch in range(1, epochs + 1):
        scheduler.step()

        feature_extractor, class_classifier = train(feature_extractor, class_classifier, source_train_loader, optimizer, epoch, class_criterion)
        test_accuracy, test_loss = val(feature_extractor, class_classifier, source_test_loader, target_test_loader, class_criterion)
        print("[Epoch %d] [src_acc: %d%%] [tgc_acc: %d%%] [src_loss: %f] [tgc_loss: %f]" % (epoch, 100 * test_accuracy[0], 100 * test_accuracy[1], test_loss[0], test_loss[1]))

        # Tracing the accuracy
        src_acc.append(test_accuracy[0])
        src_los.append(test_loss[0])
        tgt_acc.append(test_accuracy[1])
        tgt_los.append(test_loss[1])
        
        x = np.arange(start=1, stop=epoch+1)
        plt.clf()
        plt.figure(figsize=(12.8, 7.2))
        plt.plot(x, src_acc, label='Source Accuracy', linewidth=1)
        plt.plot(x, tgt_acc, label='Target Accuracy', linewidth=1)
        plt.legend(loc=0)
        plt.xlabel("Epochs(s)")
        plt.title("[Acc - Train {} Test {}] vs Epoch(s)".format(source, target))
        plt.savefig("Train_A_Test_B_{}_{}-Acc.png".format(source, target))
        plt.close()

        plt.clf()
        plt.figure(figsize=(12.8, 7.2))
        plt.plot(x, src_los, label='Source Loss', linewidth=1)
        plt.plot(x, tgt_los, label='Target Loss', linewidth=1)
        plt.legend(loc=0)
        plt.xlabel("Epochs(s)")
        plt.title("[Loss - Train {} Test {}] vs Epoch(s)".format(source, target))
        plt.savefig("Train_A_Test_B_{}_{}-Loss.png".format(source, target))
        plt.close()

        with open('statistics.txt', 'a') as textfile:
            # textfile.write(datetime.datetime.now().strftime("%d, %b %Y %H:%M:%S"))
            # textfile.write(str(src_acc))
            # textfile.write(str(tgt_acc))
            textfile.write("Source: {}, Target: {}, Accuracy: {}\n".format(source, target, max(tgt_acc)))
            print("Source: {}, Target: {}, Accuracy: {}".format(source, target, max(tgt_acc)))

        # if epoch % opt.save_interval == 0:
        #     save("./models/dann/{}/Train_A_Test_B_{}_{}_{}.pth".format(opt.tag, source, target, epoch), feature_extractor, class_classifier)

    return feature_extractor, class_classifier

def save(checkpoint_path: str, feature_extractor, class_classifier):
    state = {
        'feature_extractor': feature_extractor.state_dict(),
        'class_classifier': class_classifier.state_dict(),
    }
    torch.save(state, checkpoint_path)

def load(checkpoint_path: str, feature_extractor, class_classifier):
    state = torch.load(checkpoint_path)
    
    feature_extractor.load_state_dict(state['feature_extractor'])
    class_classifier.load_state_dict(state['class_classifier'])
    print('Model loaded from %s' % checkpoint_path)

    return feature_extractor, class_classifier

def main():
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./models/dann", exist_ok=True)
    os.makedirs("./models/dann/{}".format(opt.tag), exist_ok=True)
    
    #----------------------------------------
    # Train the model without DANN strategic
    #   SOURCE -> TARGET
    #   TARGET -> TARGET
    #----------------------------------------
    DOMAINS = [("usps", "mnistm"), ("mnistm", "svhn"), ("svhn", 'usps')]
    for SOURCE, TARGET in DOMAINS:
        train_A_test_B(SOURCE, TARGET, opt.epochs, opt.lr, opt.weight_decay)
        train_A_test_B(TARGET, TARGET, opt.epochs, opt.lr, opt.weight_decay)

if __name__ == "__main__":
    os.system("clear")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs of training")
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
    print(opt)

    main()
