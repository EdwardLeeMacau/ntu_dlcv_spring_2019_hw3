"""
  Filename    [ utils.py ]
  PackageName [ DLCVSpring 2019 ]
  Synposis    [ ]
"""
import sys

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch import nn, optim

faceFeatures = [
    "Bangs", "Big_Lips", "Black_Hair", "Blond_Hair", "Brown_Hair", 
    "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Smiling", 
    "Straight_Hair", "Wavy_Hair", "Wearing_Lipstick"
]

def set_optimizer_lr(optimizer, lr):
    """ set the learning rate in an optimizer, without rebuilding the whole optimizer """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return optimizer

def selectDevice():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    return device

def saveCheckpoint(checkpoint_path, model, optimizer, scheduler, epoch):
    state = {
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'epoch': epoch,
        'scheduler': scheduler.state_dict()
    }
    torch.save(state, checkpoint_path)

    return 

def loadCheckpoint(checkpoint_path: str, model: nn.Module, optimizer: optim, scheduler: optim.lr_scheduler.MultiStepLR):
    state = torch.load(checkpoint_path)
    resume_epoch = state['epoch']
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])

    return model, optimizer, resume_epoch, scheduler

def saveModel(checkpoint_path: str, model: nn.Module):
    state = {
        'state_dict': model.state_dict(),
    }
    torch.save(state, checkpoint_path)

    return 

def loadModel(checkpoint_path: str, model: nn.Module):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    
    return model

def checkpointToModel(checkpoint_path: str, model_path: str):
    state = torch.load(checkpoint_path)

    newState = {
        'state_dict': state['state_dict']
    }

    torch.save(newState, model_path)

    return

def saveADDA(checkpoint_path: str, source_encoder, target_encoder, classifier):
    state = {
        'source_encoder': source_encoder.state_dict(),
        'target_encoder': target_encoder.state_dict(),
        'classifier': classifier.state_dict(),
    }
    torch.save(state, checkpoint_path)

    return

def loadADDA(checkpoint_path: str, source_encoder, target_encoder, classifier):
    state = torch.load(checkpoint_path)
    print(state.keys())
    
    source_encoder.load_state_dict(state['source_encoder'])
    target_encoder.load_state_dict(state['target_encoder'])
    classifier.load_state_dict(state['classifier'])
    print('Model loaded from %s' % checkpoint_path)

    return source_encoder, target_encoder, classifier

def saveDANN(checkpoint_path: str, feature_extractor, class_classifier, domain_classifier):
    state = {
        'feature_extractor': feature_extractor.state_dict(),
        'class_classifier': class_classifier.state_dict(),
        'domain_classifier': domain_classifier.state_dict(),
    }
    torch.save(state, checkpoint_path)

def loadDANN(checkpoint_path: str, feature_extractor, class_classifier, domain_classifier):
    state = torch.load(checkpoint_path)
    print(state.keys())
    
    feature_extractor.load_state_dict(state['feature_extractor'])
    class_classifier.load_state_dict(state['class_classifier'])
    domain_classifier.load_state_dict(state['domain_classifier'])
    print('Model loaded from %s' % checkpoint_path)

    return feature_extractor, class_classifier, domain_classifier

def loadAB(checkpoint_path: str, feature_extractor, class_classifier):
    state = torch.load(checkpoint_path)
    
    feature_extractor.load_state_dict(state['feature_extractor'])
    class_classifier.load_state_dict(state['class_classifier'])
    print('Model loaded from %s' % checkpoint_path)

    return feature_extractor, class_classifier
