"""
  FileName     [ t-SNE.py ]
  PackageName  [ DLCV Spring 2019 - DANN ]
  Synopsis     [ t-SNE implementation ]
"""
import argparse
import os

import numpy as np
import pandas as pd
import PIL
import sklearn
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torchvision import transforms

import dataset
import utils
from TransferLearning.adda import Classifier, Feature
from TransferLearning.dann import (Class_Classifier, Domain_Classifier,
                                   Feature_Extractor)

DEVICE = utils.selectDevice()

def tsne(x, y, n, perpexity, source, target):
    """
    T-SNE 

    Parameters
    ----------
    x, y, n

    perpexity : float

    source, target : str
    """
    x = TSNE(n_components=2, perplexity=perpexity).fit_transform(x)
    
    plt.clf()
    plt.figure(figsize=(12.8, 7.2))
    plt.plot(x[0], x[1], 'ro', linewidth=1)
    plt.plot(y[0], y[1], 'bo', linewidth=1)
    plt.title("[T-SNE - Train {} Test {}] vs Epoch(s)".format(source, target))
    plt.savefig("t-sne_{}_{}".format(source, target))
    plt.close()

    return

def main():
    os.system("clear")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256, help="Images to read for every iteration")
    parser.add_argument("--perpexity", type=float, help="the perpexity of the t-SNE algorithm")
    parser.add_argument("--threads", type=int, default=8, help="Number of cpu threads to use during batch generation")
    parser.add_argument("--adda", action="store_true")
    parser.add_argument("--dann", action="store_true")
    opt = parser.parse_args()
    
    paths   = {}
    DOMAINS = [("usps", "mnistm"), ("mnistm", "svhn"), ("svhn", 'usps')]

    #----------------------------
    # ADDA Model
    #----------------------------
    if opt.adda:
        source_encoder = Feature()
        target_encoder = Feature()
        classifier = Classifier(128 * 7 * 7, 1000, 10)

    if opt.dann:
        feature_exatractor = Feature_Extractor()
        class_classifier = Class_Classifier()

        for SOURCE, TARGET in DOMAINS:
            sourcepath = "./hw3_data/digits/{}/".format(SOURCE)
            targetpath = "./hw3_data/digits/{}/".format(TARGET) 
            # sourcecsv  = os.path.join(sourcepath, "test.csv")
            # targetcsv  = os.path.join(targetpath, "test.csv")
            # modelpath  = os.path.join("./models/dann/20190504/ADDA_{}.pth".format(TARGET))    

            paths[(SOURCE, TARGET)].append((sourcepath, targetpath))     

        # Read model
        for (source, target) in paths:
            sourcepath, targetpath = paths[(source, target)]
            feature_extractor, class_classifier, _ = utils.loadDANN(modelpath, Feature_Extractor(), Class_Classifier(), Domain_Classifier())
            
            source_set = dataset.NumberClassify(sourcepath, train=True, black=False)

            # Random pick 20 images for each class.
            # s_df = pd.read_csv(sourcecsv)
            # t_df = pd.read_csv(targetcsv)

            imgs = []
            labels = []

            #----------------
            # Source Domain
            #----------------
            for label in range(0, 10):
                # df = s_df[s_df["label"] == label].sample(n=20)
                
                for index, (img_name, _) in df.iterrows():
                    img = PIL.Image.open(img_name)
                    imgs.append(img)
                    labels.append(label)

            x1 = transforms.ToTensor()(imgs)
            y1 = torch.Tensor(labels, dtype=torch.float)

            x1 = source_encoder(x1).view(-1, 128 * 7 * 7)

            imgs = []
            labels = []

            #----------------
            # Target Domain
            #----------------
            for label in range(0, 10):
                df = t_df[t_df["label"] == label].sample(n=20)
                
                for index, (img_name, _) in df.iterrows():
                    img = PIL.Image.open(img_name)
                    imgs.append(img)
                    labels.append(label)

            x2 = transforms.ToTensor()(imgs)
            y2 = torch.Tensor(labels, dtype=torch.float)

            x2 = source_encoder(x2).view(-1, 128, 7, 7)

            # Draw tsne
            tsne(x1, x2, n=2, perpexity=opt.perpexity, source=source, target=target)

    #----------------------------
    # DANN Model
    #----------------------------
    if opt.dann:
        feature_extractor = Feature_Extractor()
        class_classifier  = Class_Classifier()
        domain_classifier = Domain_Classifier()

        for SOURCE, TARGET in DOMAINS:
            sourcepath = "./hw3_data/digits/{}".format(SOURCE)
            targetpath = "./hw3_data/digits/{}".format(TARGET) 
            sourcecsv  = os.path.join(sourcepath, "test.csv")
            targetcsv  = os.path.join(targetpath, "test.csv")
            modelpath  = os.path.join("./models/dann/20190504/ADDA_{}.pth".format(TARGET))    

            paths[(SOURCE, TARGET)].append((sourcepath, targetpath, sourcecsv, targetcsv, modelpath))    
        
        for (source, target) in paths:
            sourcepath, targetpath, sourcecsv, targetcsv, modelpath = paths[(source, target)]
            feature_extractor, class_classifier, _ = utils.loadDANN(modelpath, feature_extractor, class_classifier, domain_classifier)
            
            # Random pick 20 images for each class.
            s_df = pd.read_csv(sourcecsv)
            t_df = pd.read_csv(targetcsv)

            # Draw tsne
            tsne(x, y, n=2, perpexity=opt.perpexity, source=source, target=target)

if __name__ == "__main__":
    main()
