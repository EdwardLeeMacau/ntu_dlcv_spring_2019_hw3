"""
  Filename    [ statistics.py ]
  PackageName [ DLCVSpring2019 - GAN / DANN ]
  Synposis    [ Statistic Information of dataset ]
"""

from collections import Counter
import os
import pandas as pd

def count_face_feature(detpath):
    """ 
    Count the number of bbox for each categories in dataset

    Parameters
    ----------
    detpath : str

    Return
    ------
    statistics : df.Series
        Statistics of the dataset.
    """
    df = pd.read_csv(detpath)
    df = df.set_index('image_name')

    return df.sum(axis=0)

def count_digit(detpath):
    """ 
    Count the number of bbox for each categories in dataset

    Parameters
    ----------
    detpath : str

    Return
    ------
    statistics : df.Series
        Statistics of the dataset.
    """
    counter = Counter()
    
    df = pd.read_csv(detpath)
    df = df.set_index('image_name')
    counter.update(df['label'].values.tolist())

    return counter

def main():
    num = count_face_feature(os.path.join("hw3_data", "face", "train.csv"))
    print(num)

    for digitCase in ('mnistm', 'svhn', 'usps'):
        counter = count_digit(os.path.join("hw3_data", "digits", digitCase, "train.csv"))
        print(digitCase, counter)

if __name__ == "__main__":
    main()
