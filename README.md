# HW3 ― GAN, ACGAN and UDA

## Demonstration

### GAN

### ACGAN

## Performance

### Target: mnistm

Train parameters

|   Parameters   | Value  |
| :------------: | :----: |
|    $\alpha$    |  0.25  |
|   batchSize    |   64   |
|     epoch      |   30   |
| Source Dataset |  usps  |
| Target Dataset | mnistm |

Result 

|     name     | class accurate | class loss | domain accurate | domain loss |
| :----------: | :------------: | :--------: | :-------------: | :---------: |
|  usps_train  |                |            |                 |             |
|  usps_test   |     96.66%     |   0.1416   |     79.12%      |   0.5516    |
| mnistm_train |     35.24%     |   2.8489   |     39.43%      |   0.8002    |
| mnistm_test  |     35.80%     |   2.8391   |     39.42%      |   0.8069    |

Source_test:    usps, 2007
Target_test:    mnistm, 10000

### Target: svhn

Train parameters

|   Parameters   | Value  |
| :------------: | :----: |
|    $\alpha$    |  0.25  |
|   batchSize    |   64   |
|     epoch      |   15   |
| Source Dataset | mnistm |
| Target Dataset |  svhn  |

Source_test:    mnistm, 10000
Target_test:    svhn, 26032

|     name     | class accurate | class loss | domain accurate | domain loss |
| :----------: | :------------: | :--------: | :-------------: | :---------: |
| mnistm_train |                |            |                 |             |
| mnistm_test  |     97.74%     |   0.0673   |     99.75%      |   0.0237    |
|  svhn_train  |     44.31%     |   1.7580   |      0.30%      |   3.6742    |
|  svhn_test   |     49.95%     |   1.5604   |      0.13%      |   3.5836    |

### Target: usps

Train parameters

|   Parameters   | Value |
| :------------: | :---: |
|    $\alpha$    | 0.25  |
|   batchSize    |  64   |
|     epoch      |  18   |
| Source Dataset | svhn  |
| Target Dataset | usps  |

Source_test:    svhn, 26032
Target_test:    usps, 2007

|    name    | class accurate | class loss | domain accurate | domain loss |
| :--------: | :------------: | :--------: | :-------------: | :---------: |
| svhn_train |                |            |                 |             |
| svhn_test  |     82.77%     |   0.5907   |      1.04%      |   0.8784    |
| usps_train |     70.74%     |   0.9812   |     74.49%      |   0.6583    |
| usps_test  |     67.16%     |   1.1177   |     74.49%      |   0.6577    |

## How to use

1. Download dataset

    ```
    bash get_dataset.sh
    ```

2. Download pretrained model and inference for Problem 1 and Problem 2

    ```
    bash hw3_p1p2.sh
    ```

3. Download pretrained model and inference for Problem 3

    ```
    bash hw3_p3.sh
    ```

3. Download pretrained model and inference for Problem 4

    ```
    bash hw3_4.sh
    ```

## Requirements

## More Information

Please read [requirement](./REQUIREMENT.md) to get more information about this HW.

## Performance Report

### Problem 1 GAN

### Problem 2 ACGAN

### Problem 3 DANN

### Problem 4 Improved DANN
