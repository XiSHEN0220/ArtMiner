# ArtMiner
Pytorch implementation of Paper "Discovering Visual Patterns in Art Collections with Spatially-consistent Feature Learning"

## Table of Content
* [Installation](#installation)
* [Single Shot Detection](#single-shot-detection)
* [Feature Learning](#feature-learning)
* [Discovery](#discovery)

## Installation

### Dependencies

The code can be used in **Linux** system with the the following dependencies: Python 2.7, Pytorch 0.3.0.post4, torchvision, tqdm, ujson, cv2, scipy

We recommend to utilize virtual environment to install all dependencies and test the code. One choice is [virtualenv](https://virtualenv.pypa.io/en/latest/).

To install pytorch 0.3.0.post4 + cuda 8.0 (For other cuda version (9.0, 7.5), the only modification is to change *cu80* to your cuda version):
``` Bash
pip install https://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp27-cp27mu-linux_x86_64.whl
```

To install other dependencies:
``` Bash
bash requirement.sh
```




### Dataset and Model

To download datasets (Brueghel + Ltll) :
``` Bash
cd data
bash download_dataset.sh
```

To download pretrained model :
``` Bash
cd model
bash download_models.sh
```


## Single Shot Detection

To test performance on single shot detection:
``` Bash
cd single_shot_detection
bash run_FeatImageNet.sh # ImageNet feature
bash run_FeatBrueghel.sh # Brueghel feature
```

You should obtain the results in the table 1,

| Feature | Cosine Similarity |
| :------: | :------: |
| ImageNet | 58.0 |
| Ours (trained on Brueghel) | 75.3 |

The visual results will be saved into visualDir that you indicate, some examples are shown below:

|:---:|:---:|:---:|:---:|:---:|:---:|
| | Query | Rank 1st | Rank 2nd | Rank 3rd | Rank 4th |

|ImageNet|![](https://github.com/XiSHEN0220/ArtMiner/blob/master/img/ssd/00.png) | ![](https://github.com/XiSHEN0220/ArtMiner/blob/master/img/ssd/11.jpg) | ![](https://github.com/XiSHEN0220/ArtMiner/blob/master/img/ssd/22.jpg) | ![](https://github.com/XiSHEN0220/ArtMiner/blob/master/img/ssd/33.jpg) | ![](https://github.com/XiSHEN0220/ArtMiner/blob/master/img/ssd/44.jpg) |

|Ours|![](https://github.com/XiSHEN0220/ArtMiner/blob/master/img/ssd/0.png) | ![](https://github.com/XiSHEN0220/ArtMiner/blob/master/img/ssd/1.jpg) | ![](https://github.com/XiSHEN0220/ArtMiner/blob/master/img/ssd/2.jpg) | ![](https://github.com/XiSHEN0220/ArtMiner/blob/master/img/ssd/3.jpg) | ![](https://github.com/XiSHEN0220/ArtMiner/blob/master/img/ssd/4.jpg) |


## Feature Learning

### Visualize Training Data
It is highly recommended to visualize the training data before the training.

Please refer to
``` Bash
cd feature_learning/visualzation/
python visualize.py --help
```
The examples saved into the output directory are shown below. <b>Red</b> / <b>Blue</b> / <b>Green</b> region indicates <b>Search</b> / <b>Validate</b> / <b>Train</b> region.

|![](https://github.com/XiSHEN0220/ArtMiner/blob/master/img/Brueghel_Rank1_1.jpg) | ![](https://github.com/XiSHEN0220/ArtMiner/blob/master/img/Brueghel_Rank1_2.jpg)|
|:---:|:---:|
| Brueghel Image 1 | Brueghel Image 2 |

|![](https://github.com/XiSHEN0220/ArtMiner/blob/master/img/Ltll_Rank1_1.jpg) | ![](https://github.com/XiSHEN0220/ArtMiner/blob/master/img/Ltll_Rank1_2.jpg)|
|:---:|:---:|
| Ltll Image 1 | Ltll Image 2 |


We also provide a script generating html table to visualize all pairs.

Please refer to:
``` Bash
cd feature_learning/visualzation/
python file2web.py --imgDir IMAGE_DIRECTORY_HERE --outHtml OUTPUT_HTML_HERE
```
### Train
To train on Brueghel dataset :
``` Bash
cd feature_learning/
bash brughel.sh
```
To train on LTLL dataset :
``` Bash
cd feature_learning/
bash ltll.sh
```

If you want to launch the training on your own dataset, indicates the image directory in *--searchDir* in *train.py*.
For more details, please refer to:
``` Bash
cd feature_learning/
python train.py --help
```



## Discovery

### Pair Discovery

To launch discovery between a pair of images, please utilize the script in *discovery/pair_discovery.py*.
One example of command is in *discovery/pair_discovery.sh* :
``` Bash
cd discovery
bash pair_discovery.sh
```

The results of discovery between the pair of images :

|![](https://github.com/XiSHEN0220/ArtMiner/blob/master/discovery/toto1.png) | ![](https://github.com/XiSHEN0220/ArtMiner/blob/master/discovery/toto2.png)|
|:---:|:---:|
| Discovery Image 1 | Discovery Image 2 |
