# ArtMiner
Pytorch implementation of Paper "Discovering Visual Patterns in Art Collections with Spatially-consistent Feature Learning"

[PDF](http://imagine.enpc.fr/~shenx/ArtMiner/artMiner2019CVPR.pdf) [WebPage](https://imagine.enpc.fr/~shenx/ArtMiner)



<p align="center">
<img src="https://github.com/XiSHEN0220/ArtMiner/blob/master/img/teaser.png" width="400px" alt="teaser">
</p>

If our project is helpful for your research, please consider citing : 
``` Bash
@inproceedings{shen2019discovery,
          title={{Discovering Visual Patterns in Art Collections with Spatially-consistent Feature Learning}},
          author={Xi, Shen and Alexei, A., Efros and Aubry, Mathieu},
          booktitle={Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
          year={2019}
        }
```
## Table of Content
* [Installation](#installation)
* [Single Shot Detection](#single-shot-detection)
* [Feature Learning](#feature-learning)
* [Discovery](#discovery)

## Installation

### Dependencies

The code can be used in **Linux** system with the the following dependencies: Python 2.7, Pytorch 0.3.0.post4, torchvision, tqdm, ujson, cv2, scipy, skimage

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

To download datasets (Brueghel + Ltll + Oxford) :
``` Bash
cd data
bash download_dataset.sh
```

To download pretrained model (Brueghel + Ltll + Oxford) :
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

The visual results will be saved into the visualDir that you indicate, some examples are shown below:

| | Query | Rank 1st | Rank 2nd | Rank 3rd | Rank 4th |
| --- | --- | --- | --- | --- | --- |
|ImageNet|![](https://github.com/XiSHEN0220/ArtMiner/blob/master/img/ssd/00.png) | ![](https://github.com/XiSHEN0220/ArtMiner/blob/master/img/ssd/11.jpg) | ![](https://github.com/XiSHEN0220/ArtMiner/blob/master/img/ssd/22.jpg) | ![](https://github.com/XiSHEN0220/ArtMiner/blob/master/img/ssd/33.jpg) | ![](https://github.com/XiSHEN0220/ArtMiner/blob/master/img/ssd/44.jpg) |
|Ours|![](https://github.com/XiSHEN0220/ArtMiner/blob/master/img/ssd/0.png) | ![](https://github.com/XiSHEN0220/ArtMiner/blob/master/img/ssd/1.jpg) | ![](https://github.com/XiSHEN0220/ArtMiner/blob/master/img/ssd/2.jpg) | ![](https://github.com/XiSHEN0220/ArtMiner/blob/master/img/ssd/3.jpg) | ![](https://github.com/XiSHEN0220/ArtMiner/blob/master/img/ssd/4.jpg) |


## Feature Learning

### Visualize Training Data
It is highly recommended to visualize the training data before the training.

One example is can be ran with : 
``` Bash
cd feature_learning/visualzation/
bash visBrueghelLtllOxford.sh
```
The examples saved into the output directory are shown below. <b>Red</b> / <b>Blue</b> / <b>Green</b> region indicates <b>Search</b> / <b>Validate</b> / <b>Train</b> region.

|![](https://github.com/XiSHEN0220/ArtMiner/blob/master/img/Brueghel_Rank1_1.jpg) | ![](https://github.com/XiSHEN0220/ArtMiner/blob/master/img/Brueghel_Rank1_2.jpg)|
|:---:|:---:|
| Brueghel Image 1 | Brueghel Image 2 |

|![](https://github.com/XiSHEN0220/ArtMiner/blob/master/img/Ltll_Rank1_1.jpg) | ![](https://github.com/XiSHEN0220/ArtMiner/blob/master/img/Ltll_Rank1_2.jpg)|
|:---:|:---:|
| Ltll Image 1 | Ltll Image 2 |


### Train
To train on Brueghel / Ltll / Oxford dataset :
``` Bash
cd feature_learning/
bash brughel.sh 
bash ltll.sh
bash oxford.sh
```

To train on your own dataset, please refer to:
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

|![](https://github.com/XiSHEN0220/ArtMiner/blob/master/discovery/FeatImageNet1.png) | ![](https://github.com/XiSHEN0220/ArtMiner/blob/master/discovery/FeatImageNet2.png)|
|:---:|:---:|
| Discovery Image 1 with ImageNet Feature| Discovery Image 2 ImageNet Feature|
|![](https://github.com/XiSHEN0220/ArtMiner/blob/master/discovery/FeatBrueghel1.png) | ![](https://github.com/XiSHEN0220/ArtMiner/blob/master/discovery/FeatBrueghel2.png)|
| Discovery Image 1 with Brueghel Feature| Discovery Image 2 Brueghel Feature|

### Ltll

To get classification results on Ltll : 
``` Bash
cd discovery
bash demo_ltll.sh
```
You should obtain the results in the table 2 in the paper, note that there is RANSAC in the algorithm, but we find very small variation by setting number of iterations to 1000 : 

| Feature | Disovery |
| :------: | :------: |
| ImageNet | 80.9 |
| Ours (trained on LTLL) | 88.5 |

### Oxford
To get retrieval results on Oxford5K : 
``` Bash
cd discovery
bash demo_oxford.sh
```
You should obtain the results in the table 2 in the paper : 

| Feature | Disovery |
| :------: | :------: |
| ImageNet | 85.0 |
| Ours (trained on Oxford) | 85.7 |
