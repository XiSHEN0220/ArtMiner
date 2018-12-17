# ArtMiner
Pytorch implementation of Paper "Discovering Visual Patterns in Art Collections with Spatially-consistent Feature Learning"

## Table of Content
* [Dependency](#dependency)
* [Dataset](#dataset)
	* [Brueghel](#brueghel)
	* [Large Time Lags Location(Ltll)](#large-time-lags-location(ltll))
* [Feature Learning](#feature-learning)
	* [Visualize Training Data](visualize-training-data)
	* [Train](visualize-training-data)
	* [Pretrained Model](pretrained-model)

* [Single Shot Detection](#single-shot-detection)
* [Discovery](#discovery)

## Dependency
The code can be used in **Linux** system with the below dependencies:
* Python 2.7
* [Pytorch 0.3.0.post4](https://pytorch.org/get-started/previous-versions/)
* torchvision
* Other dependencies: [tqdm](https://github.com/tqdm/tqdm), [ujson](https://pypi.org/project/ujson/)
 
## Dataset

### Brughel
The whole Brueghel dataset contains **1587** images : 
* Image can be downloaded via 
``` Bash
cd data
bash download_brueghel.sh
```
* Validatation / Test annotations are available in *./data/brueghelVal.json* and *./data/brueghelTest.json*

### Large Time Lags Location(Ltll)
The official site of Ltll is [here](http://users.cecs.anu.edu.au/~basura/beeldcanon/)
We provide a fast download via : 
* Image can be downloaded via 
``` Bash
cd data
bash download_ltll.sh
```


## Feature Learning

### Visualize Training Data
It is highly recommended to visualize the training data before the training. 
Please refer to 
``` Bash
cd feature_learning/visualzation/
python visualize.py --help
```
The examples saved into the output directory are shown below. <b>Red</b> / <b>Blue</b> / <b>Green</b> region indicates <b>Search</b> / <b>Validate</b> / <b>Train</b> region.
<p align="center">
<img src="https://github.com/XiSHEN0220/ArtMiner/blob/master/img/Brueghel_Rank1_1.jpg" width="420"> <img src="https://github.com/XiSHEN0220/ArtMiner/blob/master/img/Brueghel_Rank1_2.jpg" width="420"> 
</p>

<p align="center">
<img src="https://github.com/XiSHEN0220/ArtMiner/blob/master/img/Ltll_Rank1_1.jpg" width="420"> <img src="https://github.com/XiSHEN0220/ArtMiner/blob/master/img/Ltll_Rank1_2.jpg" width="420"> 
</p>


We also provide a script generating html table to visualize all pairs. 
Please refer to:
``` Bash
cd feature_learning/visualzation/
python file2web.py --help
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

## Single Shot Detection

We also release our single shot detection code in [single_shot_detection directory](https://github.com/XiSHEN0220/ArtMiner/tree/master/single_shot_detection)
To use it, please refer to : 
``` Bash
cd single_shot_detection
python retrieval.py --help
```




