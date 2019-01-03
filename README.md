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
	* [Pair Discovery](#pair-discovery)

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

The official site of Ltll is [here](http://users.cecs.anu.edu.au/~basura/beeldcanon/).

We provide a fast download : 
* Image can be downloaded via 
``` Bash
cd data
bash download_ltll.sh
```
* Annotations are available in the dictionary *./data/ltll.json*, Validatation / Test are splitted by the key 'val' and 'test' in the dictionary. 



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

### Single Shot Detection

We also release our single shot detection code in [single_shot_detection directory](https://github.com/XiSHEN0220/ArtMiner/tree/master/single_shot_detection)
To utilize it, please refer to : 
``` Bash
cd single_shot_detection
python retrieval.py --help
```

### Discovery

## Pair Discovery 

To launch discovery between a pair of images, please utilize the script in *discovery/pair_discovery.py*. 
A command example is given in *discovery/pair_discovery.sh*, user need to modify parameter *imagenetFeatPath* and/or *finetunePath*, then running with :
``` Bash
cd discovery
bash pair_discovery.sh
```

The results of discovery between a pair of images : 

|![](https://github.com/XiSHEN0220/ArtMiner/blob/master/discovery/toto1.png) | ![](https://github.com/XiSHEN0220/ArtMiner/blob/master/discovery/toto2.png)|
|:---:|:---:|
| Discovery Image 1 | Discovery Image 2 |







