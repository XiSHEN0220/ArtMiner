# ArtMiner
Pytorch implementation of Paper "Discovering Visual Patterns in Art Collections with Spatially-consistent Feature Learning"

## Table of Content
* [Brueghel Dataset](#brueghel-dataset)
* [Feature Learning](#feature-learning)
	* [Visualize Training Data](visualize-training-data)
	* [Train](visualize-training-data)
	* [Pretrained Model](pretrained-model)

* [Single Shot Detection](#single-shot-detection)
* [Discovery](#discovery)

## Brueghel Dataset
The whole Brueghel dataset contains **1587** images, the images and bounding box annotations can be found in [here](www).

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

### Single Shot Detection

We also release our single shot detection code in [single_shot_detection directory](https://github.com/XiSHEN0220/ArtMiner/tree/master/single_shot_detection)
To use it, please refer to : 
Please refer to 
``` Bash
cd single_shot_detection
python retrieval.py --help
```



