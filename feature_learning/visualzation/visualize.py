import sys
sys.path.append("../..")
sys.path.append("..")
from model.model import Model


import torch
import os
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms
import numpy as np

import outils
from tqdm import tqdm
import ujson
import argparse
from itertools import product
def BboxFromFeatPos(featW, featH, w, h, posW, posH) :

	left = posH / featH * w
	right =  (posH + 1) / featH * w
	top = posW / featW * h
	bottom =  (posW + 1) / featW * h

	return map(int, [left, top, right, bottom])

def DrawImg(searchDir, imgList, topkImg, topkScale, topkW, topkH, queryIndex, pairIndex, minNet, strideNet, searchRegion, trainRegion, validRegion, margin, saveSize):

	I = Image.open(os.path.join(searchDir, imgList[topkImg[queryIndex, pairIndex]])).convert('RGB')
	w,h = I.size
	new_w, new_h = outils.ResizeImg(margin * 2 + searchRegion + 1, topkScale[queryIndex, pairIndex], minNet, strideNet, w, h)
	featW, featH = new_h / float(strideNet), new_w /  float(strideNet) ## Dimension are inverse in PIL Image and tensor feature
	drw = ImageDraw.Draw(I, 'RGBA')

	## Train region, Red Color
	for i,j in product(range(searchRegion), range(searchRegion)) :
		bbox = BboxFromFeatPos(featW, featH, w, h, topkW[queryIndex, pairIndex] + i, topkH[queryIndex, pairIndex] + j)
		drw.rectangle(xy = bbox, fill=(255, 0, 0, 125), outline=None)

	## Valid region, Blue Color
	validPosW, validPosH = topkW[queryIndex, pairIndex] - (validRegion + 1) / 2 + 1, topkH[queryIndex, pairIndex] - (validRegion + 1) / 2 + 1
	for i,j in product(range(validRegion), range(validRegion)) :
		if i == 0 or i == validRegion - 1 or j == 0 or j == validRegion - 1 :
			bbox = BboxFromFeatPos(featW, featH, w, h, validPosW + i, validPosH + j)
			drw.rectangle(xy = bbox, fill=(0, 0, 255, 125), outline=None)

	## Train region, Green Color
	trainPosW, trainPosH = topkW[queryIndex, pairIndex] - (trainRegion + 1) / 2 + 1, topkH[queryIndex, pairIndex] - (trainRegion + 1) / 2 + 1
	bbox = BboxFromFeatPos(featW, featH, w, h, trainPosW, trainPosH)
	drw.rectangle(xy = bbox, fill=(0, 255, 0, 125), outline=None)
	bbox = BboxFromFeatPos(featW, featH, w, h, trainPosW, trainPosH + trainRegion - 1)
	drw.rectangle(xy = bbox, fill=(0, 255, 0, 125), outline=None)
	bbox = BboxFromFeatPos(featW, featH, w, h, trainPosW + trainRegion - 1, trainPosH)
	drw.rectangle(xy = bbox, fill=(0, 255, 0, 125), outline=None)
	bbox = BboxFromFeatPos(featW, featH, w, h, trainPosW + trainRegion - 1, trainPosH + trainRegion - 1)
	drw.rectangle(xy = bbox, fill=(0, 255, 0, 125), outline=None)

	saveW, saveH = int(w / (max(w, h) / float(saveSize))), int(h / (max(w, h) / float(saveSize)))
	I.resize((saveW, saveH))
	return I

def Visualize(outDir, sample, searchRegion, trainRegion, validRegion, searchDir, imgList, topkImg, topkScale, topkW, topkH, saveSize, margin, minNet, strideNet):
	nbSample = len(sample)
	for i in tqdm(range(nbSample)) :
		pair = sample[i]
		queryIndex = int(pair[0])
		pairIndex = [int(pair[1]), int(pair[2])]

		I1 = DrawImg(searchDir, imgList, topkImg, topkScale, topkW, topkH, queryIndex, pairIndex[0], minNet, strideNet, searchRegion, trainRegion, validRegion, margin, saveSize)
		I2 = DrawImg(searchDir, imgList, topkImg, topkScale, topkW, topkH, queryIndex, pairIndex[1], minNet, strideNet, searchRegion, trainRegion, validRegion, margin, saveSize)

		out1, out2 = os.path.join(outDir, 'Rank{:d}_1.jpg'.format(i)), os.path.join(outDir, 'Rank{:d}_2.jpg'.format(i))
		I1.save(out1)
		I2.save(out2)



parser = argparse.ArgumentParser()

parser.add_argument(
	'--outDir', type=str , help='output image directory')

##---- Search, Train, Validate Region ----####

parser.add_argument(
	'--searchRegion', type=int, default=2, help='feat size')

parser.add_argument(
	'--trainRegion', type=int, default = 12, help='train region, 2, 4, 6, 8, 10, 12, 14')

parser.add_argument(
	'--validRegion', type=int, default= 10, help='validation region')

##---- Training parameters ----####
parser.add_argument(
	'--modelPath', type=str, help='model weight path')

parser.add_argument(
	'--searchDir', type=str, default= '../../data/Brueghel/', help='searching directory')

parser.add_argument(
	'--margin', type=int, default= 5, help='margin, the feature describing the border part is not taken into account')

parser.add_argument(
	'--nbImgEpoch', type=int , default = 200, help='how many images for each epoch')

parser.add_argument(
	'--cuda', action='store_true', help='cuda setting')

parser.add_argument(
	'--saveSize', type=int , default = 256, help='final image size')

parser.add_argument(
	'--nbSearchImgEpoch', type=int, default = 2000, help='maximum number of searching image in one epoch')
	
parser.add_argument(
	'--featScaleBase', type=int, default = 20, help='number of features in the max dimension of the minimum scale')

parser.add_argument(
	'--nbOctave', type=int, default = 2, help='number of octave')

parser.add_argument(
	'--scalePerOctave', type=int, default = 3, help='number of scales / octave')

parser.add_argument(
	'--queryScale', type=int, nargs='+', default = [68, 60, 49, 40, 36], help='query image scale')

args = parser.parse_args()
tqdm.monitor_interval = 0
print args

## Dataset, Minimum dimension, Total patch during the training
imgList = sorted(os.listdir(args.searchDir))
nbPatchTotal = args.nbSearchImgEpoch
imgFeatMin = args.searchRegion + 2 * args.margin + 1 ## Minimum dimension of feature map in a image
msg = '\n\nVisualizing Training data : \n\n In each Epoch, \n\t1. {:d} {:d}X{:d} features are utilized to search candidate regions; \n\t2. we validate on the outermost part in {:d}X{:d} region; \n\t3. We train on 4 corners in the {:d}X{:d} region for the top {:d} pairs; \n\t4. We visualize the training data for one epoch: {:d} patches in {:d} image. Image will be saved into {}, ranked by validation vote. \n\n'.format(nbPatchTotal, args.searchRegion, args.searchRegion, args.validRegion, args.validRegion, args.trainRegion, args.trainRegion, args.nbImgEpoch, args.nbImgEpoch * 4, args.nbImgEpoch, args.outDir)
print msg


## ImageNet Pre-processing
transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
				std = [ 0.229, 0.224, 0.225 ]),
])


## Model Initialize
strideNet = 16
minNet = 15
featChannel = 256
net = Model(args.modelPath)
if args.cuda:
	net.cuda()


## Scales

scales = outils.ScaleList(args.featScaleBase, args.nbOctave, args.scalePerOctave)
msg = 'We search to match in {:d} scales, the max dimensions in the feature maps are:'.format(len(scales))
print msg
print scales
print '\n\n'


## Output
if not os.path.exists(args.outDir) :
	os.mkdir(args.outDir)

## Main Loop
print '---> Get query...'
net.eval()

if len(imgList) <= args.nbSearchImgEpoch : 
	searchImgList = imgList  
else :
	index = np.random.permutation(np.arange(len(imgList)))[:args.nbSearchImgEpoch]
	searchImgList = [imgList[i] for i in index]
		
featQuery = outils.RandomQueryFeat(nbPatchTotal, featChannel, args.searchRegion, imgFeatMin, minNet, strideNet, transform, net, args.searchDir, args.margin, searchImgList, args.cuda, args.queryScale)

print '---> Get top10 patches matching to query...'
topkImg, topkScale, topkValue, topkW, topkH = outils.RetrievalRes(nbPatchTotal, searchImgList, args.searchDir, args.margin, args.searchRegion, scales, minNet, strideNet, transform, net, featQuery, args.cuda)


print '---> Get training pairs...'
posPair, _ = outils.TrainPair(nbPatchTotal, args.searchDir, searchImgList, topkImg, topkScale, topkW, topkH, transform, net, args.margin, args.cuda, featChannel, args.searchRegion, args.validRegion, args.nbImgEpoch, minNet, strideNet)


Visualize(args.outDir, posPair, args.searchRegion, args.trainRegion, args.validRegion, args.searchDir, searchImgList, topkImg, topkScale, topkW, topkH, args.saveSize, args.margin, minNet, strideNet)
