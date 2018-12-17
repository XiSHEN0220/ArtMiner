
# coding: utf-8

import sys
sys.path.append("..")
from model.model import Model


import torch
import os
import torch.nn.functional as F

from torchvision import transforms
import numpy as np 

import outils
from tqdm import tqdm
import ujson
import argparse 

parser = argparse.ArgumentParser()

parser.add_argument(
	'--outDir', type=str , help='output model directory')

##---- Loss Parameter ----####

parser.add_argument(
	'--tripleLossThreshold', type=float , default = 0.6, help='threshold for triple loss')
		
parser.add_argument(
	'--topKLoss', type=int, default= 20, help='topk loss as negative loss')

##---- Search, Train, Validate Region ----####

parser.add_argument(
	'--searchRegion', type=int, default=2, help='feat size')
	
parser.add_argument(
	'--trainRegion', type=int, choices=[2, 4, 6, 8, 10, 12, 14], default = 12, help='train region, 2, 4, 6, 8, 10, 12, 14')
	
parser.add_argument(
	'--validRegion', type=int, default= 10, help='validation region')

##---- Training parameters ----####

parser.add_argument(
	'--imagenetFeatPath', type=str, default='../../pre-trained-models/resnet18.pth', help='imageNet feature model weight path')

parser.add_argument(
	'--finetunePath', type=str, help='finetune model weight path')

parser.add_argument(
	'--searchDir', type=str, default= '../data/Brueghel/', help='searching directory')
	
parser.add_argument(
	'--margin', type=int, default= 5, help='margin, the feature describing the border part is not taken into account')
	
parser.add_argument(
	'--nbEpoch', type=int , default = 600, help='Number of training epochs')

parser.add_argument(
	'--lr', type=float , default = 1e-5, help='learning rate')
	
parser.add_argument(
	'--nbImgEpoch', type=int , default = 200, help='how many images for each epoch')
	
parser.add_argument(
	'--batchSize', type=int , default = 4, help='batch size')
	
parser.add_argument(
	'--cuda', action='store_true', help='cuda setting')

args = parser.parse_args()
tqdm.monitor_interval = 0
print args


## Dataset, Minimum dimension, Total patch during the training
imgList = sorted(os.listdir(args.searchDir))
nbPatchTotal = 2000
imgFeatMin = args.searchRegion + 2 * args.margin + 1 ## Minimum dimension of feature map in a image 
iterEpoch = args.nbImgEpoch / args.batchSize
msg = '\n\nAlgo Description : \n\n In each Epoch, \n\t1. {:d} {:d}X{:d} features are utilized to search candidate regions; \n\t2. we validate on the outermost part in {:d}X{:d} region; \n\t3. We train on 4 corners in the {:d}X{:d} region for the top {:d} pairs; \n\t4. Batch size is {:d}, thus each epoch we do {:d} update. \n\n'.format(nbPatchTotal, args.searchRegion, args.searchRegion, args.validRegion, args.validRegion, args.trainRegion, args.trainRegion, args.nbImgEpoch, args.batchSize, iterEpoch)
print msg


## ImageNet Pre-processing
transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
				std = [ 0.229, 0.224, 0.225 ]),
])


## Model Initialize
strideNet = 16
featChannel = 256
net = Model(args.imagenetFeatPath, args.finetunePath)
if args.cuda:
	net.cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.5, 0.999))


## Scales
featScaleBase = 20 
nbOctave = 2 
scalePerOctave = 3
scales = outils.get_scales(featScaleBase, nbOctave, scalePerOctave)
msg = 'We search to match in {:d} scales, the max dimensions in the feature maps are:'.format(len(scales))
print msg
print scales
print '\n\n'


## Output
if not os.path.exists(args.outDir) : 
	os.mkdir(args.outDir)
history = {'posLoss':[], 'negaLoss':[]}
outHistory = os.path.join(args.outDir, 'history.json')

## Main Loop
for i_ in range(args.nbEpoch) : 
	logPosLoss = []
	logNegaLoss = []
	
	print 'Training Epoch {:d}'.format(i_)
	print '---> Get query...'
	net.eval()
	featQuery = outils.RandomQueryFeat(nbPatchTotal, featChannel, args.searchRegion, imgFeatMin, strideNet, transform, net, args.searchDir, args.margin, imgList, args.cuda)
	
	print '---> Get top10 patches matching to query...'
	topkImg, topkScale, topkValue, topkW, topkH = outils.RetrievalRes(nbPatchTotal, imgList, args.searchDir, args.margin, args.searchRegion, scales, strideNet, transform, net, featQuery, args.cuda)
	
	print '---> Get training pairs...'
	posPair, _ = outils.TrainPair(nbPatchTotal, args.searchDir, imgList, topkImg, topkScale, topkW, topkH, transform, net, args.margin, args.cuda, featChannel, args.searchRegion, args.validRegion, args.nbImgEpoch, strideNet)
	
	## form mini-batchs 
	posPairEpoch = outils.DataShuffle(posPair, args.batchSize)
	
	## Calculate Loss
	net.train() # switch to train mode
	for j_ in range(iterEpoch) : 
		optimizer.zero_grad()
		posSimilarityBatch = []
		negaSimilarityBatch = []
		
		for k_ in range(args.batchSize) : 
			posSimilarity, negaSimilarity = outils.PosNegaSimilarity(posPair, posPairEpoch[j_, k_], topkImg, topkScale, topkW, topkH, args.searchDir, imgList, strideNet, net, transform, args.searchRegion, args.trainRegion, args.margin, featChannel, args.cuda, args.topKLoss)
			posSimilarityBatch = posSimilarityBatch + posSimilarity
			negaSimilarityBatch = negaSimilarityBatch + negaSimilarity
		posSimilarityBatch = torch.cat(posSimilarityBatch, dim=0)
		negaSimilarityBatch = torch.cat(negaSimilarityBatch, dim=0)
		## Triplet Loss
		#loss = torch.clamp(negaSimilarityBatch - posSimilarityBatch + args.tripleLossThreshold, min=0)
		loss = torch.clamp(negaSimilarityBatch  + args.tripleLossThreshold - 1, min=0) + torch.clamp(args.tripleLossThreshold - posSimilarityBatch, min=0)
		
		## make sure that gradient is not zero
		if (loss > 0).any() : 
			loss = loss.mean()
			loss.backward()
			optimizer.step()

		logPosLoss.append( posSimilarityBatch.mean().data[0] )
		logNegaLoss.append( negaSimilarityBatch.mean().data[0] )

	# Save model, training history; print loss
	msg = 'EPOCH {:d}, positive pairs similarity: {:.4f}, negative pairs similarity: {:.4f}'.format(i_, np.mean(logPosLoss), np.mean(logNegaLoss)) 
	print msg
	history['posLoss'].append(np.mean(logPosLoss))
	history['negaLoss'].append(np.mean(logNegaLoss))

	if i_ % (4000 / args.nbImgEpoch) == (4000 / args.nbImgEpoch - 1) : 
		outModelPath = os.path.join(args.outDir, 'epoch{:d}'.format(i_))
		torch.save(net.state_dict(), outModelPath)
		with open(outHistory, 'w') as f :
			ujson.dump(history, f) 
		
		
	


