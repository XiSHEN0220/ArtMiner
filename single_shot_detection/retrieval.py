import sys
sys.path.append("..")
from model.model import Model

import os
import PIL.Image as Image
from torchvision import datasets, transforms, models

from tqdm import tqdm
from shutil import copyfile
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

## scripts
import eval
import outils
import feature

import ujson


import argparse

parser = argparse.ArgumentParser()


##---- Query Information ----####

parser.add_argument(
	'--labelJson', type=str, default = '../data/brueghelVal.json', help='label json file')

##---- Search Dataset Setting ----####
parser.add_argument(
	'--featScaleBase', type=int, default= 20, help='minimum # of features in the scale list ')

parser.add_argument(
	'--scalePerOctave', type=int, default= 3, help='# of scales in one octave ')

parser.add_argument(
	'--nbOctave', type=int, default= 2, help='# of octaves')


##---- Training parameters ----####

parser.add_argument(
	'--imagenetFeatPath', type=str, default='../../pre-trained-models/resnet18.pth', help='imageNet feature net weight path')

parser.add_argument(
	'--finetunePath', type=str, help='finetune net weight path')

parser.add_argument(
	'--searchDir', type=str, default= '../data/Brueghel/', help='searching directory')

parser.add_argument(
	'--margin', type=int, default= 5, help='margin, the feature describing the border part is not taken into account')

parser.add_argument(
	'--cuda', action='store_true', help='cuda setting')

parser.add_argument(
	'--cropSize', type=int , default = 0, help='Crop Size')

parser.add_argument(
	'--nbPred', type=int , default = 1000, help='nb of predcition')

parser.add_argument(
	'--nbDraw', type=int , default = 0, help='nb of draw image')

parser.add_argument(
	'--visualDir', type=str , default = None, help='output image directory')

parser.add_argument(
	'--queryFeatMax', type=int , default = 8, help='maximum feature in the query patch')

parser.add_argument(
	'--IoUThreshold', type=int , default = 0.3, help='IoU threshold')

parser.add_argument(
	'--detTxt', type=str , default = None, help='write detection results into text file?')

parser.add_argument(
	'--detJson', type=str , default = None, help='write detection results into json file?')

parser.add_argument(
	'--detMAPJson', type=str , default = None, help='write detection results (query mAP) into json file?')



args = parser.parse_args()
tqdm.monitor_interval = 0
print args


def ResDetection(res, searchDim, queryFeat, strideNet, cropSize, nbPred, label) :
	det = {}
	resFinal = {}
	for category in res.keys():

		if not det.has_key(category) :

			det[category] = []
			resFinal[category] = []

		for j, item in enumerate(res[category]):

			det[category].append([])
			kernelSize = queryFeat[category][j].size()[2:]
			queryName = label[category][j]['query'][0]
			queryBbox = label[category][j]['query'][1]

			bbs = []
			infos = []
			searchs = []

			for searchName in item.keys():

				infoFind = item[searchName]
				imgSize = searchDim[searchName]
				bb, infoFind = outils.FeatPos2ImgBB(infoFind, kernelSize, imgSize, strideNet, cropSize)

				bbs.append(bb)
				infos = infos + infoFind
				searchs = searchs + [searchName for i in range(len(bb))]

			## Aggregate all the results
			bbs = np.concatenate(bbs, axis=0)
			index = np.argsort(bbs[:, -1])[::-1]
			bbs = bbs[index]
			infos = [infos[i] for i in index]
			searchs = [searchs[i] for i in index]

			det[category][j] = [(searchs[i], bbs[i, -1], bbs[i, :4].astype(int)) for i in range(nbPred)]
			resFinal[category].append([(searchs[i], infos[i][0], infos[i][1], infos[i][2], infos[i][3]) for i in range(nbPred)])

	return det, resFinal


def Retrieval(searchDir,
			featMax,
			scaleList,
			strideNet,
			cropSize,
			cuda,
			transform,
			net,
			queryFeat,
			resDict,
			nbPred,
			label) :


	print 'Get search image dimension...'
	searchDim = feature.SearchImgDim(searchDir)

	for k, searchName in enumerate(tqdm(os.listdir(searchDir))) :

		searchFeatDict = feature.SearchFeat(searchDir, featMax, scaleList, strideNet, cuda, transform, net, searchName)

		for queryCategory in queryFeat.keys() :
			for j_, featQ in enumerate(queryFeat[queryCategory]) :

				wFind = []
				hFind = []
				scoreFind = []
				scaleFind = []

				for scaleName in searchFeatDict.keys() :

					featImg = searchFeatDict[scaleName]
					score = outils.Match(featImg, featQ, cuda)
					score, _ = score.max(dim = 1)
					w,h = score.size()[1], score.size()[2]
					score, _ = score.max(dim = 0)
					score, index = score.view(1, -1).topk(min(featMax, score.numel()))

					## Store results for each scale
					wFind.append(index/h)
					hFind.append(index%h)
					scoreFind.append(score)
					scaleFind = scaleFind + [int(scaleName) for i in range(len(score))]

				## Store results for each image
				wFind = torch.cat(wFind, dim=1)
				hFind = torch.cat(hFind, dim=1)
				scoreFind = torch.cat(scoreFind, dim=1)
				print wFind.size(), hFind.size(), scoreFind.size(), scoreFind.size() 
				_, indexKeep = torch.sort(scoreFind, descending = True)

				indexKeep = indexKeep[0, :min(5 * featMax, indexKeep.numel())]
				infoFind = [(wFind[0, i], hFind[0, i], scaleFind[i], scoreFind[0, i]) for i in indexKeep]
				res[queryCategory][j_][searchName] = infoFind


	det, res = ResDetection(res, searchDim, queryFeat, strideNet, cropSize, nbPred, label)

	return det, res




transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
				std = [ 0.229, 0.224, 0.225 ]),
])

## net Initialize
strideNet = 16
featChannel = 256
net = Model(args.imagenetFeatPath, args.finetunePath)
if args.cuda:
	net.cuda()

net.eval()

with open(args.labelJson, 'r') as f :
	label = ujson.load(f)

## get query feature
queryFeat = feature.QueryFeat(args.searchDir, label, 1, args.queryFeatMax, args.cropSize, strideNet, args.margin, args.cuda, transform, net)

## Initialize dictionary to store results
resDict = outils.ResDictInit(queryFeat, args.searchDir)

## Scale List
scaleList = outils.ScaleList(args.featScaleBase, args.nbOctave, args.scalePerOctave)

## Retrieval
det, resDict = Retrieval(args.searchDir,
			args.queryFeatMax,
			scaleList,
			strideNet,
			args.cropSize,
			args.cuda,
			transform,
			net,
			queryFeat,
			resDict,
			args.nbPred,
			label)

## Detection Results
for category in tqdm(det.keys()) :
	for i in range(len(det[category])) :
		for j in range(len(det[category][i])):
			det[category][i][j] = (det[category][i][j][0], 0, det[category][i][j][2])

## Evaluate Detection
det, queryTable, categoryTable, mAPPerQuery = eval.Localization(det, label, IoUThresh = args.IoUThreshold, nbPred = args.outPred)

if args.detTxt :
	f = open(args.detTxt, 'w')
	msg = '\t\t\t Localization Result of Brueghel (IoU = {:.1f}) \t\t\tNumber prediction {:d}\n'.format(args.IoUThreshold, args.outPred)
	f.write (msg)
	f.write(queryTable.get_string())
	f.write(categoryTable.get_string())
	f.close()

if args.detJson :
	with open(args.detJson, 'w') as f :
		ujson.dump(det, f)

if args.detMAP :
	with open(args.detMAPJson, 'w') as f :
		ujson.dump(mAPPerQuery, f)


if args.nbDraw > 0 :
	outils.drawBb(args.visualDir, args.searchDir, label, args.nbDraw, det)
