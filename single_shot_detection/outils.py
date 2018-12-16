from tqdm import tqdm
import os

import PIL.Image as Image
import PIL.ImageDraw as ImageDraw

import numpy as np
import feature

import torch
import torch.nn.functional as F
from torch.autograd import Variable

## Initialize results of dictionary
def ResDictInit(query, searchDir) :

	res = {}
	for category in tqdm(query.keys()) :
		res[category] = []
		for i in range(len(query[category])) :
			res[category].append({})
			for searchName in os.listdir(searchDir) :
				res[category][i][searchName] = []

	return res

## Scale List Function
def ScaleList(featScaleBase, nbOctave, scalePerOctave) :

	scaleList = np.array([featScaleBase * (2 ** nbOctave -  2**(float(scaleId) / scalePerOctave)) for scaleId in range(0, 1 + nbOctave * scalePerOctave)]).astype(int) + featScaleBase

	return scaleList

def CosineSimilarity(img_feat, kernel, kernel_one) :

	dot = F.conv2d(img_feat, kernel)
	img_feat_norm = F.conv2d(img_feat ** 2, kernel_one) ** 0.5 + 1e-7
	score = dot/img_feat_norm.expand(dot.size())

	return score.data

def Match(img_feat, kernel, useGpu) :

	feat_out_dim = kernel.size()[0]
	img_feat = Variable(img_feat)
	kernel = Variable(kernel)
	kernel_one = Variable(torch.ones(kernel.size())).cuda() if useGpu else Variable(torch.ones(kernel.size()))
	score = CosineSimilarity(img_feat, kernel, kernel_one)

	return score



## Non Maximal Suppression
def NMS(boxes, overlapThresh = 0):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes
	pick = []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	score = boxes[:,4]

	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1) * (y2 - y1)
	idxs = np.argsort(score)

	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
 		if len(idxs) == 1:
 			break
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1)
		h = np.maximum(0, yy2 - yy1)

		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]

		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))

	# return only the bounding boxes that were picked using the
	# integer data type
	return pick

def IoU(bbox, bboxArray):
	"""
	Calculate IoU betwwen bbox (only one bbox) and some bboxes (a list of bboxes)
	input :
			bbox : a numpy array, 4 dimension, top left position and bottom right position
			bboxArray : a numpy array, N * 4 dimension, N bboxes
	"""
	nb_box = len(bboxArray)
	x1 = bbox[0];
	y1 = bbox[1];
	width1 = bbox[2] - bbox[0];
	height1 = bbox[3] - bbox[1];

	x2List = bboxArray[:, 0];
	y2List = bboxArray[:, 1];
	width2 = bboxArray[:, 2] - bboxArray[:, 0];
	height2 = bboxArray[:, 3] - bboxArray[:, 1];

	endx = np.maximum(x1 + width1, x2List + width2);
	startx = np.minimum(x1, x2List);
	width = width1 + width2 - ( endx - startx );

	endy = np.maximum(y1 + height1, y2List + height2);
	starty = np.minimum(y1, y2List);
	height = height1 + height2 - (endy - starty);

	width[width <=0] = 0
	height[height <=0] = 0
	areaInter = width * height; # intersection area
	area1 = width1 * height1;
	area2 = width2 * height2;
	ratio = areaInter*1./(area1 + area2 - areaInter);

	return ratio

def FeatPos2ImgBB(infoFind, kernelSize, imgSize, strideNet, cropSize) :

	bb = np.zeros((len(infoFind), 5))

	for i, item in enumerate(infoFind):

		new_w, new_h, _, _ = feature.ImgResize(max(kernelSize), item[3], 0, strideNet, imgSize[0], imgSize[1])

		imgFeatDim1 = new_h  / strideNet
		imgFeatDim2 = new_w  / strideNet
		print new_h, new_w
		
		top = max(item[1]  - cropSize, 0)/float(imgFeatDim1) * imgSize[1]
		left = max(item[2] - cropSize, 0)/float(imgFeatDim2) * imgSize[0]
		bottom = min((item[1] + kernelSize[0] + cropSize)/float(imgFeatDim1), 1) * imgSize[1]
		right = min((item[2] + kernelSize[1] + cropSize)/float(imgFeatDim2), 1) * imgSize[0]


		bb[i] = np.array([left, top, right, bottom, item[-1]]) if item[0] == 0 else np.array([imgSize[0] - right, top, imgSize[0] - left, bottom, item[-1]])

	pick = NMS(bb, overlapThresh = 0)
	bb = bb[pick]
	infoFind = [infoFind[i] for i in pick]

	return bb, infoFind

def drawRectanglePil(I, bbox, bboxColor=[255, 0 ,0], lineWidth = 3, alpha = 100):
	"""draw a bounding box with bbox color
	"""
	assert 'PIL' in str(type(I))

	draw = ImageDraw.Draw(I)
	rgba = (bboxColor[0], bboxColor[1], bboxColor[2], alpha)
	draw.rectangle(bbox, outline=rgba)
	for i in range(1, lineWidth) :
		bboxIndex = [(bbox[0][0] + i, bbox[0][1] + i),(bbox[1][0] - i, bbox[1][1] - i)]
		draw.rectangle(bboxIndex, outline=rgba)

	return I

def drawBb(visualDir, searchDir, label, nbDraw, res) :
	os.mkdir(visualDir)

	for queryCat in res.keys() :
		subRoot = os.path.join(visualDir, queryCat)
		os.mkdir(subRoot)

		for i, item in enumerate(res[queryCat]) :
			queryName = label[queryCat][i]['query'][0]
			subSubRoot = os.path.join(subRoot, queryName)
			os.mkdir(subSubRoot)

			pathQuery = os.path.join(searchDir, queryName)
			bb = label[queryCat][i]['query'][1]

			if len(bb) != 0 :
				bbox = [(bb[0], bb[1]), (bb[2], bb[3])]

				Iquery = Image.open(pathQuery).convert('RGB')

				Iquery = drawRectanglePil(Iquery, bbox, [0, 0, 255], 4, 255)
				Iquery.save(os.path.join(subSubRoot, 'query.png'))

			else :
				copyfile(pathQuery, os.path.join(subSubRoot, 'query.png'))

			for t, (key, score, bb) in enumerate(res[queryCat][i]) :
				if t < nbDraw :
					Imatch = Image.open( os.path.join(searchDir, key) ).convert('RGB')
					bb = [(bb[0], bb[1]), (bb[2], bb[3])]

					for item in label[queryCat][i]['gt'] :
						if key == item[0] :
							bbox =  item[1]
							bbox = [(bbox[0], bbox[1]), (bbox[2], bbox[3])]

					if score :
						patchName = 'match_' + str(t) + '_TP_' + key
						Imatch = drawRectanglePil(Imatch, bb, [0, 255, 0], 4, 255)

					else :
						patchName = 'match_' + str(t) + '_' + key
						Imatch = drawRectanglePil(Imatch, bb, [255, 0, 0], 4, 255)

					Imatch.save(os.path.join(subSubRoot, patchName))
