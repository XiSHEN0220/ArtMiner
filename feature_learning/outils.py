import numpy as np
import PIL.Image as Image
import torch
import os
from itertools import product
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm

## Each image in the searching dataset will be resized
## The function below defines maximum dimensions in the feature map for different scales

def ScaleList(featScaleBase, nbOctave, scalePerOctave) :

	scaleList = np.array([featScaleBase * (2 ** nbOctave -  2**(float(scale_id)/scalePerOctave)) for scale_id in range(0, 1 + nbOctave * scalePerOctave)]).astype(int) + featScaleBase

	return scaleList

## Given a featMax (maximum dimensions in the feature map)
## The function calculate the output size to resize the image with keeping the aspect reatio
## The minimum dimensions in the feature map is at least featMin

def ResizeImg(featMin, featMax, minNet, strideNet, w, h) :

	ratio = float(w)/h
	if ratio < 1 :
		featH = featMax
		featW = max(round(ratio * featH), featMin)
	else :
		featW = featMax
		featH = max(round(featW/ratio), featMin)

	resizeW, resizeH = int((featW -1) * strideNet + minNet), int((featH -1) * strideNet + minNet)

	return resizeW, resizeH

class InfiniteSampler():
	def __init__(self, img_list):
		self.img_list = img_list
	def loop(self):

		while True:
			for i, data in enumerate(self.img_list) :
				yield i, data
			self.img_list = np.random.permutation(self.img_list)

## Random Query Feature

def RandomQueryFeat(nbPatchTotal, featChannel, searchRegion, imgFeatMin, minNet, strideNet, transform, net, searchDir, margin, imgList, useGpu, queryScale) :

	featQuery = torch.cuda.FloatTensor(nbPatchTotal, featChannel, searchRegion, searchRegion) # Store feature
	img_sampler = InfiniteSampler(imgList)
	count = 0

	for (i, img_name) in tqdm(img_sampler.loop()) :
		if count == nbPatchTotal :
			break

		## resize image
		I = Image.open(os.path.join(searchDir, img_name)).convert('RGB')
		w,h = I.size
		scale = np.random.choice(queryScale) ## Predefine some scales

		new_w, new_h = ResizeImg(imgFeatMin, scale, minNet, strideNet, w, h)
		I = I.resize((new_w, new_h))

		## Image Feature
		I_data = transform(I).unsqueeze(0)
		I_data = I_data.cuda() if useGpu else I_data ## set cuda datatype?
		I_data = net(Variable(I_data, volatile=True)).data ## volatile, since do not need gradient

		## Query feature + Query Information
		feat_w, feat_h = I_data.shape[2], I_data.shape[3]
		feat_w_pos = np.random.choice(np.arange(margin, feat_w - margin - searchRegion, 1), 1)[0]
		feat_h_pos = np.random.choice( np.arange(margin, feat_h - margin - searchRegion, 1), 1)[0]
		featQuery[count] = I_data[:, :, feat_w_pos : feat_w_pos + searchRegion, feat_h_pos : feat_h_pos + searchRegion].clone()

		count += 1

	return Variable(featQuery)

## Cosine similarity Implemented as a Convolutional Layer
## Note: we don't normalize kernel
def CosineSimilarity(img_feat, kernel, kernel_one) :

	dot = F.conv2d(img_feat, kernel, stride = 1)
	img_feat_norm = F.conv2d(img_feat ** 2, kernel_one, stride = 1) ** 0.5 + 1e-7
	score = dot/img_feat_norm.expand(dot.size())

	return score.data

## Cosine similarity and we only keep topK score
def CosineSimilarityTopK(img_feat, img_feat_norm, kernel, K) :

	dot = F.conv2d(img_feat, kernel, stride = 1)
	score = dot/img_feat_norm.expand_as(dot)
	_, _, score_w, score_h =  score.size()
	score = score.view(kernel.size()[0], score_w * score_h)
	topk_score, topk_index = score.topk(k = K, dim = 1)
	topk_w, topk_h = topk_index / score_h, topk_index % score_h

	return topk_score, topk_w, topk_h

## Feature Normalization, Divided by L2 Norm
def Normalization(feat) :

	feat_norm = (torch.sum(torch.sum(torch.sum(feat ** 2, dim = 1), dim = 1), dim = 1) + 1e-7) ** 0.5
	feat_normalized = feat / feat_norm.view(feat.size()[0], 1, 1, 1).expand_as(feat)

	return feat_normalized

## Image feature for searching image :
##                                     1. features in different scales;
##                                     2. remove feature in the border
def SearchImgFeat(searchDir, margin, searchRegion, scales, minNet, strideNet, transform, model, searchName, useGpu) :

	searchFeat = {}
	I = Image.open(os.path.join(searchDir, searchName)).convert('RGB')
	w,h = I.size

	for s in scales :

		new_w, new_h = ResizeImg(2 * margin + searchRegion + 1, s, minNet, strideNet, w, h)
		I_pil = I.resize((new_w, new_h))
		I_data = transform(I_pil).unsqueeze(0)
		I_data = Variable(I_data, volatile=True).cuda() if useGpu else Variable(I_data, volatile=True)
		feat = model.forward(I_data).data
		feat_w, feat_h = feat.shape[2], feat.shape[3]
		searchFeat[s] = Variable(feat)[:, :, margin : feat_w - margin, margin : feat_h - margin].clone()

	return searchFeat

def RetrievalRes(nbPatchTotal, imgList, searchDir, margin, searchRegion, scales, minNet, strideNet, transform, net, featQuery, useGpu) :

	resScale = torch.zeros((nbPatchTotal, len(imgList))).cuda() if useGpu else torch.zeros((nbPatchTotal, len(imgList))) # scale
	resW = torch.zeros((nbPatchTotal, len(imgList))).cuda() if useGpu else torch.zeros((nbPatchTotal, len(imgList)))     # feat_w
	resH = torch.zeros((nbPatchTotal, len(imgList))).cuda() if useGpu else torch.zeros((nbPatchTotal, len(imgList)))     # feat_h
	resScore = torch.zeros((nbPatchTotal, len(imgList))).cuda() if useGpu else torch.zeros((nbPatchTotal, len(imgList))) # score

	variableAllOne =  Variable(torch.ones(1, featQuery.size()[1], featQuery.size()[2], featQuery.size()[3])).cuda()

	for i in tqdm(range(len(imgList))) :

		search_name = imgList[i]
		searchFeat = SearchImgFeat(searchDir, margin, searchRegion, scales, minNet, strideNet, transform, net, search_name, useGpu)

		tmpScore = torch.zeros((nbPatchTotal, len(scales))).cuda() if useGpu else torch.zeros((nbPatchTotal, len(scales)))
		tmpH = torch.zeros((nbPatchTotal, len(scales))).cuda() if useGpu else torch.zeros((nbPatchTotal, len(scales)))
		tmpW = torch.zeros((nbPatchTotal, len(scales))).cuda() if useGpu else torch.zeros((nbPatchTotal, len(scales)))
		tmpScale = torch.zeros((nbPatchTotal, len(scales))).cuda() if useGpu else torch.zeros((nbPatchTotal, len(scales)))

		for j, scale in enumerate(searchFeat.keys()) :

			featImg = searchFeat[scale]
			score = CosineSimilarity(featImg, featQuery, variableAllOne)

			# Update tmp matrix
			outW = score.size()[2]
			outH = score.size()[3]
			score = score.view(score.size()[1], outW * outH)
			score, index= score.max(1)
			tmpW[:, j] = index/outH
			tmpH[:, j] = index%outH
			tmpScore[:, j] = score
			tmpScale[:, j] = scale

		tmpScore, tmpScaleIndex = tmpScore.max(1)
		tmpScaleIndex = tmpScaleIndex.unsqueeze(1)

		# Update res matrix, only keep top 10
		resScore[:, i] = tmpScore
		resScale[:, i] = torch.gather(tmpScale, 1, tmpScaleIndex)
		resW[:, i] = torch.gather(tmpW, 1, tmpScaleIndex)
		resH[:, i] = torch.gather(tmpH, 1, tmpScaleIndex)

	# Get Topk Matrix
	topkValue, topkImg = resScore.topk(k = 10, dim = 1) ## Take Top10 pairs
	topkScale = torch.gather(resScale, 1, topkImg)
	topkW = torch.gather(resW, 1, topkImg)
	topkH = torch.gather(resH, 1, topkImg)

	topkW = topkW.type(torch.cuda.LongTensor) if useGpu else topkW.type(torch.LongTensor)
	topkH = topkH.type(torch.cuda.LongTensor) if useGpu else topkH.type(torch.LongTensor)

	return topkImg, topkScale, topkValue, topkW + margin, topkH + margin


def VotePair(searchDir, imgList, topkImg, topkScale, topkW, topkH, transform, net, margin, validRegion, searchRegion, featChannel, useGpu, minNet, strideNet) :

	# Sample a pair
	queryIndex = np.random.choice(len(topkImg))
	imgPair = np.random.choice(range(10), 2, replace=False)
	info1 = (topkImg[queryIndex, imgPair[0]], topkScale[queryIndex, imgPair[0]], topkW[queryIndex, imgPair[0]] - ( validRegion + 1) / 2 + 1, topkH[queryIndex, imgPair[0]] - (validRegion + 1) / 2 + 1)
	info2 = (topkImg[queryIndex, imgPair[1]], topkScale[queryIndex, imgPair[1]], topkW[queryIndex, imgPair[1]] - ( validRegion + 1) / 2 + 1, topkH[queryIndex, imgPair[1]] - (validRegion + 1) / 2 + 1)

	# Image 1 Feature
	I1 = Image.open(os.path.join(searchDir, imgList[info1[0]])).convert('RGB')
	w,h = I1.size
	new_w, new_h = ResizeImg(2 * margin + searchRegion + 1, info1[1], minNet, strideNet, w, h)
	I1 = I1.resize((new_w, new_h))
	feat1 = net(Variable(transform(I1).unsqueeze(0).cuda(), volatile=True)).data if useGpu else net(Variable(transform(I1).unsqueeze(0), volatile=True)).data

	# Image 2 Feature
	I2 = Image.open(os.path.join(searchDir, imgList[info2[0]])).convert('RGB')
	w,h = I2.size
	new_w, new_h = ResizeImg(2 * margin + searchRegion + 1, info2[1], minNet, strideNet, w, h)
	I2 = I2.resize((new_w, new_h))
	feat2 = net(Variable(transform(I2).unsqueeze(0).cuda(), volatile=True)).data if useGpu else net(Variable(transform(I2).unsqueeze(0), volatile=True)).data

	# Normalized Feature of validated region in Image 1
	validFeat1 = torch.cat([feat1[:, :, pos_i, pos_j].clone()
						for pos_i, pos_j in product(range(info1[2], info1[2] +  validRegion ),
													range(info1[3], info1[3] +  validRegion ))
						if pos_i == info1[2] or pos_i == info1[2] + validRegion - 1 or pos_j == info1[3] or pos_j == info1[3] + validRegion - 1 ], dim = 0)
	validFeat1 = validFeat1 / ((torch.sum(validFeat1 ** 2, dim = 1, keepdim = True).expand_as(validFeat1) )**0.5)
	validFeat1 = validFeat1.unsqueeze(2).unsqueeze(3)

	# Top1 match in feat2
	variableAllOne = Variable(torch.ones(1, featChannel, 1, 1).cuda()) if useGpu else Variable(torch.ones(1, featChannel, 1, 1))
	featNorm2 = F.conv2d(Variable(feat2) ** 2, variableAllOne, stride = 1) ** 0.5 + 1e-7
	topkScore1, topkW1, topkH1 = CosineSimilarityTopK(Variable(feat2), featNorm2, Variable(validFeat1), K = 1)

	# Corresponding position
	pos2 = [[pos_i, pos_j]
			for pos_i, pos_j in product(range(info2[2], info2[2] + validRegion), range(info2[3] , info2[3] + validRegion))
			if pos_i == info2[2] or pos_i == info2[2] + validRegion - 1  or pos_j == info2[3] or pos_j == info2[3] + validRegion - 1]
	pos2 = np.array(pos2).astype('int')
	posW2 = torch.from_numpy(pos2[:, 0]).cuda() if useGpu else torch.from_numpy(pos2[:, 0])
	posH2 = torch.from_numpy(pos2[:, 1]).cuda() if useGpu else torch.from_numpy(pos2[:, 1])

	# Score : Number of Vote
	topkW1, topkH1 = topkW1.data.squeeze(), topkH1.data.squeeze()
	mask = (torch.abs(posW2 - topkW1) <= 1) *  (torch.abs(posH2 - topkH1) <= 1)
	score = torch.sum(mask)
	return queryIndex, imgPair, score


def TrainPair(nbPairTotal, searchDir, imgList, topkImg, topkScale, topkW, topkH, transform, net, margin, useGpu, featChannel, searchRegion, validRegion, nbImgEpoch, minNet, strideNet) :

	pairInfo = torch.zeros((nbPairTotal, 4)).cuda() if useGpu else torch.zeros((nbPairTotal, 4)) # query_index, pair1, pair2, score
	count = 0

	while count < nbPairTotal :

		queryIndex, imgPair, score = VotePair(searchDir, imgList, topkImg, topkScale, topkW, topkH, transform, net, margin, validRegion, searchRegion, featChannel, useGpu, minNet, strideNet)
		pairInfo[count, 0] = queryIndex
		pairInfo[count, 1] = imgPair[0]
		pairInfo[count, 2] = imgPair[1]
		pairInfo[count, 3] = score
		count += 1
		if count % 500 == 499 :
			print count

	score_sort, score_sort_index = pairInfo[:, 3].sort(descending=True)
	pairInfo = pairInfo[score_sort_index]
	return pairInfo[:nbImgEpoch], pairInfo[-nbImgEpoch:]
'''

def TrainPair(searchDir, imgList, topkImg, topkScale, topkW, topkH, transform, net, margin, useGpu, featChannel, searchRegion, validRegion, nbImgEpoch, minNet, strideNet) :

	pairInfo = torch.zeros((nbImgEpoch, 4)).cuda() if useGpu else torch.zeros((nbPairTotal, 4)) # query_index, pair1, pair2, score
	count = 0
	minScore = (validRegion * 4 - 4) * 0.6
	while count < nbImgEpoch :

		queryIndex, imgPair, score = VotePair(searchDir, imgList, topkImg, topkScale, topkW, topkH, transform, net, margin, validRegion, searchRegion, featChannel, useGpu, minNet, strideNet)
		if score >= minScore :
			pairInfo[count, 0] = queryIndex
			pairInfo[count, 1] = imgPair[0]
			pairInfo[count, 2] = imgPair[1]
			pairInfo[count, 3] = score
			count += 1
		if count % 500 == 499 :
			print count

	score_sort, score_sort_index = pairInfo[:, 3].sort(descending=True)
	pairInfo = pairInfo[score_sort_index]
	return pairInfo[:nbImgEpoch], pairInfo[-nbImgEpoch:]
'''


## Process training pairs, sampleIndex dimension: iterEpoch * batchSize
def DataShuffle(sample, batchSize) :

	nbSample = len(sample)
	iterEpoch = nbSample / batchSize

	permutationIndex = np.random.permutation(range(nbSample))
	sampleIndex = permutationIndex.reshape(( iterEpoch, batchSize)).astype(int)

	return sampleIndex

## Positive loss for a pair of positive matching
def PosCosineSimilaritytop1(feat1, feat2, pos_w1, pos_h1, pos_w2, pos_h2, variableAllOne) :

	feat1x1 = feat1[:, :, pos_w1, pos_h1].clone().contiguous()
	feat1x1 = feat1x1 / ((torch.sum(feat1x1 ** 2, dim = 1, keepdim= True).expand(feat1x1.size())) ** 0.5)

	tmp_pos_w2 = max(pos_w2 - 1, 0)
	tmp_pos_h2 = max(pos_h2 - 1, 0)
	tmp_end_w2 = min(pos_w2 + 2, feat2.size()[2])
	tmp_end_h2 = min(pos_h2 + 2, feat2.size()[3])

	featRec = feat2[:, :, tmp_pos_w2 : tmp_end_w2, tmp_pos_h2 : tmp_end_h2].clone().contiguous()
	featRecNorm = F.conv2d(featRec ** 2, variableAllOne, stride = 1) ** 0.5 + 1e-7

	return CosineSimilarityTopK(featRec, featRecNorm, feat1x1.unsqueeze(2).unsqueeze(3), K = 1)[0][0]



## Negative sample loss for a pair of negative matching
def NegaCosineSimilaritytopk(feat1, feat2, norm2, pos_w1, pos_h1, variableAllOne, topKLoss) :

	feat1x1 = feat1[:, :, pos_w1, pos_h1].clone().contiguous()
	feat1x1 = feat1x1 / ((torch.sum(feat1x1 ** 2, dim = 1, keepdim= True).expand(feat1x1.size())) ** 0.5)
	negaTopKLoss = CosineSimilarityTopK(feat2, norm2, feat1x1.unsqueeze(2).unsqueeze(3), K = topKLoss)[0]

	return torch.mean(negaTopKLoss)




def PairPos(pos_w1, pos_h1, pos_w2, pos_h2, trainRegion) :


	pos1 = [(pos_w1 , pos_h1 ), (pos_w1, pos_h1 + trainRegion - 1), (pos_w1 + trainRegion - 1, pos_h1), (pos_w1 + trainRegion - 1, pos_h1 + trainRegion - 1)]
	pos2 = [(pos_w2 , pos_h2 ), (pos_w2, pos_h2 + trainRegion - 1), (pos_w2 + trainRegion - 1, pos_h2), (pos_w2 + trainRegion - 1, pos_h2 + trainRegion - 1)]

	return pos1, pos2

def PosNegaSimilarity(posPair, posIndex, topkImg, topkScale, topkW, topkH, searchDir, imgList, minNet, strideNet, net, transform, searchRegion, trainRegion, margin, featChannel, useGpu, topKLoss) :

	# Pair information: image name, scale, W, H
	pair = posPair[posIndex]
	queryIndex = int(pair[0])
	pairIndex = [int(pair[1]), int(pair[2])]
	info1 = (topkImg[queryIndex, pairIndex[0]], topkScale[queryIndex, pairIndex[0]], topkW[queryIndex, pairIndex[0]] - (trainRegion + 1) / 2 + 1, topkH[queryIndex, pairIndex[0]] - (trainRegion + 1) / 2 + 1)
	info2 = (topkImg[queryIndex, pairIndex[1]], topkScale[queryIndex, pairIndex[1]], topkW[queryIndex, pairIndex[1]] - (trainRegion + 1) / 2 + 1 , topkH[queryIndex, pairIndex[1]] - (trainRegion + 1) / 2 + 1)

	## features of pair images
	I1 = Image.open(os.path.join(searchDir, imgList[info1[0]])).convert('RGB')
	w,h = I1.size
	new_w, new_h = ResizeImg(margin * 2 + searchRegion + 1, info1[1], minNet, strideNet, w, h)
	feat1 = net(Variable(transform(I1.resize((new_w, new_h))).unsqueeze(0).cuda())) if useGpu else net(Variable(transform(I1.resize((new_w, new_h))).unsqueeze(0)))

	I2 = Image.open(os.path.join(searchDir, imgList[info2[0]])).convert('RGB')
	w,h = I2.size
	new_w, new_h = ResizeImg(margin * 2 + searchRegion + 1, info2[1], minNet, strideNet, w, h)
	feat2 = net(Variable(transform(I2.resize((new_w, new_h))).unsqueeze(0).cuda())) if useGpu else net(Variable(transform(I2.resize((new_w, new_h))).unsqueeze(0)))



	variableAllOne = Variable(torch.ones(1, featChannel, 1, 1).cuda()) if useGpu else  Variable(torch.ones(1, featChannel, 1, 1))

	norm2 = F.conv2d(feat2 ** 2, variableAllOne, stride = 1) ** 0.5 + 1e-7
	norm1 = F.conv2d(feat1 ** 2, variableAllOne, stride = 1) ** 0.5 + 1e-7

	pos1, pos2 = PairPos(info1[2], info1[3], info2[2], info2[3], trainRegion)

	posTop1Similarity = []
	negaTopKSimilarity = []

	for (pair1, pair2) in zip(pos1, pos2) :

		posTop1Similarity.append( PosCosineSimilaritytop1(feat1, feat2, pair1[0], pair1[1], pair2[0], pair2[1], variableAllOne) )
		nega1 = NegaCosineSimilaritytopk(feat1, feat2, norm2, pair1[0], pair1[1], variableAllOne, topKLoss)
		nega2 = NegaCosineSimilaritytopk(feat2, feat1, norm1, pair2[0], pair2[1], variableAllOne, topKLoss)
		if nega1.data[0] > nega2.data[0] :
			negaTopKSimilarity.append(nega1)
		else :
			negaTopKSimilarity.append(nega2)


	return posTop1Similarity, negaTopKSimilarity
