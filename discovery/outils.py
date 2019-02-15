import numpy as np
from skimage import measure
import torch
from torch.autograd import Variable
from scipy.signal import convolve2d
from itertools import combinations

## resize the image to the indicated scale
def ResizeImg(featMax, featMin, minNet, strideNet, w, h) :

	ratio = float(w)/h
	if ratio < 1 : 
		featH = featMax 
		featW = max(round(ratio * featH), featMin )
		
	else : 
		featW = featMax 
		featH = max(round(featW/ratio), featMin )
	resizeW = (featW - 1) * strideNet + minNet
	resizeH = (featH - 1) * strideNet + minNet

	return int(resizeW), int(resizeH), float(resizeW)/w, float(resizeH)/h
	
	
def VoteMatrix(tolerance) : 
	ax = np.arange(-1 * tolerance , tolerance + 1)
	xx, yy = np.meshgrid(ax, ax)
	kernel = np.exp(-(xx**2 + yy**2) / (tolerance**2)) 
	vote = torch.from_numpy(kernel.astype(np.float32))
	return vote
	
def SaliencyCoef(feat) : 
	tmp_feat = feat[ :, :, 1 : feat.size()[2] - 1 , 1 : feat.size()[3] - 1]
	tmp_coef =torch.cat([torch.sum(feat[ :, :, 2 : feat.size()[2] , 1 : feat.size()[3] - 1] * tmp_feat, dim = 1, keepdim=True),
						 torch.sum(feat[ :, :, : feat.size()[2] - 2 , 1 : feat.size()[3] - 1] * tmp_feat, dim = 1, keepdim=True),
						 torch.sum(feat[ :, :, 1 : feat.size()[2] - 1 , : feat.size()[3] - 2] * tmp_feat, dim = 1, keepdim=True), 
						 torch.sum(feat[ :, :, 1 : feat.size()[2] - 1, 2 : feat.size()[3]] * tmp_feat, dim = 1, keepdim=True) ] , dim = 1)
	saliency_coef = torch.cuda.FloatTensor(feat.size()[2], feat.size()[3]).fill_(1)
	saliency_coef[1 : feat.size()[2] - 1, 1 : feat.size()[3] - 1] = torch.mean(tmp_coef, dim=1).squeeze()
	
	return saliency_coef.unsqueeze(0).unsqueeze(0)
	
def FeatImgRef(I, scaleImgRef, minNet, strideNet, margin, transform, model, featChannel, computeSaliencyCoef) : 

	# Resize image
	pilImgW, pilImgH = I.size
	resizeW, resizeH, wRatio, hRatio =  ResizeImg(scaleImgRef, 2 * margin + 1, minNet, strideNet, pilImgW, pilImgH)
	pilImg = I.resize((resizeW, resizeH))
	
	## Image feature
	feat=transform(pilImg).unsqueeze(0)
	feat = Variable(feat, volatile=True).cuda() 
	feat = model.forward(feat).data
	featW, featH = feat.size()[2], feat.size()[3] ## attention : featW and featH correspond to pilImgH and pilImgW respectively 
	feat = feat / (1e-7 + (torch.sum(feat **2, dim = 1, keepdim=True).expand(feat.size())**0.5) )
	featSaliency = SaliencyCoef(feat)
	feat = feat * (1 - featSaliency.expand(feat.size())) if computeSaliencyCoef else feat
	feat = feat[:, :, margin : -margin, margin : -margin].contiguous().view(featChannel, -1)
	feat = feat.view(featChannel, -1)
	
	## Other information
	bbox = [margin * strideNet / wRatio, margin * strideNet / hRatio, (featH - margin) * strideNet / wRatio, (featW - margin) * strideNet / hRatio]
	imgBbox = map(int, bbox)
	featW, featH = featW - 2 * margin, featH - 2 * margin
	listW = (torch.range(0, featW -1, 1)).unsqueeze(1).expand(featW, featH).contiguous().view(-1).type(torch.LongTensor)
	listH = (torch.range(0, featH -1, 1)).unsqueeze(0).expand(featW, featH).contiguous().view(-1).type(torch.LongTensor)
	
	return feat, pilImgW, pilImgH, featW, featH, listW, listH, imgBbox

	
def FeatImgRefBbox(I, scaleImgRef, minNet, strideNet, margin, transform, model, featChannel, computeSaliencyCoef, bb) : 

	# Resize image
	pilImgW, pilImgH = I.size
	resizeW, resizeH, wRatio, hRatio =  ResizeImg(scaleImgRef, 2 * margin + 1, minNet, strideNet, bb[2] - bb[0], bb[3] - bb[1])
	bb0, bb1, bb2, bb3 = bb[0] * wRatio, bb[1] * hRatio, bb[2] * wRatio, bb[3] * hRatio
	pilImg = I.resize((int(pilImgW * wRatio), int(pilImgH * hRatio)))
	
	wLeftMargin = min(int(bb0) / strideNet, margin)
	hTopMargin = min(int(bb1) / strideNet, margin)
	wRightMargin = min(int(pilImg.size[0] - bb2) / strideNet, margin)
	hBottomMargin = min(int(pilImg.size[1] - bb3) / strideNet, margin)
	pilImg = pilImg.crop( (bb0 - wLeftMargin * strideNet, bb1 - hTopMargin * strideNet, bb2 + wRightMargin * strideNet, bb3 + hBottomMargin * strideNet) )
	
			
	## Image feature
	feat=transform(pilImg)
	feat=feat.unsqueeze(0)
	feat = Variable(feat, volatile=True).cuda() 
	feat = model.forward(feat).data
	feat = feat / (1e-7 + (torch.sum(feat **2, dim = 1, keepdim=True).expand(feat.size())**0.5) )
	featSaliency = SaliencyCoef(feat)
	feat = feat * (1 - featSaliency.expand(feat.size())) if computeSaliencyCoef else feat
	feat = feat[:, :, hTopMargin : feat.size()[2] - hBottomMargin, wLeftMargin : feat.size()[3] - wRightMargin ].contiguous()
	featW, featH = feat.size()[2], feat.size()[3] ## attention : featW and featH correspond to pilImgH and pilImgW respectively 
	feat = feat.view(featChannel, -1)
	
	## Other information
	bbox = [0, 0, bb2 - bb0, bb3 - bb1]
	imgBbox = map(int, bbox)
	listW = (torch.range(0, featW -1, 1)).unsqueeze(1).expand(featW, featH).contiguous().view(-1).type(torch.LongTensor)
	listH = (torch.range(0, featH -1, 1)).unsqueeze(0).expand(featW, featH).contiguous().view(-1).type(torch.LongTensor)
	
	return feat, I, pilImgW, pilImgH, featW, featH, listW, listH, bb
	
def imgFeat(minNet, strideNet, I, model, transform, scale) : 
	w,h = I.size
	new_w, new_h, _, _ = ResizeImg(scale, 1, minNet, strideNet, w, h)
	Ifeat = Variable(transform(I.resize((new_w, new_h))).unsqueeze(0).cuda(), volatile=True) 
	Ifeat = model(Ifeat)
	return Ifeat
	
def MatchPair(minNet, strideNet, model, transform, scales, feat1, feat1W, feat1H, I2, listW, listH, featChannel, tolerance, vote):

	match1 = []
	match2 = []
	similarity = []
	nbFeat = feat1W * feat1H
	bestScore = 0
	matchSetT = [] # We estimate the transformation from matchSetT
	
	for i in range(len(scales)) : 
		# Normalized I2 feature
		tmp_I2 = imgFeat(minNet, strideNet, I2, model, transform, scales[i])
		tmp_I2 = tmp_I2.data
		tmp_norm = torch.sum(tmp_I2 ** 2, dim = 1, keepdim=True) ** 0.5 + 1e-7
		tmp_I2 = tmp_I2 / tmp_norm.expand(tmp_I2.size())
		
		# Hough Transformation Grid, only for the current scale
		tmp_feat_w, tmp_feat_h = tmp_I2.shape[2], tmp_I2.shape[3]
		tmp_transformation = np.zeros((feat1W + tmp_feat_w, feat1H + tmp_feat_h))
		
		# I2 spatial information
		tmp_nbFeat = tmp_feat_w * tmp_feat_h
		tmp_I2 = tmp_I2.view(featChannel, -1).contiguous().transpose(0, 1)
		tmp_w = (torch.range(0, tmp_feat_w-1,1)).unsqueeze(1).expand(tmp_feat_w, tmp_feat_h).contiguous().view(-1).type(torch.LongTensor)
		tmp_h = (torch.range(0, tmp_feat_h-1, 1)).unsqueeze(0).expand(tmp_feat_w, tmp_feat_h).contiguous().view(-1).type(torch.LongTensor)

		# Feature Similarity
		score = torch.mm(tmp_I2, feat1)
		
		# Top1 match for both images
		topk0_score, topk0_index = score.topk(k=1, dim = 0)
		topk1_score, topk1_index = score.topk(k=1, dim = 1)
		
		index0 = torch.cuda.FloatTensor(tmp_nbFeat, nbFeat).fill_(0).scatter_(0, topk0_index, topk0_score)
		index1 = torch.cuda.FloatTensor(tmp_nbFeat, nbFeat).fill_(0).scatter_(1, topk1_index, topk1_score)
		intersectionScore = index0 * index1
		intersection = intersectionScore.nonzero()
		
		for item in intersection : 
			i2, i1 = item[0], item[1]
			w1, h1, w2, h2 = listW[i1], listH[i1], tmp_w[i2], tmp_h[i2]
			
			# Store all the top1 matches
			match1.append([(w1 + 0.49) / feat1W, (h1 + 0.49) / feat1H])
			match2.append([(w2 + 0.49) / tmp_feat_w, (h2 + 0.49) / tmp_feat_h])
			
			shift_w = int(w1 - w2  + tmp_feat_w - 2)
			shift_h = int(h1 - h2 + tmp_feat_h - 2)
			
			# Update the current hough transformation grid
			tmp_transformation[shift_w, shift_h] = tmp_transformation[shift_w, shift_h] + 1
			
			similarity.append(intersectionScore[i2, i1] ** 0.5)
		
		# Update the hough transformation grid with gaussian voting
		tmp_transformation = convolve2d(tmp_transformation, vote, mode='same')
		
		# Find the best hough transformation, and store the correspondant matches on matchSetT
		mode_non_zero = np.where((tmp_transformation == tmp_transformation.max()))
		score = tmp_transformation[mode_non_zero[0][0], mode_non_zero[1][0]]
		
		if score > bestScore :
			tmp_match1_ransac = []
			tmp_match2_ransac = []
			begin = len(match1) - len(intersection)  
			tmp_index = []
			for j, item in enumerate(intersection) : 
				i2, i1 = item[0], item[1]
				w1, h1, w2, h2 = listW[i1], listH[i1], tmp_w[i2], tmp_h[i2]
				shift_w = int(w1 - w2  + tmp_feat_w -2)
				shift_h = int(h1 - h2 + tmp_feat_h -2)
				diff_w, diff_h = np.abs(shift_w - mode_non_zero[0][0]), np.abs(shift_h - mode_non_zero[1][0])
				if diff_w <= tolerance and diff_h <= tolerance : 
					tmp_index.append(j+begin)
				
			matchSetT = tmp_index
			bestScore = score
			
	
	match1, match2, similarity = np.array(match1), np.array(match2), np.array(similarity)
	if len(matchSetT) == 0 :
		return [], [] , [], []
		
	return np.hstack((match1, np.ones((match1.shape[0], 1)))), np.hstack((match2, np.ones((match2.shape[0], 1)))), similarity, matchSetT
	
def Homography(X, Y):
	#X, Y, dimension : 4 * 3 
	A = np.zeros((8, 9))
	for i in range(4) : 
		u, v, u_, v_ = Y[i, 0], Y[i, 1], X[i, 0], X[i, 1] 
		A[2 * i] = np.array([0, 0, 0, -u, -v, -1, v_ * u, v_ * v, v_])
		A[2 * i + 1] = np.array([u, v, 1, 0, 0, 0, -u_ * u, -u_ * v, -u_])
		 
	#svd composition
	u, s, v = np.linalg.svd(A)
	#reshape the min singular value into a 3 by 3 matrix
	H21 = np.reshape(v[8], (3, 3))
	return H21
	
def Affine(X, Y):
	nb_points = X.shape[0]
	H21 = np.linalg.lstsq(Y, X[:, :2])[0]
	H21 = H21.T
	H21 = np.array([[H21[0, 0], H21[0, 1], H21[0, 2]], 
					[H21[1, 0], H21[1, 1], H21[1, 2]],
					[0, 0, 1]])
	return H21

def Hough(X, Y) : 
	nb_points = X.shape[0]
	ones = np.ones((nb_points, 1))
	H21x = np.linalg.lstsq(np.hstack((Y[:, 0].reshape((-1, 1)), ones)), X[:, 0])[0]
	H21y = np.linalg.lstsq(np.hstack((Y[:, 1].reshape((-1, 1)), ones)), X[:, 1])[0]
	
	H21 = np.array([[H21x[0], 0, H21x[1]], 
					[0, H21y[0], H21y[1]],
					[0, 0, 1]])
	return H21
	
def Prediction(X, Y, H21) : 

	estimX = (np.dot(H21, Y.T)).T
	estimX = estimX / estimX[:, 2].reshape((-1, 1))
	return np.sum((X[:, :2]- estimX[:, :2])**2, axis=1)**0.5
	
	
def ScoreRANSAC(match1, match2, matchSetT, sampleIndex, score, tolerance, nbSamplePoint, paramEstimate) : 
	
	#All The Data
	nbMatch = len(matchSetT)
	sampleIndex = np.array(matchSetT)[sampleIndex]
	X = match1[sampleIndex]
	Y = match2[sampleIndex]
	H21 = paramEstimate(X, Y)
	 
	error = Prediction(match1, match2, H21)
	
	isInlier = error < tolerance
	inlier = {}
	
	for i in range(len(match1)) : 
		if isInlier[i] : 
			score_i = score[i] * np.exp(-1 * error[i] ** 2 / tolerance ** 2)
			#score_i = score[i]
			key = (match1[i][0], match1[i][1])
			if inlier.has_key(key) and inlier[key][1] < score_i : 
				inlier[key] = [match2[i], score_i] 
			elif not inlier.has_key(key) : 
				inlier[key] = [match2[i], score_i]
			
	return H21, sum([item[1] for item in inlier.values()]), matchSetT
	
def RANSAC(nbIter, match1, match2, matchSetT, score, tolerance, nbSamplePoint) : 
	
	bestParams = []
	bestScore = 0
	
	if nbSamplePoint == 2 : 
		paramEstimate = Hough
		transformation = 'Hough'
		if nbIter == 1 :
			nbSamplePoint = len(matchSetT)
	elif nbSamplePoint == 3 : 
		paramEstimate = Affine
		transformation = 'Affine'
	elif nbSamplePoint == 4:
		paramEstimate = Homography
		transformation = 'Homography'
	
	nbMatch = len(matchSetT)
	nbCombination = int(np.prod([nbMatch - i for i in range(nbSamplePoint)]) / np.prod([i + 1 for i in range(nbSamplePoint)]))	
	sampleIndexList = np.array(list(combinations(range(nbMatch),nbSamplePoint)))[np.random.choice(np.arange(nbCombination), min(nbCombination, nbIter), replace=False)]
	

	for i in range(len(sampleIndexList)) : 
		H21, pairScore, matchSetT = ScoreRANSAC(match1, match2, matchSetT, sampleIndexList[i], score, tolerance, nbSamplePoint, paramEstimate)
		if pairScore > bestScore : 
			bestParams = H21
			bestScore = pairScore
	if len(bestParams) == 0 : 
		return [], 0, 0
		
	error = Prediction(match1, match2, bestParams)
	
	isInlier = error < tolerance
	inlier = {}
	for i in range(len(error)) : 
		if isInlier[i] : 
			score_i = score[i]
			key = (match1[i][0], match1[i][1])
			if inlier.has_key(key) and inlier[key][1] < score_i : 
				inlier[key] = [match2[i], score_i] 
			elif not inlier.has_key(key) : 
				inlier[key] = [match2[i], score_i] 
				
	return bestParams, bestScore, inlier

## find the feature size of target images
## The matching points in target image are not necessarily in the same scale
## Thins function aims at finding optimal feature dimension for the target image
## What we do is to take a bbox [(0,0), (0,1), (1,0), (1,1)] in source image and map it to target image
## Since we know the feature dimension of source image, we can thus calculate feature dimension of target image
 
def FeatSizeImgTarget(H21, sourceFeatW, sourceFeatH) :

	map00_x, map00_y= H21[0, 2] / H21[2, 2], H21[1, 2] / H21[2, 2]
	map01_x, map01_y= (H21[0, 1] + H21[0, 2]) / (H21[2, 1] + H21[2, 2]), (H21[1, 1] + H21[1, 2]) / (H21[2, 1] + H21[2, 2])
	map10_x, map10_y= (H21[0, 0] + H21[0, 2]) / (H21[2, 0] + H21[2, 2]), (H21[1, 0] + H21[1, 2]) / (H21[2, 0] + H21[2, 2])
	
	targetFeatW = (((map10_x - map00_x) * sourceFeatW) ** 2 + ((map10_y - map00_y) * sourceFeatH) **2 ) ** 0.5
	targetFeatH = (((map01_x - map00_x) * sourceFeatW) ** 2 + ((map01_y - map00_y) * sourceFeatH) **2 ) ** 0.5
	
	return int(targetFeatW), int(targetFeatH)
	
## Keep only bijective matching: matches that are verified in both directions (from I1 to I2 and I2 to I1)
def BackwardVerification(feat2W, feat2H, feat1W, feat1H, inlier) : 

	match_bijective = {} # from I2 to I1
	for pos1 in inlier.keys() : 
		pos1_w, pos1_h = int(pos1[0] * feat1W), int(pos1[1] * feat1H)
		pos2, score = inlier[pos1]
		pos2_w, pos2_h = int(pos2[0] * feat2W), int(pos2[1] * feat2H)
		key = (pos2_w, pos2_h)
		if match_bijective.has_key(key) and match_bijective[key][1]   < score : 
			match_bijective[key] = [(pos1_w, pos1_h), score] 
		elif not match_bijective.has_key(key) : 
			match_bijective[key] = [(pos1_w, pos1_h), score]
		
	match2 = match_bijective.keys()
	match1 = [match_bijective[key][0] for key in match2]
	score =  [match_bijective[key][1] for key in match2]
	
	return match1, match2, score

## Remove one small Connected Component (CC)
def RemoveSmallCC(match1, match2, mask1, mask2, threshold, score) :
	if len(match1) <= threshold or len(match2) <= threshold: 
		return [], [], []
	
	match1, match2 = np.array(match1), np.array(match2)
	mask1[match1[:, 0].flatten().astype(int), match1[:, 1].flatten().astype(int)] = 1
	mask2[match2[:, 0].flatten().astype(int), match2[:, 1].flatten().astype(int)] = 1
	label1 = measure.label(mask1, connectivity=2)
	label2 = measure.label(mask2, connectivity=2)
	drop_pos1 = []
	drop_pos2 = []
	
	for i in range(1, len(np.unique(label1))) : 
		posx, posy = np.where(label1 == i)
		pos = [(posx[i], posy[i]) for i in range(len(posx))]
		if len(pos) < threshold :
			drop_pos1 = drop_pos1 + pos
			
	for i in range(1, len(np.unique(label2))) : 
		posx, posy = np.where(label2 == i)
		pos = [(posx[i], posy[i]) for i in range(len(posx))]
		if len(pos) < threshold :
			drop_pos2 = drop_pos2 + pos
	update_match1 = []
	update_match2 = []
	update_score = []
	for i in range(len(match1)) : 
		if tuple(match1[i].tolist()) in drop_pos1 or tuple(match2[i][:2].tolist()) in drop_pos2: 
			continue
		else : 
			update_match1.append(tuple(match1[i].tolist()))
			update_match2.append(tuple(match2[i].tolist()))
			update_score.append(score[i])
	return update_match1, update_match2, update_score

## Keep only Large CCs
def KeepOnlyLargeCC(match1, match2, mask1, mask2, threshold, score) : 
	update_match1, update_match2, score = RemoveSmallCC(match1, match2, np.zeros(mask1.shape), np.zeros(mask2.shape), threshold, score)
	while len(update_match1) != len(match1) : 
		match1, match2 = update_match1, update_match2
		update_match1, update_match2, score = RemoveSmallCC(match1, match2, np.zeros(mask1.shape), np.zeros(mask2.shape), threshold, score)
	return match1, match2, score
	
def GetCC(match1, match2, mask1, mask2, score) : 
	match1_arr = np.array(match1).astype(int)
	mask1[match1_arr[:, 0].flatten(), match1_arr[:, 1].flatten()] = 1
	label1, nb_label1 = measure.label(mask1, connectivity=2, return_num=True)
	dict_match = dict(zip(match1, match2))
	dict_score = dict(zip(match1, score))
	
	CC = []
	CC_score = np.zeros(nb_label1)
	for i in range(1, nb_label1 + 1) : 
		CC.append({})
		posx, posy = np.where(label1 == i)
		tmp_match1 = [(posx[j], posy[j]) for j in range(len(posx))]
		tmp_match2 = [dict_match[item] for item in tmp_match1]
		CC[i-1] = dict(zip(tmp_match1, tmp_match2))
		CC[i-1]['mask2shape'] = mask2.shape
		CC_score[i -1 ] = sum(dict_score[item] for item in tmp_match1)
	return CC, CC_score
	
def ExtendRemove(match) : 
	tmp_match = np.array([[match[i][0], match[i][1]] for i in range(len(match))]).astype(int)
	binary_matrix = np.zeros((tmp_match[:, 0].max() + 1, tmp_match[:, 1].max() + 1)).astype(int)
	binary_matrix[tmp_match[:, 0], tmp_match[:, 1]] = 1
	extension = (convolve2d(binary_matrix, np.ones((3,3)), mode='same') >= 1).astype(int)
	final_match_x, final_match_y = np.where(extension == True)
	return [(final_match_x[i], final_match_y[i]) for i in range(len(final_match_x))]
	
	
def ScaleList(featScaleBase, nbOctave, scalePerOctave) :

	scaleList = np.array([featScaleBase * (2 ** nbOctave -  2**(float(scaleId) / scalePerOctave)) for scaleId in range(0, 1 + nbOctave * scalePerOctave)]).astype(int) + featScaleBase

	return scaleList

