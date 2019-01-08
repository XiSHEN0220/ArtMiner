# coding: utf-8
import outils

import sys
sys.path.append("..")
from model.model import Model

import torch 
import os
from torchvision import datasets, transforms,models
import numpy as np 
from tqdm import tqdm
import ujson 
import cv2
import argparse
import warnings
import PIL.Image as Image
from scipy.misc import imresize


from scipy.signal import convolve2d
import time

if not sys.warnoptions:
	warnings.simplefilter("ignore")

def SkipIteration(I1, I2, saveQuality, out1, out2) : 
	
	I1RGBA = cv2.cvtColor(np.array(I1), cv2.COLOR_RGBA2BGRA)
	I2RGBA = cv2.cvtColor(np.array(I2), cv2.COLOR_RGBA2BGRA)
	
	mask1 = np.ones((I1RGBA.shape[0], I1RGBA.shape[1])) * 100
	mask2 = np.ones((I2RGBA.shape[0], I2RGBA.shape[1])) * 100
	
	I1RGBA[:, :, 3] = mask1
	I2RGBA[:, :, 3] = mask2
	
	ratio1 = max(max(I1RGBA.shape[0], I1RGBA.shape[1]) / float(saveQuality), 1)
	I1RGBA =imresize(I1RGBA, (int(I1RGBA.shape[0] / ratio1), int(I1RGBA.shape[1] / ratio1)))
	
	ratio2 = max(I2RGBA.shape[0], I2RGBA.shape[1]) / float(saveQuality)
	I2RGBA =imresize(I2RGBA, (int(I2RGBA.shape[0] / ratio2), int(I2RGBA.shape[1] / ratio2)))
	
	
	cv2.imwrite(out1, I1RGBA)
	cv2.imwrite(out2, I2RGBA)
	
## Blur the mask
def BlurMask(mask) : 

	mask = convolve2d(mask, np.ones((5,5)) / 25., mode='same')
	mask = convolve2d(mask, np.ones((5,5)) / 25., mode='same')
	mask = convolve2d(mask, np.ones((5,5)) / 25., mode='same')
	
	return mask
	
	

def PairDiscovery(img1Path, img2Path, model, transform, tolerance, minFeatCC, margin, scaleImgRef, scaleList, houghInitial, nbSamplePoint, nbIter, saveQuality, computeSaliencyCoef, out1, out2, printMsg = True) : 
	
	msg = 'discovery between : {} and {}'.format(img1Path, img2Path)
	if printMsg : 
		print msg
	
	strideNet = 16
	minNet = 15
	
	featChannel = 256 
	
	
	vote = outils.VoteMatrix(tolerance)
	
	I1 = Image.open(img1Path).convert('RGB')
	feat1, pilImg1W, pilImg1H, feat1W, feat1H, list1W, list1H, img1Bbox  = outils.FeatImgRef(I1, scaleImgRef, minNet, strideNet, margin, transform, model, featChannel, computeSaliencyCoef)
	toleranceRef = tolerance / scaleImgRef
	I2 = Image.open(img2Path).convert('RGB')
	pilImg2W, pilImg2H = I2.size
	
	match1, match2, similarity, matchSetT = outils.MatchPair(minNet, strideNet, model, transform, scaleList, feat1, feat1W, feat1H, I2, list1W, list1H, featChannel, tolerance, vote)
	matchSetT = matchSetT if houghInitial else range(len(match1)) 
	
	if len(matchSetT) < nbSamplePoint  : 
		if out1 : 
			SkipIteration(I1, I2, saveQuality, out1, out2)
		return 0.
	
	bestParams, bestScore, inlier = outils.RANSAC(nbIter, match1, match2, matchSetT, similarity, toleranceRef, nbSamplePoint)
	
	if len(bestParams) == 0 :
		if out1 :
			SkipIteration(I1, I2, saveQuality, index)
		return 0.
	
	feat2W, feat2H = outils.FeatSizeImgTarget(bestParams, feat1W, feat1H)
	
	if feat2W == 0 or feat2H == 0 or feat2W >= 1000 or feat2H >= 1000: 
		if out1 :
			SkipIteration(I1, I2, saveQuality, out1, out2)
		return 0.
	
	match1, match2, score = outils.BackwardVerification(feat2W, feat2H, feat1W, feat1H, inlier)
	
	finalMask1 = np.ones((img1Bbox[3] - img1Bbox[1], img1Bbox[2] - img1Bbox[0]), dtype=np.uint8) * 100
	finalMask2 = np.ones((pilImg2H, pilImg2W), dtype=np.uint8) * 100

	
	mask2 = np.zeros((feat2W, feat2H))
	mask1 = np.zeros((feat1W, feat1H))
	
	match1, match2, score = outils.KeepOnlyLargeCC(match1, match2, mask1, mask2, minFeatCC, score)
	if len(match1) == 0 :
		if out1 :
			SkipIteration(I1, I2, saveQuality, out1, out2)
		return 0.
		
	_, score = outils.GetCC(match1, match2, mask1, mask2, score)
	sumScore = np.sum(score)
	finalScore = sumScore/float(feat1.size()[1]) 
	
	if out1 : 
		match1, match2 = np.array(outils.ExtendRemove(match1)), np.array(outils.ExtendRemove(match2))
	
		mask1[match1[:, 0], match1[:, 1]] = 1
		mask2[match2[:, 0], match2[:, 1]] = 1



		mask1 = imresize(mask1, (finalMask1.shape[0], finalMask1.shape[1])) / 128 > 0
		mask2 = imresize(mask2, (finalMask2.shape[0], finalMask2.shape[1])) / 128 > 0
	
	
		finalMask1[mask1 ] = 255
		finalMask2[mask2 ] = 255
	
		 
	

	
		I1RGBA = cv2.cvtColor(np.array(I1), cv2.COLOR_RGBA2BGRA)
		I2RGBA = cv2.cvtColor(np.array(I2), cv2.COLOR_RGBA2BGRA)

	
		mask1Index = finalMask1 > 0
		mask2Index =  finalMask2 > 0
	
		mask1 = np.ones((I1RGBA.shape[0], I1RGBA.shape[1])) * 100
		mask1[img1Bbox[1] : img1Bbox[3], img1Bbox[0] : img1Bbox[2]] = finalMask1
		mask2 = finalMask2
	
		mask1, mask2 = mask1.astype(np.uint8), mask2.astype(np.uint8)
		I1RGBA[:, :,3] = mask1
		I2RGBA[:, :,3] = mask2
	
	
		ratio1 = max(max(I1RGBA.shape[0], I1RGBA.shape[1]) / float(saveQuality), 1)
		I1RGBA =imresize(I1RGBA, (int(I1RGBA.shape[0] / ratio1), int(I1RGBA.shape[1] / ratio1)))
	
		ratio2 = max(I2RGBA.shape[0], I2RGBA.shape[1]) / float(saveQuality)
		I2RGBA =imresize(I2RGBA, (int(I2RGBA.shape[0] / ratio2), int(I2RGBA.shape[1] / ratio2)))
	
		I1RGBA[:, :, 3] = BlurMask(I1RGBA[:, :, 3]).astype(np.uint8)
		I2RGBA[:, :, 3] = BlurMask(I2RGBA[:, :, 3]).astype(np.uint8)
	
	
		cv2.imwrite(out1, I1RGBA)
		cv2.imwrite(out2, I2RGBA)
	
	return finalScore
	
	
	
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	##---- Search Dataset Setting ----####
	parser.add_argument(
		'--featScaleBase', type=int, default= 20, help='minimum # of features in the scale list ')

	parser.add_argument(
		'--scalePerOctave', type=int, default= 3, help='# of scales in one octave ')

	parser.add_argument(
		'--nbOctave', type=int, default= 2, help='# of octaves')

	##---- Model Setting ----####

	parser.add_argument(
		'--imagenetFeatPath', type=str, default='../model/resnet18.pth', help='imageNet feature net weight path')

	parser.add_argument(
		'--finetunePath', type=str, help='finetune net weight path')

	parser.add_argument(
		'--margin', type=int, default= 3, help='margin, the feature describing the border part is not taken into account')

	parser.add_argument(
		'--cuda', action='store_true', help='cuda setting')

	parser.add_argument(
		'--tolerance', type=float , default = 2., help='tolerance expressed by nb of features (2 for retrieval with image 1 for retrieval with region)')

	parser.add_argument(
		'--scaleImgRef', type=int , default = 40, help='maximum feature in the target image')
		
	parser.add_argument(
		'--houghInitial', action='store_true', help='sampling point from hough transform sets')
	
	parser.add_argument(
		'--img1Path', type=str, help='input path of image 1')
	
	parser.add_argument(
		'--out1', type=str, help='output path of image 1')
	
	parser.add_argument(
		'--img2Path', type=str, help='input path of image 2')
		
	parser.add_argument(
		'--out2', type=str, help='output path of image 2')
		
	parser.add_argument(
		'--nbSamplePoint', type=int, default = 3 , help='nb sample point = 2 ==> Hough, nb sample point = 3 ==> Affine, nb sample point = 4 ==> Homography')
	
	parser.add_argument(
		'--nbIter', type=int, default = 100 , help='nb iteration, nbIter = 1 ==> Hough transformation, parameter estimated with all points in the matchSet')
	
	parser.add_argument(
		'--saveQuality', type=int, default = 1000, help='output image quality')
		
	parser.add_argument(
		'--computeSaliencyCoef', action='store_true', help='using saliency coefficient for the feature of reference image?')
	
	parser.add_argument(
		'--minFeatCC', type=int, default = 4, help='minimum number of features in CC')
	
	
	


	
	args = parser.parse_args()
	print args
	
	net = Model(args.imagenetFeatPath, args.finetunePath)

	if args.cuda:
		net.cuda()
	net.eval()
	
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
					std = [ 0.229, 0.224, 0.225 ]),
	])
	
	scaleList = outils.ScaleList(args.featScaleBase, args.nbOctave, args.scalePerOctave)
	
	score = PairDiscovery(args.img1Path, args.img2Path, net, transform, args.tolerance, args.minFeatCC, args.margin, args.scaleImgRef, scaleList, args.houghInitial, args.nbSamplePoint, args.nbIter, args.saveQuality, args.computeSaliencyCoef, args.out1, args.out2)
	
	msg = 'score is {:.4f}'.format(score)
	print msg
	
	
