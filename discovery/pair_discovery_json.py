from pair_discovery import PairDiscovery
import numpy as np 
import outils
import ujson 
import argparse
import sys
sys.path.append("..")
from model.model import Model


from torchvision import datasets, transforms,models
from tqdm import tqdm


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
	'--nbSamplePoint', type=int, default = 3 , help='nb sample point = 2 ==> Hough, nb sample point = 3 ==> Affine, nb sample point = 4 ==> Homography')

parser.add_argument(
	'--nbIter', type=int, default = 1000 , help='nb iteration, nbIter = 1 ==> Hough transformation, parameter estimated with all points in the matchSet')

parser.add_argument(
	'--saveQuality', type=int, default = 1000, help='output image quality')
	
parser.add_argument(
	'--computeSaliencyCoef', action='store_true', help='using saliency coefficient for the feature of reference image?')

parser.add_argument(
	'--minFeatCC', type=int, default = 3, help='minimum number of features in CC')

parser.add_argument(
	'--architecture', type=str, default = 'resnet18', choices = ['resnet18', 'resnet34'], help='which architecture, resnet18 or resnet34, by default is resnet18')
	
parser.add_argument(
	'--jsonPairFile', type=str, help='json pair file')
	
parser.add_argument(
	'--indexBegin', type=int, help='begin index')
	
parser.add_argument(
	'--indexEnd', type=int, help='end index')
	
parser.add_argument(
	'--outJson', type=str, help='output json file')


args = parser.parse_args()
print args

net = Model(args.finetunePath, args.architecture)

if args.cuda:
	net.cuda()
net.eval()

transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
				std = [ 0.229, 0.224, 0.225 ]),
])

scaleList = outils.ScaleList(args.featScaleBase, args.nbOctave, args.scalePerOctave)

with open(args.jsonPairFile, 'r') as f :
	pairs = ujson.load(f)

results = []
count = 0
for i in tqdm(range(args.indexBegin, args.indexEnd)) : 
	score1 = PairDiscovery(pairs[i][0].encode('utf-8'), pairs[i][1].encode('utf-8'), net, transform, args.tolerance, args.minFeatCC, args.margin, args.scaleImgRef, scaleList, args.houghInitial, args.nbSamplePoint, args.nbIter, args.saveQuality, args.computeSaliencyCoef, None, None, False)
	score2 = PairDiscovery(pairs[i][1].encode('utf-8'), pairs[i][0].encode('utf-8'), net, transform, args.tolerance, args.minFeatCC, args.margin, args.scaleImgRef, scaleList, args.houghInitial, args.nbSamplePoint, args.nbIter, args.saveQuality, args.computeSaliencyCoef, None, None, False)
	info = [pairs[i][0].encode('utf-8'), pairs[i][1].encode('utf-8'), score1] if score1 > score2 else [pairs[i][1].encode('utf-8'), pairs[i][0].encode('utf-8'), score2]
	results.append(info)
	count += 1
	if count % 1000 == 999 : 
		msg = '{:d} pairs are processed...'.format(count)
		print (msg)
		with open(args.outJson, 'w') as f :
			ujson.dump(results, f)
with open(args.outJson, 'w') as f :
	ujson.dump(results, f)
#msg = 'score is {:.4f}'.format(score)
#print msg
	
