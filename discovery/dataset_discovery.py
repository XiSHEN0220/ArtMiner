import pair_discovery 
import outils
import os 
import sys
from tqdm import tqdm
import argparse

sys.path.append("..")


from model.model import Model
import matplotlib.pyplot as plt
from torchvision import datasets, transforms,models
import ujson 

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
	'--tolerance', type=float , default = 2., help='tolerance expressed by nb of features (2 for retrieval with image 1 for retrieval with region)')

parser.add_argument(
	'--scaleImgRef', type=int , default = 40, help='maximum feature in the target image')
	
parser.add_argument(
	'--houghInitial', action='store_true', help='sampling point from hough transform sets')
	
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

parser.add_argument(
	'--labelJson', type=str, default = '../data/ltllLabel.json', help='labels of cross domain matching')

parser.add_argument(
	'--valOrTest', type=str, default = 'val', help='validation or test, default : val')

parser.add_argument(
	'--searchDir', type=str, default = '../data/Ltll/', help='searching image dataset')

parser.add_argument(
	'--outResJson', type=str, default = 'res.json', help='output json file to store the results')

args = parser.parse_args()
print args
	
net = Model(args.imagenetFeatPath, args.finetunePath)
net.cuda() ## Not support cpu version
net.eval()

transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
				std = [ 0.229, 0.224, 0.225 ]),
])

scaleList = outils.ScaleList(args.featScaleBase, args.nbOctave, args.scalePerOctave)


with open(args.labelJson, 'r') as f :
	label = ujson.load(f)
	
res = {}
for sourceImgName in tqdm(label[args.valOrTest]) :
	sourceImgPath = os.path.join(args.searchDir, sourceImgName)
	res[sourceImgName] = []
	for targetImgName in tqdm(label['searchImg']) : 
		targetImgPath = os.path.join(args.searchDir, targetImgName)
		score = pair_discovery.PairDiscovery(sourceImgPath, targetImgPath, net, transform, args.tolerance, args.minFeatCC, args.margin, args.scaleImgRef, scaleList, args.houghInitial, args.nbSamplePoint, args.nbIter, args.saveQuality, args.computeSaliencyCoef, None, None, False)
		res[sourceImgName].append((targetImgName, score))
	
nbSourceImg = len(res.keys())
truePosCount = 0
for sourceImgName in res.keys() : 
	res[sourceImgName] = sorted(res[sourceImgName], key=lambda s: s[1], reverse=True)
	if label['annotation'][sourceImgName] == label['annotation'][res[sourceImgName][0][0]] : 
		truePosCount += 1
		
res['accuracy'] = truePosCount / float(nbSourceImg)
msg = '***** Final accuracy is {:.3f} *****'.format(res['accuracy'])
print msg

with open(args.outResJson, 'w') as f : 
	ujson.dump(res, f)


