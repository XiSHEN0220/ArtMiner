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
from scipy.sparse import csr_matrix
from scipy import misc
from scipy.sparse.csgraph import connected_components
import os
import PIL.Image as Image

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
	'--jsonPairScoreFile', type=str, help='json pair file with score')
	
parser.add_argument(
	'--outDir', type=str, help='output directory')

parser.add_argument(
	'--scoreThreshold', type=float, default= 0.015, help='output directory')

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


## Construct Graph
with open(args.jsonPairScoreFile, 'r') as f :
	pairs = ujson.load(f)

nodeDict = {}
validIndex = []

for i, (img1, img2, score) in enumerate(pairs) : 
	if score >  args.scoreThreshold : 
		validIndex.append(i)
		if img1 not in nodeDict.keys() : 
			nodeDict[img1] = len(nodeDict.keys())
		
		if img2 not in nodeDict.keys() : 
			nodeDict[img2] = len(nodeDict.keys())
		
graph = np.zeros((len(nodeDict.keys()), len(nodeDict.keys())))
for i in validIndex : 
	img1, img2, score = pairs[i]
	graph[nodeDict[img1], nodeDict[img2]] = 1
graph = graph + graph.T
graph = csr_matrix(graph)
nbCC, CCLabel = connected_components(csgraph=graph, directed=False, return_labels=True)

if nbCC > 0 and not os.path.exists(args.outDir) : 
	os.mkdir(args.outDir)

clusterID = 1
for i in range(nbCC) : 
	cluster = np.where(CCLabel == i)[0]
	if len(cluster) >= 3 :
		
		outDir = os.path.join(args.outDir, 'cluster{:d}'.format(clusterID))
		os.mkdir(outDir)
		pairInCluster = [j for j in validIndex if nodeDict[pairs[j][0]] in cluster]
		clusterDict = {}
		count = 0
		for j in pairInCluster : 
			img1, img2, score = pairs[j]
			out1 = os.path.join(outDir, 'item{:d}.png'.format(count))
			count += 1 
			out2 = os.path.join(outDir, 'item{:d}.png'.format(count))
			count += 1
			if img1 not in clusterDict : 
				clusterDict[img1] = []
			if img2 not in clusterDict : 
				clusterDict[img2] = []
			clusterDict[img1].append(out1)
			clusterDict[img2].append(out2)
			PairDiscovery(img1.encode('utf-8'), img2.encode('utf-8'), net, transform, args.tolerance, args.minFeatCC, args.margin, args.scaleImgRef, scaleList, args.houghInitial, args.nbSamplePoint, args.nbIter, args.saveQuality, args.computeSaliencyCoef, out1, out2, True)
			
		for j, key in enumerate(clusterDict.keys()) : 
			nbImg = len(clusterDict[key])
			Iorg = np.array(Image.open(key).convert('RGB'))
			h, w, _ = Iorg.shape
			mask = np.zeros((h, w))
			
			for k in range(nbImg) : 
				m = np.array(Image.open(clusterDict[key][k]))[:, :, 3] 
				
				mask = mask + misc.imresize(m, (h,w)) * 1.0
			mask = mask / nbImg
			I = Image.fromarray(np.concatenate((Iorg, mask.astype(np.uint8).reshape((h,w,1))), axis=2))
			out = os.path.join(outDir, 'img{:d}.png'.format(j + 1))
			I.save(out)
		cmd = 'rm {}*'.format(os.path.join(outDir, 'item'))
		print cmd 
		os.system(cmd)
		
				
		clusterID += 1
		

