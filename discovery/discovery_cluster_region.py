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
from itertools import combinations

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
	'--scoreThreshold', type=float, default= 0.015, help='score threshold')

parser.add_argument(
	'--iouThreshold', type=float, default= 0.3, help='iou threhsold')

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
	if len(cluster) <= 3 : 
		continue
	pairInCluster = [j for j in validIndex if nodeDict[pairs[j][0]] in cluster]
	pairMask = []
	
	for j in pairInCluster : 
		img1, img2, score = pairs[j]
		out1 = os.path.join(args.outDir, 'item1.png')
		out2 = os.path.join(args.outDir, 'item2.png')
		PairDiscovery(img1.encode('utf-8'), img2.encode('utf-8'), net, transform, args.tolerance, args.minFeatCC, args.margin, args.scaleImgRef, scaleList, args.houghInitial, args.nbSamplePoint, args.nbIter, args.saveQuality, args.computeSaliencyCoef, out1, out2, True)
		
		I1 = Image.open(img1).convert('RGB')
		w, h = I1.size
		ratio = max(w / 100., h /100. )
		w, h = int(w / ratio), int(h / ratio)
		m1 = misc.imresize(np.array(Image.open(out1))[:, :, 3] * 1.0, (h,w)) > 128
		
		
		I2 = Image.open(img2).convert('RGB')
		w, h = I2.size
		ratio = max(w / 100., h /100. )
		w, h = int(w / ratio), int(h / ratio)
		m2 = misc.imresize(np.array(Image.open(out2))[:, :, 3] * 1.0, (h,w)) > 128
		pairMask.append([img1, img2, np.expand_dims(m1, axis=0), np.expand_dims(m2, axis=0) ])
		
	
	graph = np.zeros((len(pairMask), len(pairMask)))
	
	for j,k in combinations(range(len(pairMask)), 2) : 
		
		if pairMask[j][0] == pairMask[k][0] and np.sum(pairMask[j][2] & pairMask[k][2]) / (np.sum(pairMask[j][2] | pairMask[k][2]).astype(np.float32)) >  args.iouThreshold:
			graph[j, k] = 1
			continue
		
		if pairMask[j][0] == pairMask[k][1] and np.sum(pairMask[j][2] & pairMask[k][3]) / (np.sum(pairMask[j][2] | pairMask[k][3]).astype(np.float32)) >  args.iouThreshold:
			graph[j, k] = 1
			continue
			
		if pairMask[j][1] == pairMask[k][0] and np.sum(pairMask[j][3] & pairMask[k][2]) / (np.sum(pairMask[j][3] | pairMask[k][2]).astype(np.float32)) >  args.iouThreshold:
			graph[j, k] = 1
			continue
		
		if pairMask[j][1] == pairMask[k][1] and np.sum(pairMask[j][3] & pairMask[k][3]) / (np.sum(pairMask[j][3] | pairMask[k][3]).astype(np.float32)) >  args.iouThreshold:
			graph[j, k] = 1
			continue
	
	graph = graph + graph.T
	graph = csr_matrix(graph)
	nbSubCC, subCCLabel = connected_components(csgraph=graph, directed=False, return_labels=True)
	
	for j in range(nbSubCC) : 
		cluster = np.where(subCCLabel == j)[0] 
		if len(cluster) > 3 : 
			outDir = os.path.join(args.outDir, 'cluster{:d}'.format(clusterID))
			os.mkdir(outDir)
			imgDict = {}
			for k in cluster :
				img1, img2, m1, m2 = pairMask[k]
				
				if imgDict.has_key(img1) : 
					imgDict[img1].append(m1)
				else : 
					imgDict[img1] = [m1]
				
				if imgDict.has_key(img2) : 
					imgDict[img2].append(m2)
				else : 
					imgDict[img2] = [m2]
				
			for k, key in enumerate(imgDict.keys()) : 
				m = np.mean((np.concatenate(imgDict[key] , axis=0) * 255.), axis= 0)
				
				Iorg = np.array(Image.open(key).convert('RGB'))
				h, w, _ = Iorg.shape
				m = misc.imresize(m, (h,w))
				m[m<140] = 140
				I = Image.fromarray(np.concatenate((Iorg, m.astype(np.uint8).reshape((h,w,1))), axis=2))
				out = os.path.join(outDir, 'img{:d}.png'.format(k + 1))
				I.save(out)
			clusterID += 1
cmd = 'rm {}*'.format(os.path.join(args.outDir, 'item'))
print cmd 
os.system(cmd)
	
	

