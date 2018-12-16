import sys
sys.path.append("..")
from net.net import net

import os
import PIL.Image as Image
from torchvision import datasets, transforms,nets

from tqdm import tqdm
from shutil import copyfile
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

import evaluation # evalution.retrieval, evalution.localization
import distance # metric.match
import outils #outils.nms
import illustration # illustration.draw_bb
import search_feat # search_feat.search_dim

import option #option.options
import ujson


import argparse 

parser = argparse.ArgumentParser()

parser.add_argument(
	'--outDir', type=str , help='output net directory')

##---- Query Information ----####

parser.add_argument(
	'--labelJson', type=str , help='label json file')

##---- Search Dataset Setting ----####
parser.add_argument(
	'--featScaleBase', type=int, default= 20, help='minimum # of features in the scale list ')
	
parser.add_argument(
	'--scalePerOctave', type=int, default= 3, help='# of scales in one octave ')
	
parser.add_argument(
	'--nbOctave', type=int, default= 2, help='# of octaves')


##---- Training parameters ----####

parser.add_argument(
	'--imagenetFeatPath', type=str, default='../../pre-trained-nets/resnet18.pth', help='imageNet feature net weight path')

parser.add_argument(
	'--finetunePath', type=str, help='finetune net weight path')

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

parser.add_argument(
	'--shuffle', action='store_true', help='shuffle data or not')

args = parser.parse_args()
tqdm.monitor_interval = 0
print args





	
def get_detection_res(res, search_dim, query, strideNet, crop_size, nbPred, label, stride) : 
	det = {}
	new_res = {}
	for category in res.keys():
		if not det.has_key(category) : 
			det[category] = []
			new_res[category] = []
		for j, item in enumerate(res[category]):
			det[category].append([])
			
			kernel_size = query[category][j].size()[2:]
			query_name = label[category][j]['query'][0]
			query_bbox = label[category][j]['query'][1]
            
			bbs = []
			infos = []
			searchs = []
			for searchName in item.keys():
				info_find = item[searchName]
				img_size = search_dim[searchName]
				bb, info_find = outils.get_bb(info_find, kernel_size, img_size, strideNet, crop_size, stride)
				if query_name == searchName:
					iou = outils.IoU(np.array(query_bbox), bb)
					pick = np.where(iou == 0)[0]
					if len(pick) ==0 :
						continue
					bb = bb[pick]
					info_find = [info_find[i] for i in pick]
                    
				bbs.append(bb)
				infos = infos + info_find
				searchs = searchs + [searchName for i in range(len(bb))]
			bbs = np.concatenate(bbs, axis=0)
			index = np.argsort(bbs[:, -1])[::-1]
			bbs = bbs[index]
			infos = [infos[i] for i in index]
			searchs = [searchs[i] for i in index]
			
			det[category][j] = [(searchs[i], bbs[i, -1], bbs[i, :4].astype(int)) for i in range(nbPred)]
			new_res[category].append([(searchs[i], infos[i][0], infos[i][1], infos[i][2], infos[i][3]) for i in range(nbPred)])
			
	return det, new_res


def Retrieval(searchDir, 
			featMax, 
			scaleList, 
			strideNet,
			crop_size, 
			cuda, 
			transform,
			net, 
			queryFeat,
			resDict,
			nbPred,
			label,
			stride) : 
	
	
	print 'Get search image dimension...'
	searchDim = outils.SearchImgDim(searchDir)

	for k, searchName in enumerate(tqdm(os.listdir(searchDir))) : 
	
		search_feat_dict = outils.SearchFeat(searchDir, featMax, scaleList, strideNet, cuda, transform, net, searchName)
		
		for query_category in query.keys() :
			for j_, feat_query in enumerate(query[query_category]) :	
			
				flip_find = []
				w_find = []
				h_find = []
				score_find = []
				scale_find = []
				
				for scale_name in search_feat_dict.keys() : 
				
					feat_img = search_feat_dict[scale_name]
					score = distance.match(feat_img, feat_query, cuda, eta, K, distance_name, metric, stride)
					
					score, _ = score.max(dim = 1)
					
					w,h = score.size()[1], score.size()[2]
					
					score, find_flip = score.max(dim = 0)
					score, index = score.view(1, -1).topk(min(featMax, score.numel()))
					find_flip = find_flip.view(1, -1).squeeze(0)[index[0]]
					find_w, find_h = index/h, index%h
					
					flip_find.append(find_flip)
					w_find.append(find_w)
					h_find.append(find_h)
					score_find.append(score)
					scale_find = scale_find + [int(scale_name) for i in range(len(find_flip))]
				flip_find = torch.cat(flip_find, dim=0)
				w_find = torch.cat(w_find, dim=1)
				h_find = torch.cat(h_find, dim=1)
				score_find = torch.cat(score_find, dim=1)
				
				_, index_keep = torch.sort(score_find, descending = True)
				index_keep = index_keep[0, :min(5 * featMax, index_keep.numel())]
				info_find = [(flip_find[i], w_find[0, i], h_find[0, i], scale_find[i], score_find[0, i]) for i in index_keep]
				res[query_category][j_][searchName] = info_find
				
				
		
	det, res = get_detection_res(res, search_dim, query, strideNet, crop_size, nbPred, label, stride)
		
	return det, res
	



transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
				std = [ 0.229, 0.224, 0.225 ]),
])

## net Initialize
strideNet = 16
featChannel = 256
net = net(args.imagenetFeatPath, args.finetunePath)
if args.cuda:
	net.cuda()

net.eval()

with open(args.labelJson, 'r') as f :
	label = ujson.load(f)

## get query feature
queryFeat = outils.QueryFeat(args.searchDir, label, args.featMin, args.featMax, args.cropSize, strideNet, args.margin, args.cuda, transform, net)

## Initialize dictionary to store results
resDict = outils.ResDictInit(queryFeat, args.searchDir)


scaleList = outils.ScaleList(args.featScaleBase, args.nbOctave, args.scalePerOctave)

det, res = find_patch(args.searchDir, 
			args.featMax, 
			scaleList, 
			args.
			args.strideNet,
			args.crop_size, 
			args.cuda, 
			transform,
			net, 
			query, 
			args.search_flip, 
			args.eta,
			args.K,
			args.metric,
			res,
			args.out_pred,
			label,
			args.stride)
	

	for category in tqdm(det.keys()) : 
		for i in range(len(det[category])) :
			for j in range(len(det[category][i])):
				det[category][i][j] = (det[category][i][j][0], 0, det[category][i][j][2])

	det, query_table_loc, category_table_loc = evaluation.localization(det, label, IoU_thresh = args.IoU_threshold, nbPred = args.out_pred)
	
	f = open(args.detection_txt, 'w') 
	f.write ('\t\t\t Localization Result of Brueghel (IoU = 0.3) \t\t\tNumber prediction %d\n'%args.out_pred)
	f.write(query_table_loc.get_string())
	f.write(category_table_loc.get_string())
	f.close()

	if args.detection_pkl :
		with open(args.detection_pkl, 'w') as f :
			ujson.dump(det, f)
	with open('toto.ujson', 'w') as f :
		ujson.dump(res, f)
				
	if args.nb_draw > 0 : 
		illustration.draw_bb(args.visual_root, args.searchDir, label, args.nb_draw, det)	   
		
def run():
	# figure out the experiments type
	args = option.options().parse()
	
	if args.subcommand is None:
		raise ValueError("ERROR: specify the dataset")
	if args.cuda and not torch.cuda.is_available():
		raise ValueError("ERROR: cuda is not available, try running on CPU")


	main(args)
	
if __name__ == "__main__":
   run()	
		
		
