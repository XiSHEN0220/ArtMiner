from torch.autograd import Variable
import torch
import PIL.Image as Image
from tqdm import tqdm
import os

def ImgResize(featMin, featMax, cropSize, strideNet, w, h) :

	ratio = float(w)/h
	if ratio < 1 :
		feat_h = featMax + 2 * cropSize
		feat_w = max(round(ratio * feat_h), featMin + 2 * cropSize)

	else :
		feat_w = featMax + 2 * cropSize
		feat_h = max(round(feat_w/ratio), featMin + 2 * cropSize)
	new_w = feat_w * strideNet
	new_h = feat_h * strideNet

	return int(new_w), int(new_h), float(new_w)/w, float(new_h)/h

## Calculate all the query feature, store in a dictionary
def QueryFeat(searchDir, label, featMin, featMax, cropSize, strideNet, margin, useGpu, transform, model) :
	print '\nGet query feature...\n'
	queryFeat = {}
	for category in tqdm(label.keys()) :
		if not queryFeat.has_key(category) :
			queryFeat[category] = []
		for item in tqdm(label[category]) :
			imgName = item['query'][0]
			imgPil=Image.open(os.path.join(searchDir, imgName)).convert('RGB')
			w, h = imgPil.size
			bb = item['query'][1]

			## resize the query image to a proper size
			_,_, wRatio, hRatio =  ImgResize(featMin, featMax, cropSize, strideNet,bb[2] - bb[0], bb[3] - bb[1])
			bb0, bb1, bb2, bb3 = bb[0] * wRatio, bb[1] * hRatio, bb[2] * wRatio, bb[3] * hRatio
			imgPil = imgPil.resize((int(w * wRatio), int(h * hRatio)))

			## margin setting to allevate zeropadding influence to the border
			wLeftMargin = min(int(bb0) / strideNet, margin)
			hTopMargin = min(int(bb1) / strideNet, margin)
			wRightMargin = min(int(imgPil.size[0] - bb2) / strideNet, margin)
			hBottomMargin = min(int(imgPil.size[1] - bb3) / strideNet, margin)

			imgPil = imgPil.crop( (bb0 - wLeftMargin * strideNet, bb1 - hTopMargin * strideNet, bb2 + wRightMargin * strideNet, bb3 + hBottomMargin * strideNet) )

			## deep feature
			imgPil=transform(imgPil)
			imgPil=imgPil.unsqueeze(0)
			imgFeat = Variable(imgPil, volatile = True).cuda() if useGpu else Variable(imgPil, volatile = True)
			imgFeat = model.forward(imgFeat).data

			imgFeat=imgFeat[:, :, hTopMargin + cropSize : imgFeat.size()[2] - (hBottomMargin + cropSize), wLeftMargin + cropSize : imgFeat.size()[3] - (wRightMargin + cropSize)].clone().contiguous()
			norm =torch.sum(torch.sum(torch.sum(imgFeat ** 2, 1, keepdim=True), 2, keepdim=True), 3, keepdim=True)  ** 0.5 + 1e-7
			imgFeat = imgFeat / norm.expand(imgFeat.size())
            print imgFeat.size()
			## store the features to accelerate the code
			queryFeat[category].append(imgFeat)

	return queryFeat

## Searching Image Dimension
def SearchImgDim(searchDir) :

	searchDim = {}
	for imgName in os.listdir(searchDir) :
		I = Image.open(os.path.join(searchDir, imgName)).convert('RGB')
		searchDim[imgName] = I.size

	return searchDim

## Search Image Feature
def SearchFeat(searchDir, featMin, scaleList, strideNet, useGpu, transform, net, searchName) :

	searchFeat = {}

	I = Image.open(os.path.join(searchDir, searchName)).convert('RGB')
	w,h = I.size

	for scale in scaleList :

		new_w, new_h, _, _ = ImgResize(featMin, scale, 0, strideNet, w, h)
		IPil = I.resize((new_w, new_h))
		IData = transform(IPil).unsqueeze(0)
		IData = Variable(IData, volatile = True).cuda() if useGpu else Variable(IData, volatile = True)
		feat = net.forward(IData).data
		searchFeat[str(scale)] = feat

	return searchFeat
