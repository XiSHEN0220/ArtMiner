from tqdm import tqdm 
import PIL.Image as Image
from torch.autograd import Variable 
import torch 

def QueryImgResize(featMin, featMax, cropSize, strideNet, w, h) :

	ratio = float(w)/h
	if ratio < 1 : 
		feat_h = featMax + 2 * cropSize
		feat_w = max(round(ratio * feat_h), featMin + 2 * cropSize)
		
	else : 
		feat_w = featMax + 2 * cropSize
		feat_h = max(round(feat_w/ratio), featMin + 2 * cropSize)
	new_w = feat_w * stride_size_net
	new_h = feat_h * stride_size_net

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
			_,_, wRatio, hRatio =  QueryImgResize(featMin, featMax, cropSize, strideNet,bb[2] - bb[0], bb[3] - bb[1])
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
		
			## store the features to accelerate the code 
			queryFeat[category].append(imgFeat)
			
	return queryFeat
	
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
	
def match(img_feat, kernel, cuda, eta, K, distance_name, metric, strd) :
	feat_out_dim = kernel.size()[0]
	img_feat = Variable(img_feat)
	kernel = Variable(kernel)
	
	score = metric(img_feat, kernel, cuda, eta, K, distance_name, strd)
	
	
	return score
	
def CosineSimilarity(img_feat, kernel, kernel_one) :

	dot = F.conv2d(img_feat, kernel, stride = 1)
	img_feat_norm = F.conv2d(img_feat ** 2, kernel_one, stride = 1) ** 0.5 + 1e-7
	score = dot/img_feat_norm.expand(dot.size())
	
	return score.data
