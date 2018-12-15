from prettytable import PrettyTable
import numpy as np
from tqdm import tqdm

def localization(res, label, IoUThresh = 0.7, nbPred = 100) :
	"""
	calculate mAP for Pattern Localization task
	inputs:
			loc_result: 	'1.png' : [{'150.png' : [25, 28, 65, 79]}, {'523.png' : [122, 200, 158, 400]}]\n
								...

			ground_truth: a dictionary with key img name, value a sub-dictionary
			each sub-dictionary is a dictionary with key image name, value bbox position

			dict_cate_imgname: a dictionary with key category name, query image name
	"""
	mAPPerQuery = {}
	mAPPerCat = {}
	mAP = []

	print 'Pattern Localization Per Query...'

	for category in tqdm(res.keys()) :
		mAPPerQuery[category] = []
		for i, item in enumerate(res[category]) :
			gt = label[category][i]['gt']
			flagGt = np.ones(len(gt))
			nbGt = len(gt)
			queryRes = []

			for j, (predName, _, bb) in enumerate(item) :
				if j >= nbPred :
					break
				for k, gtItem in enumerate(gt) :
					if flagGt[k] and predName == gtItem[0] and outils.IoU(bb, np.array(gtItem[1]).reshape((1, -1)))[0] > IoUThresh :
						queryRes.append((len(queryRes) + 1) / float(j + 1))
						flagGt[k] = 0
						res[category][i][j]= (res[category][i][j][0], k+1, res[category][i][j][2])
			mAPPerQuery[category].append(np.sum(queryRes)/nbGt)
			mAP.append(np.sum(queryRes)/nbGt)
		mAPPerCat[category] = np.mean(mAPPerQuery[category])


	queryTable = PrettyTable()
	queryTable.field_names = ['Query name', 'Average Precision']

	for category in mAPPerQuery.keys() :
		for i in range(len(mAPPerQuery[category])) :
			queryTable.add_row([label[category][i]['query'][0], str(mAPPerQuery[category][i])])

	queryTable.add_row(['mAP', str(np.mean(mAP))])

	print 'IoU threshold : %.2f\tNumber prediction : %d'%(IoUThresh, nbPred)
	print queryTable


	catTable = PrettyTable()
	catTable.field_names = ['Category name', 'Average Precision']

	for key in mAPPerCat.keys() :
		catTable.add_row([key, str(mAPPerCat[key])])

	catTable.add_row(['mAP', str(np.mean(mAPPerCat.values()))])

	print 'IoU threshold : %.2f\tNumber prediction : %d'%(IoUThresh, nbPred)
	print catTable

	return res, queryTable, catTable, mAPPerQuery
