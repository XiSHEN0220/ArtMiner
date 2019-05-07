
import argparse
import os
import re
parser = argparse.ArgumentParser()

parser.add_argument(
	'--outHtml', type=str, help='output html file')

parser.add_argument(
	'--imgDir', type=str, help='image directory')

parser.add_argument(
	'--scoreTH', type=float, help='score threshold')

args = parser.parse_args()



### Writing the table format###
f = open(args.outHtml, 'w')
f.write('<html>\n')
f.write('<head>\n')
f.write('\t<title></title>\n')
f.write('\t<meta name=\"keywords\" content= \"Visual Result\" />  <meta charset=\"utf-8\" />\n')
f.write('\t<meta name=\"robots\" content=\"index, follow\" />\n')
f.write('\t<meta http-equiv=\"Content-Script-Type\" content=\"text/javascript\" />\n')
f.write('\t<meta http-equiv=\"expires\" content=\"0\" />\n')
f.write('\t<meta name=\"description\" content= \"Project page of style.css\" />\n')
f.write('\t<link rel=\"stylesheet\" type=\"text/css\" href=\"style.css\" media=\"screen\" />\n')
f.write('\t<link rel=\"shortcut icon\" href=\"favicon.ico\" />\n')
f.write('</head>\n')
f.write('<body>\n')
f.write('<div id="website">\n')
f.write('<center>\n')
f.write('\t<div class=\"blank\"></div>\n')
f.write('\t<h1>\n')
f.write('\t\tVisual Results\n')
f.write('\t</h1>\n')
f.write('</center>\n')
f.write('<div class=\"blank\"></div>\n')
f.write('<center>\n')
f.write('<div>\n')

f.write('</div>\n')

### --- ###

f.write('<table>\n')
Count = 0
f.write('<table>\n')
f.write('\t<tr>\n')
f.write('\t\t<th>No.</th>\n')
f.write('\t\t<th>I1</th>\n')
f.write('\t\t<th>I2 </th>\n')
f.write('\t\t<th>Score </th>\n')

f.write('\t</tr>\n')

pairList = os.listdir(args.imgDir)
pairList = [item for item in pairList if '_I1_' in item]
pairList = sorted(pairList, key=lambda s: float(s[:-4].split('S')[-1]), reverse=True)
count = 1
for i in range(len(pairList)) : 
	imgName1 = pairList[i]
	score = float(imgName1[:-4].split('S')[-1])
	imgName2 = imgName1.replace('_I1_', '_I2_')
	imgPath1, imgPath2 = os.path.join(args.imgDir, imgName1), os.path.join(args.imgDir, imgName2)
	if score > args.scoreTH : 
		f.write('\t<tr>\n')
		msg = '\t\t<td>{:d}</td>\n'.format(count)
		f.write(msg)
		msg = '\t\t<td> <a download=\" {} \" href=\"{}\" title="ImageName"> <img  src=\"{}\" /></a></td>\n'.format(imgPath1, imgPath1, imgPath1)
		f.write(msg)
		msg = '\t\t<td> <a download=\" {} \" href=\"{}\" title="ImageName"> <img  src=\"{}\" /></a></td>\n'.format(imgPath2, imgPath2, imgPath2)
		f.write(msg)
		msg = '\t\t<td>{:.4f}</td>\n'.format(score)
		f.write(msg)
		f.write('\t</tr>\n')
		count += 1
	else : 
		break
	
	
	
	

	
f.write('</table>\n')
f.write('</center>\n</div>\n </body>\n</html>\n')
f.close()

