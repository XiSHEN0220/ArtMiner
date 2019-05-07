
import argparse
import os
import re
parser = argparse.ArgumentParser()

parser.add_argument(
	'--outHtml', type=str, help='output html file')

parser.add_argument(
	'--imgDir', type=str, help='image directory')

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


clusterDir = os.listdir(args.imgDir)

for i in range(len(clusterDir)) : 
	clusterPath = os.path.join(args.imgDir, clusterDir[i])
	
	
	
	f.write('<table>\n')
	caption = '\t\t<caption>Cluster {:d}</caption>\n'.format(i)
	f.write(caption)
	f.write('\t<tr>\n')
	imgs = os.listdir(clusterPath)
	for j in range(len(imgs)) : 
		imgPath = os.path.join(clusterPath, imgs[j])
		msg = '\t\t<td> <a download=\" {} \" href=\"{}\" title="ImageName"> <img  src=\"{}\" /></a></td>\n'.format(imgPath, imgPath, imgPath)
		f.write(msg)
		if j % 10 == 9 :
			f.write('\t</tr>\n')
			f.write('\t<tr>\n')
	f.write('\t</tr>\n')
	
	f.write('</table>\n')
	
f.write('</center>\n</div>\n </body>\n</html>\n')
f.close()

