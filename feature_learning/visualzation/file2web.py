
import argparse
import os
import re

parser = argparse.ArgumentParser('Visualizing Training sample, top200 pairs from randomly top 2000 pairs')

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
f.write('\t\tVisualize Training Sample\n')
f.write('\t</h1>\n')
f.write('</center>\n')
f.write('<div class=\"blank\"></div>\n')
f.write('<center>\n')
f.write('<div>\n')

f.write('</div>\n')

### ---HTML Table--- ###
f.write('<table>\n')
f.write('\t<tr>\n')
f.write('\t\t<th># Rank</th>\n')
f.write('\t\t<th>Img 1 </th>\n')
f.write('\t\t<th>Img 2 </th>\n')
f.write('\t</tr>\n')

nbPair = len(os.listdir(args.imgDir)) / 2 ## Nb of row

for j in range(nbPair) : 
	f.write('\t<tr >\n')
	msg = '\t\t<th>{:d}</th>\n'.format(j + 1) 
	f.write(msg)## Rank
	img1 = os.path.join(args.imgDir, 'Rank{:d}_1.jpg'.format(j))
	msg = '\t\t<td><a download=\"{}\" href=\"{}\" title="ImageName"> <img  src=\"{}\" /></a> </td>\n'.format(img1, img1, img1)
	f.write(msg)## Img 1
	img2 = os.path.join(args.imgDir, 'Rank{:d}_2.jpg'.format(j))
	msg = '\t\t<td><a download=\"{}\" href=\"{}\" title="ImageName"> <img  src=\"{}\" /></a> </td>\n'.format(img2, img2, img2)
	f.write(msg)## Img 2
	f.write('\t</tr>\n')
	
f.write('</table>\n')
f.write('</center>\n</div>\n </body>\n</html>\n')
f.close()

