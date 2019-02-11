python visualize.py --outDir brueghel --searchRegion 2 --trainRegion 12 --validRegion 10 --modelPath ../../model/resnet18.pth  --searchDir ../../data/Brueghel/ --margin 5 --nbImgEpoch 200 --cuda --saveSize 256 --nbSearchImgEpoch 2000

python file2web.py --imgDir brueghel --outHtml brueghel.html

python visualize.py --outDir ltll --searchRegion 2 --trainRegion 12 --validRegion 10 --modelPath ../../model/resnet18.pth  --searchDir ../../data/Ltll/ --margin 5 --nbImgEpoch 200 --cuda --saveSize 256 --nbSearchImgEpoch 2000

python file2web.py --imgDir ltll --outHtml ltll.html

python visualize.py --outDir oxford --searchRegion 2 --trainRegion 12 --validRegion 10 --modelPath ../../model/resnet18.pth  --searchDir ../../data/Ltll/ --margin 5 --nbImgEpoch 200 --cuda --saveSize 256 --nbSearchImgEpoch 2000

python file2web.py --imgDir oxford --outHtml oxford.html

