python pair_discovery.py --cuda --tolerance 2. --scaleImgRef 40 --houghInitial --img1Path ../data/Brueghel/Paradise\ with\ the\ Fall\ of\ Man\ \(The\ Hague\).jpg --img2Path ../data/Brueghel/A\ Paradise\ scene\ \(Sotheby\'s\,\ London\,\ 1985\).jpg --out1 FeatImageNet1.png --out2 FeatImageNet2.png --computeSaliencyCoef --nbIter 1000

python pair_discovery.py --cuda --tolerance 2. --scaleImgRef 40 --houghInitial --img1Path ../data/Brueghel/Paradise\ with\ the\ Fall\ of\ Man\ \(The\ Hague\).jpg --img2Path ../data/Brueghel/A\ Paradise\ scene\ \(Sotheby\'s\,\ London\,\ 1985\).jpg --out1 FeatBrueghel1.png --out2 FeatBrueghel2.png --finetunePath ../model/brueghelModel.pth --computeSaliencyCoef --nbIter 1000

