#python train.py --topKLoss 10 --tripleLossThreshold 0.8 --outDir Oxford_K10_TH08_S2T12V10_100Img --cuda --finetunePath ../model/resnet18.pth --searchDir ../data/Oxford5K/img/ --nbImgEpoch 100


#python train.py --topKLoss 10 --tripleLossThreshold 0.8 --outDir Oxford_K10_TH08_S2T12V10_100Img_Query --cuda --finetunePath ../model/resnet18.pth --searchDir ../data/Oxford5K/img/ --nbImgEpoch 100

python trainOxford.py --topKLoss 10 --tripleLossThreshold 0.8 --outDir Oxford_K10_TH08_S2T12V10_200ImgLabel --cuda --finetunePath ../model/resnet18.pth --architecture resnet18 --searchDir ../data/Oxford5K/img/ --nbImgEpoch 200 --queryScale 25 30 35
