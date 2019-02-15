

python dataset_discovery.py --computeSaliencyCoef --valOrTest test --outResJson out1.json --houghInitial --nbIter 1000 --minFeatCC 3 --finetunePath ../model/resnet18.pth
python dataset_discovery.py --computeSaliencyCoef --valOrTest test --outResJson out2.json --finetunePath ../model/ltllModel.pth --houghInitial --nbIter 1000 --minFeatCC 3




