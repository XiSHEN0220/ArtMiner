python oxford_retrieval.py --scaleImgRef 25 --houghInitial --cuda --featScaleBase 20 --finetunePath ../model/resnet.pth --valOrTest test --outResJson Oxford_resnet.json --scaleImgRef 25

python oxford_retrieval.py --scaleImgRef 25 --houghInitial --cuda --featScaleBase 20 --finetunePath ../model/oxfordModel.pth --valOrTest test --outResJson Oxford_finetune.json --scaleImgRef 25
