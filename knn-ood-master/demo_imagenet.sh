
python feat_extract_largescale.py --in-dataset imagenet_lta  --out-datasets inat sun50 places50 dtd  --name resnet50  --model-arch resnet50
python run_imagenet.py --in-dataset imagenet_lta --out-datasets inat sun50 places50 dtd  --name resnet50  --model-arch resnet50
