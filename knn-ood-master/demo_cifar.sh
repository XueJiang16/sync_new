python feat_extract.py --in-dataset CIFAR-100  --out-datasets SVHN lsun isun places365 dtd --name resnet18  --model-arch resnet18
python run_cifar.py --in-dataset CIFAR-100  --out-datasets SVHN lsun isun places365 dtd --name resnet18  --model-arch resnet18

#python feat_extract.py --in-dataset CIFAR-100  --out-datasets LSUN_FIX ImageNet_FIX Imagenet_resize CIFAR-100 --name resnet18  --model-arch resnet18
#python run_cifar.py --in-dataset CIFAR-100  --out-datasets LSUN_FIX ImageNet_FIX Imagenet_resize CIFAR-100 --name resnet18  --model-arch resnet18