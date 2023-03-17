python feat_extract.py --in-dataset CIFAR-100  --out-datasets SVHN LSUN iSUN dtd places365 --name resnet18  --model-arch resnet18
python run_cifar.py --in-dataset CIFAR-100  --out-datasets SVHN LSUN iSUN dtd places365 --name resnet18  --model-arch resnet18

python feat_extract.py --in-dataset CIFAR-100  --out-datasets LSUN_FIX ImageNet_FIX Imagenet_resize CIFAR-100 --name resnet18  --model-arch resnet18
python run_cifar.py --in-dataset CIFAR-100  --out-datasets LSUN_FIX ImageNet_FIX Imagenet_resize CIFAR-100 --name resnet18  --model-arch resnet18