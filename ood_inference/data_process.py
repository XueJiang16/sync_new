import torchvision as tv
from PIL import Image

transform_imagenet = tv.transforms.Compose([
    tv.transforms.Resize(256),
    # tv.transforms.Resize(self.resize_size),
    # tv.transforms.Resize(248, interpolation=tv.transforms.InterpolationMode.BICUBIC),
    tv.transforms.CenterCrop(224),
    # tv.transforms.CenterCrop(self.crop_size),
    # tv.transforms.Resize((480, 480)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize([123.675/255, 116.28/255, 103.53/255],
                            [58.395/255, 57.12/255, 57.375/255]),
])

transform_cifar = tv.transforms.Compose([
    tv.transforms.Resize(32),
    tv.transforms.CenterCrop(32),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize([0.4914, 0.4822, 0.4465],
                            [0.2023, 0.1994, 0.2010]),
    # tv.transforms.Normalize([129.304/255, 124.07/255, 112.434/255],
    #                         [68.17/255, 65.392/255, 70.418/255]),
])


def prepare_data(filename, transform):
    results = dict()
    sample = Image.open(filename)
    if sample.mode != 'RGB':
        sample = sample.convert('RGB')
    sample = transform(sample)
    results['img'] = sample.unsqueeze(0)
    results['type'] = 3
    return results
