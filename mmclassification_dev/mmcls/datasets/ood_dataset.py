import random
import warnings
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import mmcv
import numpy as np
from mmcv import FileClient
from torch.utils.data import Dataset
import torch
import json
from PIL import Image
import torchvision as tv
import os
import copy
from collections import Counter

# from .base_dataset import BaseDataset
from .builder import DATASETS
from .pipelines import Compose

class OODBaseDataset(Dataset):
    def __init__(self, name, pipeline, noise_engine=None, len_limit=-1):
        super().__init__()
        self.pipeline = Compose(pipeline)
        self.file_list = []
        self.data_prefix = None
        self.name = name
        self.transform = tv.transforms.Compose([
            tv.transforms.Resize(256),
            tv.transforms.CenterCrop(224),
            # tv.transforms.Resize((480, 480)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize([123.675/255, 116.28/255, 103.53/255],
                                    [58.395/255, 57.12/255, 57.375/255]),
        ])
        self.noise_engine = noise_engine
        self.len_limit = len_limit
        self.data_infos = []

    def parse_datainfo(self):
        random.seed(111)
        random.shuffle(self.file_list)
        for sample in self.file_list:
            info = dict(img_prefix=self.data_prefix)
            info['img_info'] = {'filename': sample}
            info['filename'] = sample
            info['type'] = 3  # no type
            info['label'] = -1  # no label
            self.data_infos.append(info)

    def __len__(self):
        return self.len_limit if self.len_limit!=-1 else len(self.file_list)
        # return len(self.file_list)

    def prepare_data(self, idx):
        results = copy.deepcopy(self.data_infos[idx])
        sample = Image.open(os.path.join(results['img_prefix'], results['img_info']['filename']))
        if sample.mode != 'RGB':
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        if self.noise_engine == "uniform":
            sample = sample + ((torch.rand_like(sample) - 0.5) / 5)
        results['img'] = sample
        return self.pipeline(results)

    def __getitem__(self, idx):
        return self.prepare_data(idx)


@DATASETS.register_module()
class TxtDataset(OODBaseDataset):
    def __init__(self, name, path, data_ann, pipeline, train_label=None, **kwargs):
        super().__init__(name, pipeline, **kwargs)
        self.data_prefix = path
        # self.file_list = glob.glob(os.path.join(path, '*'))
        self.data_ann = data_ann
        self.train_label = train_label
        with open(self.data_ann) as f:
            samples = [x.strip().rsplit(' ', 1) for x in f.readlines()]
        for filename in samples:
            self.file_list.append(filename)
        self.parse_datainfo()

    def parse_datainfo(self):
        random.seed(111)
        random.shuffle(self.file_list)
        if self.train_label is not None:
            train_labels = []
            with open(self.train_label, 'r') as f:
                for line in f.readlines():
                    segs = line.strip().split(' ')
                    train_labels.append(int(segs[-1]))
            train_label_index = Counter(train_labels)

        for sample in self.file_list:
            info = dict(img_prefix=self.data_prefix)
            info['img_info'] = {'filename': sample[0]}
            info['filename'] = sample[0]
            gt_label = int(sample[-1])
            info['label'] = gt_label
            if self.train_label is not None:
                freq = train_label_index[gt_label]
                if freq > 100:
                    info['type'] = 0  # head
                elif freq < 20:
                    info['type'] = 2  # tail
                else:
                    info['type'] = 1  # mid
            else:
                info['type'] = 3  # no type
            self.data_infos.append(info)

@DATASETS.register_module()
class JsonDataset(OODBaseDataset):
    # INaturalist
    def __init__(self, name, path, data_ann, pipeline, **kwargs):
        super().__init__(name, pipeline, **kwargs)
        # self.file_list = glob.glob(os.path.join(path, '*'))
        self.data_prefix = path
        self.data_ann = data_ann
        with open(self.data_ann) as f:
            ann = json.load(f)
        images = ann['images']
        images_dict = dict()
        for item in images:
            images_dict[item['id']] = item['file_name']
        annotations = ann['annotations']
        samples = []
        for item in annotations:
            samples.append(images_dict[item['image_id']])
        for filename in samples:
            self.file_list.append(filename)
        self.parse_datainfo()


@DATASETS.register_module()
class FolderDataset(OODBaseDataset):
    def __init__(self, name, path, pipeline, data_ann=None, **kwargs):
        super().__init__(name, pipeline, **kwargs)
        # self.file_list = glob.glob(os.path.join(path, '*'))
        self.data_prefix = path
        images = os.listdir(path)
        for filename in images:
            self.file_list.append(filename)
        self.parse_datainfo()