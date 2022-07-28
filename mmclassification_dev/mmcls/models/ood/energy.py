from mmcv.runner import BaseModule
import torch
import os
import numpy as np
from collections import Counter

from ..builder import OOD
from mmcls.models import build_classifier


@OOD.register_module()
class Energy(BaseModule):
    def __init__(self, classifier, num_classes, temperature=1, **kwargs):
        super(Energy, self).__init__()
        self.local_rank = os.environ['LOCAL_RANK']
        self.classifier = build_classifier(classifier)
        self.classifier.eval()
        self.num_classes = num_classes
        self.temperature = temperature
        # self.criterion = torch.nn.CrossEntropyLoss().to("cuda:{}".format(self.local_rank))


    def forward(self, **input):
        if "type" in input:
            type = input['type']
            del input['type']
        with torch.no_grad():
            outputs = self.classifier(return_loss=False, softmax=False, post_process=False, **input)
            confs = self.temperature * torch.logsumexp(outputs / self.temperature, dim=1)
        return confs, type

@OOD.register_module()
class EnergyCustom(BaseModule):
    def __init__(self, classifier, num_classes, temperature=1, target_file=None, target_noise=0,**kwargs):
        super(EnergyCustom, self).__init__()
        self.local_rank = os.environ['LOCAL_RANK']
        self.classifier = build_classifier(classifier)
        self.classifier.eval()
        self.num_classes = num_classes
        self.temperature = temperature
        self.criterion = torch.nn.Softmax(dim=-1).to("cuda:{}".format(self.local_rank))
        if target_file is not None:
            cls_idx = []
            with open(target_file, 'r') as f:
                for line in f.readlines():
                    segs = line.strip().split(' ')
                    cls_idx.append(int(segs[-1]))
            cls_idx = np.array(cls_idx, dtype='int')
            label_stat = Counter(cls_idx)
            cls_num = [-1 for _ in range(num_classes)]
            for i in range(num_classes):
                cat_num = int(label_stat[i])
                cls_num[i] = cat_num
            target = cls_num / np.sum(cls_num)
            self.target = torch.tensor(target).to("cuda:{}".format(self.local_rank)).unsqueeze(0)
        else:
            self.target = torch.ones((1, self.num_classes)).to("cuda:{}".format(self.local_rank)) / self.num_classes
        if target_noise != 0:
            self.target = add_noise(self.target, target_noise)

    def forward(self, **input):
        if "type" in input:
            type = input['type']
            del input['type']
        with torch.no_grad():
            outputs = self.classifier(return_loss=False, softmax=False, post_process=False, **input)
            confs = self.temperature * torch.logsumexp(outputs / self.temperature, dim=1)
            softmax_output = self.criterion(outputs)
            targets = self.target
            sim = -softmax_output * targets
            sim = sim.sum(1) / (torch.norm(softmax_output, dim=1) * torch.norm(targets, dim=1))
            confs = confs * sim
        return confs, type

