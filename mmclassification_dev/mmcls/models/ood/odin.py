from mmcv.runner import BaseModule
import torch
import os
import numpy as np
from collections import Counter

from ..builder import OOD
from mmcls.models import build_classifier
from .utils import add_noise


@OOD.register_module()
class ODIN(BaseModule):
    def __init__(self, classifier, num_classes, temperature=1000, epsilon=0, **kwargs):
        super(ODIN, self).__init__()
        self.local_rank = os.environ['LOCAL_RANK']
        self.classifier = build_classifier(classifier)
        self.classifier.eval()
        self.num_classes = num_classes
        self.temperature = temperature
        self.epsilon = epsilon
        self.criterion = torch.nn.CrossEntropyLoss().to("cuda:{}".format(self.local_rank))


    def forward(self, **input):
        if "type" in input:
            type = input['type']
            del input['type']
        x = input['img'].requires_grad_(True)
        self.classifier.zero_grad()
        outputs = self.classifier(return_loss=False, softmax=False, post_process=False, **input)
        maxIndexTemp = torch.argmax(outputs, dim=1)
        outputs = outputs / self.temperature
        labels = maxIndexTemp.to(torch.long)
        loss = self.criterion(outputs, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(x.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        # Adding small perturbations to images
        with torch.no_grad():
            tempInputs = torch.add(x.data, -self.epsilon, gradient)
            input['img'] = tempInputs
            outputs = self.classifier(return_loss=False, softmax=False, post_process=False, **input)
            outputs = outputs / self.temperature

            # Calculating the confidence after adding perturbations
            nnOutputs = outputs
            nnOutputs = nnOutputs - torch.max(nnOutputs, dim=1, keepdim=True)[0]
            nnOutputs = torch.exp(nnOutputs) / torch.sum(torch.exp(nnOutputs), dim=1, keepdim=True)
            confs, _ = torch.max(nnOutputs, dim=1)
            confs = confs.detach().clone()
        return confs, type

@OOD.register_module()
class ODINCustom(BaseModule):
    def __init__(self, classifier, num_classes, temperature=1000, epsilon=0, target_file=None, target_noise=0,**kwargs):
        super(ODINCustom, self).__init__()
        self.local_rank = os.environ['LOCAL_RANK']
        self.classifier = build_classifier(classifier)
        self.classifier.eval()
        self.num_classes = num_classes
        self.temperature = temperature
        self.epsilon = epsilon
        self.criterion = torch.nn.CrossEntropyLoss().to("cuda:{}".format(self.local_rank))
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
        x = input['img'].requires_grad_(True)
        self.classifier.zero_grad()
        outputs = self.classifier(return_loss=False, softmax=False, post_process=False, **input)
        maxIndexTemp = torch.argmax(outputs, dim=1)
        outputs = outputs / self.temperature
        labels = maxIndexTemp.to(torch.long)
        loss = self.criterion(outputs, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(x.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        # Adding small perturbations to images
        with torch.no_grad():
            tempInputs = torch.add(x.data, -self.epsilon, gradient)
            input['img'] = tempInputs
            outputs = self.classifier(return_loss=False, softmax=False, post_process=False, **input)
            outputs = outputs / self.temperature

            # Calculating the confidence after adding perturbations
            nnOutputs = outputs
            nnOutputs = nnOutputs - torch.max(nnOutputs, dim=1, keepdim=True)[0]
            nnOutputs = torch.exp(nnOutputs) / torch.sum(torch.exp(nnOutputs), dim=1, keepdim=True)

            targets = self.target
            out_softmax = torch.nn.functional.softmax(outputs, dim=1)
            sim = - out_softmax * targets
            sim = torch.sum(sim, dim=1) / (torch.norm(out_softmax, dim=1) * torch.norm(targets, dim=1))
            sim = sim.unsqueeze(1)
            nnOutputs = sim * nnOutputs
            confs, _ = torch.max(nnOutputs, dim=1)
            confs = confs.detach().clone()
        return confs, type




