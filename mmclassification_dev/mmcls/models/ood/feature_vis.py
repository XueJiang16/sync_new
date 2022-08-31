from mmcv.runner import BaseModule
import torch
import os
import numpy as np
from collections import Counter

from ..builder import OOD
from mmcls.models import build_classifier

@OOD.register_module()
class FeatureVis(BaseModule):
    def __init__(self, classifier, num_classes, target_file=None, **kwargs):
        super(Cosine, self).__init__()
        self.local_rank = os.environ['LOCAL_RANK']
        self.classifier = build_classifier(classifier)
        self.classifier.eval()
        self.num_classes = num_classes
        self.criterion = torch.nn.Softmax(dim=-1).to("cuda:{}".format(self.local_rank))

    def forward(self, **input):
        if "type" in input:
            type = input['type']
            del input['type']
        with torch.no_grad():
            outputs = self.classifier(return_loss=False, softmax=False, post_process=False, **input)
            softmax_output = self.criterion(outputs)
            targets = self.target
            sim = -softmax_output * targets
            sim = sim.sum(1) / (torch.norm(softmax_output, dim=1) * torch.norm(targets, dim=1))
            confs = sim
        return confs, type

