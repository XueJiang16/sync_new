from mmcv.runner import BaseModule
import torch
import torch.nn.functional as F
import os
import numpy as np
from collections import Counter
import json

from ..builder import OOD
from mmcls.models import build_classifier
from .utils import add_noise


@OOD.register_module()
class KNN(BaseModule):
    def __init__(self, classifier, dumped_feature, k, **kwargs):
        super(KNN, self).__init__()
        self.local_rank = os.environ['LOCAL_RANK']
        self.classifier = build_classifier(classifier)
        self.classifier.eval()
        self.id_features = torch.load(dumped_feature)



    def forward(self, **input):
        if "type" in input:
            type = input['type']
            del input['type']
        with torch.no_grad():
            outputs = self.classifier(return_loss=False, softmax=False, post_process=False, **input)
            confs = self.temperature * torch.logsumexp(outputs / self.temperature, dim=1)
        return confs, type
