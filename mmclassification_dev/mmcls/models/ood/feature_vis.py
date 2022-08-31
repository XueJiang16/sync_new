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
        super(FeatureVis, self).__init__()
        self.local_rank = os.environ['LOCAL_RANK']
        self.classifier = build_classifier(classifier)
        self.classifier.eval()

    def forward(self, **input):
        if "type" in input:
            type = input['type']
            del input['type']
        with torch.no_grad():
            print(input.keys())
            assert False
            _, C4_features = self.classifier(return_loss=False, softmax=False, post_process=False,
                                             require_backbone_features_idx=0, **input)
        return confs, type

