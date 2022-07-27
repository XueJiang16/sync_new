from mmcv.runner import BaseModule
import torch
import os
import numpy as np
from collections import Counter

from ..builder import OOD
from mmcls.models import build_classifier


@OOD.register_module()
class AugContrast(BaseModule):
    def __init__(self, classifier, num_classes, **kwargs):
        super(AugContrast, self).__init__()
        self.local_rank = os.environ['LOCAL_RANK']
        self.classifier = build_classifier(classifier)
        self.classifier.eval()
        self.num_classes = num_classes

    def forward(self, **input):
        if "type" in input:
            type = input['type']
            del input['type']
        with torch.no_grad():
            outputs_orig, features_orig = self.classifier(return_loss=False, softmax=False, post_process=False,
                                                          require_features=True, **input)
            input["img"] = input["img"] + (torch.rand_like(input["img"])-0.5) / 4
            outputs_aug, features_aug = self.classifier(return_loss=False, softmax=False, post_process=False,
                                                        require_features=True, **input)
            confs = -torch.abs(outputs_orig - outputs_aug).sum(-1)
        return confs, type