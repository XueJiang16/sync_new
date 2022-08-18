from mmcv.runner import BaseModule    # noqa
import torch  # noqa
import os
import numpy as np  # noqa
from collections import Counter  # noqa

from ..builder import OOD
from mmcls.models import build_classifier, build_ood_model    # noqa


def no_ood_detector(**kwargs):
    raise RuntimeError("No Feature-level OOD Detector Configured!")


@OOD.register_module()
class ThresholdActivation(BaseModule):
    def __init__(self, classifier, **kwargs):
        super(ThresholdActivation, self).__init__()
        self.local_rank = os.environ['LOCAL_RANK']
        self.classifier = build_classifier(classifier)
        self.classifier.eval()


    def forward(self, **input):
        if "type" in input:
            type = input['type']
            del input['type']

        with torch.no_grad():
            confs_orig = self.classifier(return_loss=False, softmax=False, post_process=False, **input)
            confs_th_act = self.classifier(return_loss=False, softmax=False, post_process=False, th_act=True, **input)
            ood_scores = -torch.abs(confs_orig-confs_th_act).sum(1)
        return ood_scores, type
