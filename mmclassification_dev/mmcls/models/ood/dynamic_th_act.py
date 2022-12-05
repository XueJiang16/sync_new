from mmcv.runner import BaseModule    # noqa
import torch  # noqa
import torch.nn as nn

import os
import numpy as np  # noqa
from collections import Counter  # noqa

from ..builder import OOD
from mmcls.models import build_classifier, build_ood_model    # noqa

def no_ood_detector(**kwargs):
    raise RuntimeError("No Feature-level OOD Detector Configured!")

@OOD.register_module()
class DynamicThresholdActivation(BaseModule):
    def __init__(self, k=1, ood_detector=None,**kwargs):
        super(DynamicThresholdActivation, self).__init__()
        self.local_rank = os.environ['LOCAL_RANK']
        self.has_ood_detector = True if ood_detector else False
        if self.has_ood_detector:
            self.ood_detector = build_ood_model(ood_detector)
        else:
            self.ood_detector = no_ood_detector
        self.k = k

    def forward(self, **input):
        if "type" in input:
            type = input['type']
            del input['type']

        with torch.no_grad():
            outputs = self.ood_detector.classifier(return_loss=False, softmax=True, post_process=False,
                                                    require_backbone_features=False, th_act=False, **input)
            pred_conf, pred_class = torch.max(outputs,dim=-1)
            update_k = 1.5*self.k*pred_conf+0.5*self.k
            new_outputs = self.ood_detector.classifier(return_loss=False, softmax=True, post_process=False,
                                                    require_backbone_features=False, th_act=update_k, **input)
            # ood_scores = self.ood_detector(**input)
            ood_scores, _ = torch.max(new_outputs)
            return ood_scores, type