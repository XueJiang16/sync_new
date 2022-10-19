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
        self.criterion = torch.nn.Softmax(dim=-1).to("cuda:{}".format(self.local_rank))


    def forward(self, **input):
        if "type" in input:
            type = input['type']
            del input['type']

        with torch.no_grad():
            #  the threshold act diff
            confs_orig = self.classifier(return_loss=False, softmax=False, post_process=False, th_act=False,**input)
            confs_orig = self.criterion(confs_orig)
            # # confs_orig, pred_idx_orig = torch.max(confs_orig, dim=-1)
            # # pred_idx_orig = pred_idx_orig.unsqueeze(1)
            confs_th_act = self.classifier(return_loss=False, softmax=False, post_process=False, th_act=True, **input)
            confs_th_act = self.criterion(confs_th_act)
            # # confs_th_act = torch.gather(confs_th_act, 1, pred_idx_orig).squeeze(1)
            # ood_scores = -torch.abs(confs_orig-confs_th_act).sum(1)
            ood_scores = -torch.nn.functional.kl_div(confs_orig, confs_th_act, reduction='none').mean(1)
            # _, features_orig = self.classifier(return_loss=False, softmax=False, post_process=False,
            #                                    th_act=False,require_features=True, **input)
            # _, features_th_act = self.classifier(return_loss=False, softmax=False, post_process=False,
            #                                      th_act=True, require_features=True,**input)
            # ood_scores = - torch.linalg.norm(features_orig-features_th_act, ord=2, dim=-1)
            #  cosine sim
            # sim = features_orig * features_th_act
            # ood_scores = sim.sum(1) / (torch.norm(features_orig, dim=1) * torch.norm(features_th_act, dim=1))

            #KL
            # ood_scores = -torch.nn.functional.kl_div(features_orig, features_th_act, reduction='none').mean(1)
            return ood_scores, type
