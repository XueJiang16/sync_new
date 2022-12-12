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
class FeatureReweight(BaseModule):
    def __init__(self, num_crop, img_size, threshold, fuse_const=0,k=1, order=1, ood_detector=None, mode='cosine',**kwargs):
        super(FeatureReweight, self).__init__()
        self.local_rank = os.environ['LOCAL_RANK']
        self.has_ood_detector = True if ood_detector else False
        if self.has_ood_detector:
            self.ood_detector = build_ood_model(ood_detector)
        else:
            self.ood_detector = no_ood_detector
        self.num_crop = num_crop
        self.img_size = img_size
        self.threshold = threshold
        self.order = order
        self.mode = mode
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.k = k
        # self.fuse_const=fuse_const


    def kap(self, x):
        b, c, h, w = x.shape
        x_gap = self.gap(x).view(b, c)

        ## kap
        x = x.view(b, c, -1)
        ## h*w -> top k
        num = int(self.k * (h * w))
        # num = int(self.k * h)
        topk_v, _ = x.topk(num, dim=-1)
        out = topk_v.mean(dim=-1)
        # topk_v, _ = out.topk(num, dim=-1)
        # out = topk_v.mean(dim=-1)
        mean_gap = x_gap.mean(dim=-1)
        mean_kap = out.mean(dim=-1)
        out = out * (mean_gap / mean_kap).unsqueeze(-1)
        return out

    def forward(self, **input):
        if "type" in input:
            type = input['type']
            del input['type']

        with torch.no_grad():
            _, feature_c5 = self.ood_detector.classifier(return_loss=False, softmax=False, post_process=False,
                                                         require_backbone_features=True, **input)
            input['type'] = type
            if self.mode == 'kap':
                channel_mean = self.kap(feature_c5).unsqueeze(-1)
                channel_sim = torch.abs(feature_c5.flatten(2) - channel_mean).mean(dim=-1)  # for ID: .mean(dim=-2)
                print("mean:{}, std:{}".format(channel_sim[0].mean(), channel_sim[0].std()))
            else:
                raise NotImplementedError

        return channel_sim.mean(-1), type
