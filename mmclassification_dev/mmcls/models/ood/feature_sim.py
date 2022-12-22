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
    def __init__(self, ood_detector=None, mode='mean',**kwargs):
        super(FeatureReweight, self).__init__()
        self.local_rank = os.environ['LOCAL_RANK']
        self.has_ood_detector = True if ood_detector else False
        if self.has_ood_detector:
            self.ood_detector = build_ood_model(ood_detector)
        else:
            self.ood_detector = no_ood_detector
        self.mode = mode


    def forward(self, **input):
        if "type" in input:
            type = input['type']
            del input['type']

        with torch.no_grad():
            _, feature_c5 = self.ood_detector.classifier(return_loss=False, softmax=False, post_process=False,
                                                         require_backbone_features=True, **input)
            input['type'] = type
            if self.mode == 'mean':
                feature_crops = feature_c5.flatten(2)
                # feature_crops = feature_crops[:,::4].contiguous()
                # value, index = feature_crops.mean(-1).max(dim=-1)  # (N, C, H*W) -> (N, C)
                # patch_sim = torch.zeros_like(value)
                # for i,j in enumerate(index):
                #     patch_sim[i] = torch.abs(feature_crops[i,j] - value[i]).mean(dim=-1)
                patch_mean = feature_crops.mean(-1).unsqueeze(-1)  # (N, C, H*W) -> (N, C)
                # patch_sim = torch.abs(feature_crops - patch_mean).mean(dim=(-1, -2))  # for ID: .mean(dim=-2)
                patch_sim = torch.clamp(feature_crops - patch_mean, min=0).mean(dim=(-1, -2))
                # feature_crops[(feature_crops-patch_mean) < 0] = 0
                # patch_sim = feature_crops.mean(dim=(-1,-2))
            elif self.mode == 'clamp':
                feature_crops = feature_c5.flatten(2)
                feature_crops = torch.clamp(feature_crops, min=0, max=1.2)
                patch_mean = feature_crops.mean(-1).unsqueeze(-1)  # (N, C, H*W) -> (N, C)
                # patch_mean = (0.4 / (feature_crops.std(dim=-1)+0.05)).unsqueeze(-1)
                patch_sim = torch.clamp((feature_crops - patch_mean), min=0, max=0.5).mean(
                    dim=(-1, -2))  # for ID: .mean(dim=-2)
                # patch_sim = torch.abs(feature_crops - patch_mean).mean(dim=(-1, -2))  # for ID: .mean(dim=-2)
            elif self.mode == 'channel_mean':
                feature_crops = feature_c5.flatten(2)
                patch_mean = feature_crops.mean(1).unsqueeze(1)  # (N, C, H*W) -> (N, C)
                patch_sim = torch.abs(feature_crops - patch_mean).mean(dim=(-1, -2))  # for ID: .mean(dim=-2)
            elif self.mode == 'median':
                feature_crops = feature_c5.flatten(2)
                patch_median = feature_crops.median(-1)[0].unsqueeze(-1)  # (N, C, H*W) -> (N, C)
                patch_sim = torch.abs(feature_crops - patch_median).flatten(1).median(dim=-1)[0]  # for ID: .mean(dim=-2)
            elif self.mode =='std':
                patch_sim = feature_c5.std(dim=(-1,-2)).mean(-1)
            else:
                raise NotImplementedError

        return patch_sim, type
