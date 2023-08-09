from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_postprocessor import BasePostprocessor


class FSPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(FSPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.thred = self.args.thred # search hyper
        self.args_dict = self.config.postprocessor.postprocessor_sweep

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        feature5, output = net.forward_ta(data, ta_thred=0.2, threshold=1)
        _, pred = torch.max(output, dim=1)
        energyconf = torch.logsumexp(output.data.cpu(), dim=1)

        # FS
        feature_crops = feature5.flatten(2)
        patch_mean = feature_crops.mean(-1).unsqueeze(-1)  # (N, C, H*W) -> (N, C)
        fs_conf = torch.abs(feature_crops - patch_mean).mean(dim=(-1, -2))  # for ID: .mean(dim=-2)
        fs_conf = fs_conf.data.cpu()
        conf = fs_conf + 0.0028 * energyconf
        return pred, conf

    def set_hyperparam(self, hyperparam: list):
        self.thred = hyperparam[0]

    def get_hyperparam(self):
        return self.thred
