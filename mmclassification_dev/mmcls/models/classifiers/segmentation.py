from mmseg.models.segmentors import EncoderDecoder


# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..builder import CLASSIFIERS, build_backbone, build_head, build_neck
from ..heads import MultiLabelClsHead
from ..utils.augment import Augments
from .base import BaseClassifier


@CLASSIFIERS.register_module()
class ImageSegmentation(BaseClassifier):

    def __init__(self,
                 segmentor,
                 init_cfg=None):
        super(ImageSegmentation, self).__init__(init_cfg)
        self.segmentor = EncoderDecoder(**segmentor)
            
    def forward(self, **input):
        if "type" in input:
            type = input['type']
            del input['type']
        with torch.no_grad():
            import ipdb; ipdb.set_trace()
            outputs = self.segmentor.inference(**input)
            out_softmax = torch.nn.functional.softmax(outputs, dim=1)
            # out_softmax = outputs
            confs, _ = torch.max(out_softmax, dim=-1)
        return confs, type
