# Copyright (c) OpenMMLab. All rights reserved.
from .cls_head import ClsHead
from .conformer_head import ConformerHead
from .deit_head import DeiTClsHead
from .linear_head import LinearClsHead
from .multi_label_head import MultiLabelClsHead
from .multi_label_linear_head import MultiLabelLinearClsHead
from .stacked_head import StackedLinearClsHead, DiceStackedLinearClsHead
from .vision_transformer_head import VisionTransformerClsHead
from .dice_head import DiceHead
from .react_head import ReactHead

__all__ = [
    'ClsHead', 'LinearClsHead', 'StackedLinearClsHead', 'MultiLabelClsHead',
    'MultiLabelLinearClsHead', 'VisionTransformerClsHead', 'DeiTClsHead',
    'ConformerHead', 'DiceHead', 'DiceStackedLinearClsHead', 'ReactHead'
]
