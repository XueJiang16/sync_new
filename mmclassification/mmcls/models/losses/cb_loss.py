import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from collections import Counter

from ..builder import LOSSES


@LOSSES.register_module()
class CBLoss(nn.Module):
    def __init__(self, meta_file, num_classes=1000, beta=0.9999, **kwargs):
        super(CBLoss, self).__init__()
        cls_idx = []
        with open(meta_file, 'r') as f:
            for line in f.readlines():
                segs = line.strip().split(' ')
                cls_idx.append(int(segs[-1]))
        cls_idx = np.array(cls_idx, dtype='int')
        label_stat = Counter(cls_idx)
        cls_num_list = [-1 for _ in range(num_classes)]
        for i in range(num_classes):
            cat_num = int(label_stat[i])
            cls_num_list[i] = cat_num
        self.cls_num_list = cls_num_list
        self.local_rank = os.environ['LOCAL_RANK']
        effective_num = 1.0 - np.power(beta, self.cls_num_list)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.cls_num_list)
        self.weight = torch.FloatTensor(per_cls_weights).to("cuda:{}".format(self.local_rank))

    def forward(self, x, target, avg_factor):
        loss = F.cross_entropy(x, target, weight=self.weight, reduction='none')
        loss = loss.sum() / avg_factor
        return loss