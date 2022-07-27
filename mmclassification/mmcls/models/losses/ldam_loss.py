import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..builder import LOSSES
from collections import Counter

@LOSSES.register_module()
class LDAMLoss(nn.Module):
    def __init__(self, meta_file, num_classes=1000, max_m=0.5, weight=None, s=30, **kwargs):
        super(LDAMLoss, self).__init__()
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

        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target, avg_factor):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        loss = F.cross_entropy(self.s * output, target, weight=self.weight, reduction='none')
        # loss = F.cross_entropy(self.s * output, target, weight=None, reduction='none')
        loss = loss.sum() / avg_factor
        return loss