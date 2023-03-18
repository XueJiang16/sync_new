from mmcv.runner import BaseModule
import torch
import torch.nn.functional as F
import os
import numpy as np
from collections import Counter
import json
import faiss

from ..builder import OOD
from mmcls.models import build_classifier
from .utils import add_noise


@OOD.register_module()
class KNN(BaseModule):
    def __init__(self, classifier, dumped_feature, k=1000, **kwargs):
        super(KNN, self).__init__()
        self.local_rank = os.environ['LOCAL_RANK']
        self.classifier = build_classifier(classifier)
        self.classifier.eval()
        self.id_features = torch.load(dumped_feature)
        self.id_features = self.id_features / torch.linalg.norm(self.id_features, dim=-1, keepdim=True)
        self.id_features = self.id_features.numpy()
        self.index = faiss.IndexFlatL2(self.id_features.shape[1])
        self.index.add(self.id_features)
        self.k = k

    def forward(self, **input):
        if "type" in input:
            type = input['type']
            del input['type']
        with torch.no_grad():
            outputs = self.classifier(return_loss=False, softmax=False, post_process=False, **input)
            outputs = outputs / torch.linalg.norm(outputs, dim=-1, keepdim=True)
            outputs = outputs.cpu().numpy()
            D, _ = self.index.search(outputs, self.k)
            scores = -D[:, -1]
            confs = torch.tensor(scores, device="cuda:{}".format(self.local_rank))
        return confs, type
