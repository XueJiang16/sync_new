from mmcv.runner import BaseModule
import torch
import os
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import shutil

from ..builder import OOD
from mmcls.models import build_classifier

@OOD.register_module()
class FeatureVis(BaseModule):
    def __init__(self, classifier, num_classes, target_file=None, **kwargs):
        super(FeatureVis, self).__init__()
        self.local_rank = os.environ['LOCAL_RANK']
        self.classifier = build_classifier(classifier)
        self.classifier.eval()

    def forward(self, **input):
        if "type" in input:
            type = input['type']
            del input['type']
        with torch.no_grad():

            filenames = [x['filename'] for x in input['img_metas']]
            if 'ood_data' in filenames[0]:
                mid_path = 'OOD'
            else:
                mid_path = 'ID'
            out_dir = os.path.join('./vis_features/', mid_path)
            os.makedirs(out_dir, exist_ok=True)
            _, C4_features = self.classifier(return_loss=False, softmax=False, post_process=False,
                                             require_backbone_features_idx=0, **input)
            k = 0.1
            C4_features[C4_features<k] = 0
            C4_features[C4_features>=k] = 1
            C4_features = C4_features.mean(1).cpu().numpy()
            C4_features[:,0,0] = 0
            C4_features[:,-1,-1] = 1
            for i in range(len(filenames)):
                plt.matshow(C4_features[i])
                filename = os.path.splitext(os.path.basename(filenames[i]))[0]
                plt.savefig(os.path.join(out_dir, '{}_heatmap.jpg'.format(filename)))
                plt.close()
                shutil.copy(filenames[i], out_dir)
            confs = [0]*len(filenames)
        return confs, type

